#!/usr/bin/env python3
"""End-to-end local test for the full RoomStyle backend flow.

Tests every phase in sequence:
  1.  Health check              (server is up)
  2.  Register user             (Phase 2 — auth)
  3.  Login                     (Phase 2 — auth)
  4.  Create project            (Phase 2 — projects)
  5.  Presign + upload photo    (Phase 2 — S3 presigned URL)
  6.  Search IKEA furniture     (Phase 2 — product search)
  7.  Submit generation         (Phase 3 — Gemini pipeline)
  8.  Poll until done           (Phase 3 — async result)
  9.  Print image URL           (Phase 3 — success)

Prerequisites
─────────────
  Terminal 1 — start the server:
      uv run uvicorn app.main:app --reload

  Terminal 2 — run this script:
      uv run python scripts/test_e2e_local.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = "http://localhost:8000"
ROOM_IMAGE = Path(__file__).parent / "house.jpg"
TEST_EMAIL = "e2e_test@roomstyle.dev"
TEST_PASSWORD = "TestPass123!"
STYLE = "scandinavian"

# ── Helpers ───────────────────────────────────────────────────────────────────

def ok(label: str, value: str = "") -> None:
    print(f"  [PASS] {label}" + (f"  →  {value}" if value else ""))

def fail(label: str, detail: str) -> None:
    print(f"  [FAIL] {label}  →  {detail}")
    sys.exit(1)

def step(n: int, title: str) -> None:
    print(f"\nStep {n}: {title}")
    print("─" * 50)

# ── Test runner ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\n========================================")
    print("  RoomStyle — End-to-End Local Test")
    print("========================================")

    client = httpx.Client(base_url=BASE, timeout=60.0)
    token: str = ""
    project_id: str = ""
    photo_id: str = ""
    furniture: list[dict] = []

    # ── Step 1: Health check ──────────────────────────────────────────────
    step(1, "Health check")
    try:
        r = client.get("/health")
        r.raise_for_status()
        ok("Server is running", r.json().get("status", ""))
    except httpx.ConnectError:
        fail("Cannot reach server", f"Is it running?  →  uv run uvicorn app.main:app --reload")

    # ── Step 2: Register (ok if already exists) ───────────────────────────
    step(2, "Register user")
    r = client.post("/auth/register", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
        "name": "E2E Test User",
    })
    if r.status_code == 201:
        ok("Registered new user", TEST_EMAIL)
    elif r.status_code == 409:
        ok("User already exists — continuing", TEST_EMAIL)
    else:
        fail("Register failed", r.text)

    # ── Step 3: Login ─────────────────────────────────────────────────────
    step(3, "Login")
    r = client.post("/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    if r.status_code != 200:
        fail("Login failed", r.text)
    token = r.json()["token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    ok("JWT token received", token[:30] + "…")

    # ── Step 4: Create project ────────────────────────────────────────────
    step(4, "Create project")
    r = client.post("/projects", json={
        "title": "E2E Test Room",
        "room_type": "living room",
        "style_prompt": STYLE,
        "budget_limit": 2000.0,
    })
    if r.status_code != 201:
        fail("Create project failed", r.text)
    project_id = r.json()["project_id"]
    ok("Project created", project_id)

    # ── Step 5: Presign upload URL + PUT image to S3 ──────────────────────
    step(5, "Upload room photo to S3")
    if not ROOM_IMAGE.exists():
        fail("Room image not found", str(ROOM_IMAGE))

    r = client.post(f"/projects/{project_id}/uploads/presign", json={
        "file_name": "house.jpg",
        "content_type": "image/jpeg",
    })
    if r.status_code != 200:
        fail("Presign request failed", r.text)

    presign_data = r.json()
    photo_id = presign_data["photo_id"]
    upload_url = presign_data["upload_url"]
    ok("Presigned URL received", upload_url[:60] + "…")

    # Upload directly to S3 (no auth header — presigned URL is self-contained)
    image_bytes = ROOM_IMAGE.read_bytes()
    put_r = httpx.put(upload_url, content=image_bytes, headers={"Content-Type": "image/jpeg"}, timeout=30.0)
    if put_r.status_code not in (200, 204):
        fail("S3 upload failed", f"HTTP {put_r.status_code} — {put_r.text[:200]}")
    ok("Photo uploaded to S3", f"photo_id={photo_id}")

    # ── Step 6: Search IKEA for furniture ─────────────────────────────────
    step(6, "Search IKEA furniture")
    r = client.get("/products", params={"q": "sofa", "source": "ikea"}, timeout=30.0)
    if r.status_code != 200:
        fail("Product search failed", r.text)

    products = r.json()
    if not products:
        fail("No products returned", "IKEA search returned empty list")

    # Pick only the first product that has an image_url
    furniture = next(
        (
            [{
                "name": p["name"],
                "image_url": p.get("image_url"),
                "product_id": p["product_id"],
                "price": p.get("price", 0),
                "source": p.get("source", "ikea"),
                "buy_url": p.get("buy_url"),
            }]
            for p in products if p.get("image_url")
        ),
        [],
    )

    if not furniture:
        fail("No products with images", "Cannot proceed without furniture image_urls")

    ok(f"Selected: {furniture[0]['name'][:50]}", f"${furniture[0]['price']}")

    # ── Step 7: Submit generation ─────────────────────────────────────────
    step(7, "Submit room generation (202 Accepted)")
    r = client.post("/generate/room", json={
        "project_id": project_id,
        "photo_id": photo_id,
        "furniture": furniture,
        "style_name": STYLE,
        "prompt_text": "bright and airy with natural wood tones",
    })
    if r.status_code != 202:
        fail("Generation submit failed", r.text)

    generation_id = r.json()["generation_id"]
    ok("Generation job accepted", generation_id)

    # ── Step 8: Poll until done ───────────────────────────────────────────
    step(8, "Polling generation status")
    print("  (Gemini is generating the image — this takes ~20–40 seconds)")

    max_wait = 120   # seconds
    interval = 5
    elapsed = 0

    while elapsed < max_wait:
        time.sleep(interval)
        elapsed += interval

        r = client.get(f"/generations/{generation_id}")
        if r.status_code == 500:
            fail("Generation failed on server", r.json().get("detail", r.text))

        data = r.json()
        status = data.get("status")
        print(f"  [{elapsed:3d}s] status={status}")

        if status == "done":
            break
    else:
        fail("Timed out", f"Generation did not complete within {max_wait}s")

    # ── Step 9: Print results ─────────────────────────────────────────────
    step(9, "Generation complete")
    image_url = data.get("image_url")
    total_cost = data.get("total_cost", 0)
    over_budget = data.get("over_budget", False)
    products_out = data.get("products", [])

    ok("Image URL", image_url or "(null — check S3 config)")
    ok("Total cost", f"${total_cost:.2f}")
    ok("Over budget", str(over_budget))
    for p in products_out:
        ok(f"  Product: {p['name'][:45]}", f"${p['price']}")

    print("\n========================================")
    print("  ALL STEPS PASSED")
    print("========================================\n")

    if image_url:
        print(f"  Generated image: {image_url}\n")


if __name__ == "__main__":
    main()
