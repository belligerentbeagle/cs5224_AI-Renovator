# RoomStyle — Local Test Scripts

Two test scripts live here. One tests Gemini in isolation, one runs the full backend stack end-to-end.

---

## Prerequisites

Fill in `.env` at the project root before running anything:

```env
GEMINI_API_KEY=...
DATABASE_URL=postgresql://<DB_USER>:<DB_PASSWORD>@<RDS_ENDPOINT>:5432/<DB_NAME>?sslmode=require
JWT_SECRET=...
AWS_REGION=ap-southeast-1
S3_BUCKET=roomstyle-cs5224
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Install dependencies:

```bash
uv sync
```

---

## 1. Gemini image generation test (`test_gemini_local.py`)

Tests only the Gemini image generation step. No server, no database, no AWS needed — just the Gemini API key.

**What it does:**
- Reads a room photo and one or more furniture images from disk
- Sends them to Gemini with a style prompt
- Saves the generated room image as `output_<style>.jpg` in the project root

**Run:**

```bash
# Basic — room photo + one furniture item
uv run python scripts/test_gemini_local.py \
  --room scripts/house.jpg \
  --furniture scripts/sofa.png

# With custom style and prompt
uv run python scripts/test_gemini_local.py \
  --room scripts/house.jpg \
  --furniture scripts/sofa.png \
  --style modern \
  --prompt "bright and airy with natural wood tones"

# Multiple furniture items
uv run python scripts/test_gemini_local.py \
  --room scripts/house.jpg \
  --furniture scripts/sofa.png scripts/chair.jpg \
  --style industrial
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--room` | Yes | Path to the room photo (JPEG or PNG) |
| `--furniture` | No | One or more furniture image paths |
| `--style` | No | Interior style, default: `scandinavian` |
| `--prompt` | No | Extra design instruction |

**Expected output:**
```
[room]      house.jpg  (24 KB)
[furniture] sofa.png  (1025 KB)

Calling Gemini (model: gemini-2.5-flash-image, style: scandinavian, furniture: 1) ...
finish_reason: FinishReason.STOP
parts returned: ['IMAGE']

Saved → /path/to/output_scandinavian.jpg  (1800 KB)
SUCCESS — open the file to inspect the generated room image.
```

---

## 2. End-to-end backend test (`test_e2e_local.py`)

Tests the entire backend stack through HTTP — exactly how a real frontend would use it.

**What it does (9 steps):**

| Step | Endpoint | What happens |
|---|---|---|
| 1 | `GET /health` | Confirms the server is up |
| 2 | `POST /auth/register` | Creates a test user (skips if already exists) |
| 3 | `POST /auth/login` | Gets a JWT token, attaches it to all subsequent requests |
| 4 | `POST /projects` | Creates a project with a $2000 budget |
| 5 | `POST /projects/{id}/uploads/presign` | Gets a presigned S3 URL, then uploads `house.jpg` directly to S3 |
| 6 | `GET /products?q=sofa&source=ikea` | Searches IKEA, picks the first product that has an image |
| 7 | `POST /generate/room` | Submits the generation job — returns `202` immediately |
| 8 | `GET /generations/{id}` | Polls every 5 seconds until Gemini finishes (~20–40s) |
| 9 | Prints result | Shows the S3 URL of the generated image |

> **Note:** The test script never imports from `app/`. It talks to the running server over HTTP the same way a real frontend would. All the app logic (RDS, S3, IKEA API, Gemini) runs inside the server.

**How to run:**

```bash
# Terminal 1 — start the server
uv run uvicorn app.main:app --reload

# Terminal 2 — run the test
uv run python scripts/test_e2e_local.py
```

**Expected output:**
```
========================================
  RoomStyle — End-to-End Local Test
========================================

Step 1: Health check
──────────────────────────────────────────────────
  [PASS] Server is running  →  ok

Step 2: Register user
──────────────────────────────────────────────────
  [PASS] Registered new user  →  e2e_test@roomstyle.dev

Step 3: Login
──────────────────────────────────────────────────
  [PASS] JWT token received  →  eyJhbGciOiJIUzI1NiIsInR5cCI6Ik…

Step 4: Create project
──────────────────────────────────────────────────
  [PASS] Project created  →  89a4eb90-...

Step 5: Upload room photo to S3
──────────────────────────────────────────────────
  [PASS] Presigned URL received  →  https://roomstyle-cs5224.s3.amazonaws.com/...
  [PASS] Photo uploaded to S3  →  photo_id=f5e4ad3a-...

Step 6: Search IKEA furniture
──────────────────────────────────────────────────
  [PASS] Selected: STOCKHOLM 2025  →  $1999.0

Step 7: Submit room generation (202 Accepted)
──────────────────────────────────────────────────
  [PASS] Generation job accepted  →  ac29753e-...

Step 8: Polling generation status
──────────────────────────────────────────────────
  (Gemini is generating the image — this takes ~20–40 seconds)
  [  5s] status=pending
  [ 10s] status=pending
  [ 20s] status=done

Step 9: Generation complete
──────────────────────────────────────────────────
  [PASS] Image URL  →  https://roomstyle-cs5224.s3.ap-southeast-1.amazonaws.com/generations/.../output.jpg
  [PASS] Total cost  →  $1999.00
  [PASS] Over budget  →  True

========================================
  ALL STEPS PASSED
========================================
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Cannot reach server` | Server not running | Run `uv run uvicorn app.main:app --reload` in Terminal 1 |
| `Register failed → Internal Server Error` | bcrypt version mismatch | Run `uv sync` — `bcrypt<4.0.0` is pinned in `pyproject.toml` |
| `S3 upload failed` | AWS credentials missing or wrong | Check `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env` |
| `connection refused` on DB | RDS security group blocking your IP | Add your IP to the `roomstyle-db-sg` inbound rules on port 5432 |
| `password authentication failed` | Wrong `DATABASE_URL` credentials | Ensure `DATABASE_URL` in `.env` matches the RDS master user and password |
| `Gemini returned no image` | Billing not enabled on Google AI | Enable billing at [aistudio.google.com](https://aistudio.google.com) |
| `Generation timed out` | Gemini took >120s | Re-run — occasionally slow; or check server logs for errors |

---

## Sample images

| File | Use |
|---|---|
| `scripts/house.jpg` | Empty room — use as the room photo input |
| `scripts/sofa.png` | Baroque white sofa — use as a furniture item |
