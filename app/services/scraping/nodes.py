"""LangGraph node functions for the URL indexer graph.

Each function receives the full IndexerState and returns a partial dict that
LangGraph merges back into the state.  Nodes are pure-ish functions —
side-effects are limited to HTTP calls and the LLM API.

Node order:
  fetch_page → extract_structured → [route] → clean_html → llm_extract → normalise
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.services.scraping.errors import ExtractionError, NetworkError
from app.services.scraping.state import IndexerState

# ── Constants ─────────────────────────────────────────────────────────────────

_SCRAPER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_FETCH_TIMEOUT = httpx.Timeout(connect=10.0, read=25.0, write=5.0, pool=5.0)
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB hard cap
_CLEAN_TEXT_LIMIT = 4_000              # chars sent to LLM

# Tags that add noise without product signal
_NOISE_TAGS = {
    "script", "style", "noscript", "iframe", "nav", "header",
    "footer", "aside", "form", "button", "svg", "picture",
}

# Deterministic namespace for product_id generation
_SCRAPE_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")

# Hostname → source tag
_SOURCE_MAP = {
    "ikea.com": "ikea",
    "taobao.com": "taobao",
    "world.taobao.com": "taobao",
}


# ── LLM schema ────────────────────────────────────────────────────────────────

class _ProductExtraction(BaseModel):
    """Structured output the LLM must return."""

    is_product_page: bool = Field(
        description="True only if this page shows a single purchasable product."
    )
    name: str | None = Field(
        None, description="Full product name / title."
    )
    price: float | None = Field(
        None, description="Numeric price — no currency symbols."
    )
    currency: str | None = Field(
        None, description="ISO 4217 currency code, e.g. USD, SGD, CNY."
    )
    image_url: str | None = Field(
        None, description="Absolute URL of the main product image."
    )
    in_stock: bool = Field(
        True, description="False only if the page explicitly says out-of-stock."
    )


# ── Node 1: fetch_page ────────────────────────────────────────────────────────

async def fetch_page(state: IndexerState) -> dict:
    url: str = state["url"]
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": _SCRAPER_UA},
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.content[:_MAX_RESPONSE_BYTES]
            encoding = response.encoding or "utf-8"
            return {"raw_html": content.decode(encoding, errors="replace")}

    except httpx.TimeoutException as exc:
        raise NetworkError(f"Timed out fetching {url}") from exc
    except httpx.HTTPStatusError as exc:
        raise NetworkError(
            f"HTTP {exc.response.status_code} from {url}"
        ) from exc
    except httpx.RequestError as exc:
        raise NetworkError(f"Network error fetching {url}: {exc}") from exc


# ── Node 2: extract_structured ────────────────────────────────────────────────

def extract_structured(state: IndexerState) -> dict:
    """Pull product data from JSON-LD and OpenGraph tags.

    Returns whatever was found — may be incomplete.  The router node decides
    whether this is sufficient or whether the LLM is needed.
    """
    html: str = state["raw_html"] or ""
    soup = BeautifulSoup(html, "lxml")

    partial: dict[str, Any] = {}

    # ── JSON-LD (highest quality) ────────────────────────────────────────────
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        # Handle @graph wrapper
        if isinstance(data, dict) and "@graph" in data:
            data = next(
                (n for n in data["@graph"] if n.get("@type") == "Product"),
                None,
            )
        elif isinstance(data, list):
            data = next(
                (n for n in data if isinstance(n, dict) and n.get("@type") == "Product"),
                None,
            )

        if not isinstance(data, dict) or data.get("@type") != "Product":
            continue

        partial["name"] = data.get("name")

        # Price can be nested in offers (single dict or list)
        offers = data.get("offers") or {}
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        if isinstance(offers, dict):
            raw_price = offers.get("price") or offers.get("lowPrice")
            try:
                partial["price"] = float(str(raw_price).replace(",", ""))
            except (TypeError, ValueError):
                pass
            partial["currency"] = offers.get("priceCurrency")
            avail = offers.get("availability", "")
            partial["in_stock"] = "OutOfStock" not in avail

        # Image
        image = data.get("image")
        if isinstance(image, list):
            image = image[0]
        if isinstance(image, dict):
            image = image.get("url")
        partial["image_url"] = image

        break  # First valid Product schema wins

    # ── OpenGraph fallback ────────────────────────────────────────────────────
    def og(prop: str) -> str | None:
        tag = soup.find("meta", property=prop)
        return tag["content"] if tag and tag.get("content") else None  # type: ignore[index]

    if not partial.get("name"):
        partial["name"] = og("og:title")
    if not partial.get("image_url"):
        partial["image_url"] = og("og:image")
    if partial.get("price") is None:
        raw = og("product:price:amount") or og("og:price:amount")
        try:
            partial["price"] = float(str(raw).replace(",", "")) if raw else None
        except ValueError:
            pass
    if not partial.get("currency"):
        partial["currency"] = og("product:price:currency") or og("og:price:currency")

    return {"partial": partial}


# ── Node 3: clean_html ────────────────────────────────────────────────────────

def clean_html(state: IndexerState) -> dict:
    """Strip noise from the HTML and return a short readable text for the LLM."""
    html: str = state["raw_html"] or ""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Prefer a focused content region over the whole body
    content = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find(id=re.compile(r"product|content|main", re.I))
        or soup.find(class_=re.compile(r"product|content|main", re.I))
        or soup.body
        or soup
    )

    text = content.get_text(separator="\n", strip=True)  # type: ignore[union-attr]
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return {"cleaned_text": text[:_CLEAN_TEXT_LIMIT]}


# ── Node 4: llm_extract ───────────────────────────────────────────────────────

async def llm_extract(state: IndexerState) -> dict:
    """Call Gemini to extract product fields from cleaned page text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ExtractionError("GEMINI_API_KEY environment variable is not set.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(_ProductExtraction)

    partial: dict = state.get("partial") or {}
    cleaned: str = state.get("cleaned_text") or ""

    # Give the LLM what structured extraction already found so it can confirm
    # or fill in the gaps rather than starting cold.
    known = {k: v for k, v in partial.items() if v is not None}
    known_hint = (
        f"\n\nStructured metadata already extracted (may be partial):\n{json.dumps(known)}"
        if known
        else ""
    )

    prompt = (
        f"You are a product data extractor for a furniture shopping app.\n"
        f"URL: {state['url']}{known_hint}\n\n"
        f"Page text (truncated):\n{cleaned}\n\n"
        "Extract the product details. "
        "Set is_product_page=false if this is a homepage, category page, or blog post."
    )

    result: _ProductExtraction = await structured_llm.ainvoke(prompt)  # type: ignore[assignment]

    if not result.is_product_page:
        raise ExtractionError(
            "The URL does not appear to be a product page. "
            "Please provide a direct link to a single furniture item."
        )

    return {
        "extracted": {
            "name": result.name,
            "price": result.price,
            "currency": result.currency,
            "image_url": result.image_url,
            "in_stock": result.in_stock,
        }
    }


# ── Node 5: normalise ─────────────────────────────────────────────────────────

def normalise(state: IndexerState) -> dict:
    """Merge partial + extracted, validate required fields, build ScrapedProduct dict."""
    # extracted is set by llm_extract; partial is set by extract_structured.
    # If LLM ran, extracted takes precedence.  If LLM was skipped, promote partial.
    data: dict = {**(state.get("partial") or {}), **(state.get("extracted") or {})}

    name: str | None = data.get("name")
    price_raw = data.get("price")

    if not name:
        raise ExtractionError("Could not extract a product name from this page.")
    try:
        price = float(price_raw) if price_raw is not None else None
    except (TypeError, ValueError):
        price = None
    if price is None:
        raise ExtractionError("Could not extract a product price from this page.")

    url: str = state["url"]
    hostname = urlparse(url).hostname or ""
    source = next(
        (tag for domain, tag in _SOURCE_MAP.items() if hostname.endswith(domain)),
        "scraped",
    )

    product_id = str(uuid.uuid5(_SCRAPE_NS, url))

    return {
        "product": {
            "product_id": product_id,
            "name": name,
            "price": price,
            "currency": data.get("currency", "USD"),
            "source": source,
            "image_url": data.get("image_url"),
            "buy_url": url,
            "in_stock": data.get("in_stock", True),
            "style_tags": [],
            "scraped": True,
        }
    }
