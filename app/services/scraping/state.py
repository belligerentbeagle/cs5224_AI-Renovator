"""LangGraph state definition for the URL indexer graph."""
from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


class IndexerState(TypedDict, total=False):
    url: str                    # input — always present
    raw_html: str | None        # set by fetch_page
    partial: dict[str, Any]     # set by extract_structured (may be incomplete)
    cleaned_text: str | None    # set by clean_html
    extracted: dict[str, Any]   # set by llm_extract (complete)
    product: dict[str, Any]     # set by normalise — final output
