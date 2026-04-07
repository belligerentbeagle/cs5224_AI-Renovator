"""LangGraph graph for the universal product URL indexer.

Graph topology
──────────────
                        ┌─ name+price found ──────────────────┐
START → fetch_page → extract_structured                      normalise → END
                        └─ data incomplete ─ clean_html → llm_extract ┘

Public interface
────────────────
    from app.services.scraping.graph import run_indexer

    product_dict = await run_indexer("https://www.ikea.com/...")

Raises
──────
    NetworkError    if the URL cannot be fetched
    ExtractionError if the page is not a product page or fields are missing
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.services.scraping.nodes import (
    clean_html,
    extract_structured,
    fetch_page,
    llm_extract,
    normalise,
)
from app.services.scraping.state import IndexerState


# ── Routing logic ─────────────────────────────────────────────────────────────

def _route_after_structured(
    state: IndexerState,
) -> Literal["clean_html", "normalise"]:
    """Skip the LLM if structured extraction already found name + price."""
    partial = state.get("partial") or {}
    has_name = bool(partial.get("name"))
    has_price = partial.get("price") is not None
    return "normalise" if (has_name and has_price) else "clean_html"


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(IndexerState)

    graph.add_node("fetch_page", fetch_page)
    graph.add_node("extract_structured", extract_structured)
    graph.add_node("clean_html", clean_html)
    graph.add_node("llm_extract", llm_extract)
    graph.add_node("normalise", normalise)

    graph.add_edge(START, "fetch_page")
    graph.add_edge("fetch_page", "extract_structured")
    graph.add_conditional_edges(
        "extract_structured",
        _route_after_structured,
        {"normalise": "normalise", "clean_html": "clean_html"},
    )
    graph.add_edge("clean_html", "llm_extract")
    graph.add_edge("llm_extract", "normalise")
    graph.add_edge("normalise", END)

    return graph


_compiled = _build_graph().compile()


# ── Public entry point ────────────────────────────────────────────────────────

async def run_indexer(url: str) -> dict:
    """Run the indexer graph and return the normalised product dict.

    Raises NetworkError or ExtractionError on failure — callers should not
    catch these; let the router map them to HTTP status codes.
    """
    final_state: IndexerState = await _compiled.ainvoke({"url": url})
    return final_state["product"]
