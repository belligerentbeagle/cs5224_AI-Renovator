"""Refinement service.

Interprets a natural-language instruction from the user (e.g. "make it more minimal",
"warmer tones", "replace the sofa") into updated generation params, then creates a new
DesignGeneration record that the caller can hand off to the existing pipeline.
"""

from __future__ import annotations

import os
from typing import TypedDict
from uuid import UUID

from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.orm import DesignGeneration, GenerationStatus
from app.models.schemas import GenerationPending
from app.stores.design_generation_store import DesignGenerationStore

_STYLES = ["Modern", "Scandinavian", "Cozy Warm", "Futuristic", "Nature", "Industrial"]


# ── LangGraph state + output schema ──────────────────────────────────────────

class _RefineState(TypedDict):
    original_style: str
    original_prompt: str
    user_message: str
    refined_style: str
    refined_prompt: str


class _RefinedParams(BaseModel):
    style_name: str
    prompt_text: str


# ── Graph node ────────────────────────────────────────────────────────────────

def _interpret_node(state: _RefineState) -> dict:
    """Call Gemini to interpret the user's instruction into structured params."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.3,
    )
    structured = llm.with_structured_output(_RefinedParams)

    system = (
        f"You are an interior design assistant helping refine an AI-generated room design.\n"
        f"Current design — style: '{state['original_style']}', "
        f"instructions: '{state['original_prompt'] or 'none'}'.\n"
        f"Interpret the user's refinement request and return:\n"
        f"- style_name: one of {_STYLES} (keep current unless user asks to change)\n"
        f"- prompt_text: a concise 1-2 sentence instruction for the image AI. "
        f"Be specific about style, lighting, colour palette, mood, or furniture arrangement."
    )

    result = structured.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": state["user_message"]},
    ])
    return {"refined_style": result.style_name, "refined_prompt": result.prompt_text}


# ── Compiled graph (singleton) ────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(_RefineState)
    g.add_node("interpret", _interpret_node)
    g.set_entry_point("interpret")
    g.add_edge("interpret", END)
    return g.compile()


_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Service ───────────────────────────────────────────────────────────────────

class RefineService:
    """Interprets a refinement message and creates a new pending generation."""

    def submit_refine(
        self,
        generation_id: UUID,
        message: str,
        user_id: UUID,
        db: Session,
    ) -> GenerationPending:
        """Interpret user message and queue a new generation with refined params.

        The caller is responsible for scheduling run_generation_pipeline() as a
        background task using the returned generation_id.

        Args:
            generation_id: The generation the user wants to refine
            message: Natural-language refinement instruction
            user_id: Requesting user (for auth — currently informational)
            db: Database session

        Returns:
            GenerationPending with the new generation_id

        Raises:
            HTTPException 404: If the original generation is not found
        """
        gen_store = DesignGenerationStore(db)
        original = gen_store.get_by_id(generation_id)
        if not original:
            raise HTTPException(status_code=404, detail="Generation not found")

        # Interpret the instruction via LangGraph
        result = _get_graph().invoke({
            "original_style": original.style_name or "Modern",
            "original_prompt": original.prompt_text or "",
            "user_message": message,
            "refined_style": "",
            "refined_prompt": "",
        })

        # New generation — same project + room photo, updated style/prompt
        new_gen = DesignGeneration(
            project_id=original.project_id,
            input_photo_id=original.input_photo_id,
            style_name=result["refined_style"],
            prompt_text=result["refined_prompt"],
            status=GenerationStatus.pending,
        )
        gen_store.add(new_gen)
        db.commit()
        db.refresh(new_gen)

        return GenerationPending(generation_id=new_gen.design_id, status="pending")
