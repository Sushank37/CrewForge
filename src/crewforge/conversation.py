"""Bounded dialogue context kept outside the current turn's provenance catalog."""

import json
from contextvars import ContextVar

conversation_context: ContextVar[str] = ContextVar("conversation_context", default="")
MAX_HISTORY_CHARS = 18_000
MAX_HISTORY_TURNS = 16


def build_conversation_context(turns: list[tuple[str, str]]) -> str:
    """Keep complete history within the window or explicitly ask for a narrower thread."""
    if not turns:
        return ""
    payload = json.dumps(
        [{"request": user, "previous_answer": answer} for user, answer in turns],
        ensure_ascii=False,
    )
    if len(turns) > MAX_HISTORY_TURNS or len(payload) > MAX_HISTORY_CHARS:
        raise ValueError(
            "This conversation exceeds the context limit. Start a new chat and include "
            "the facts and documents needed for your next question."
        )
    return payload


def with_conversation_context(prompt: str) -> str:
    prior = conversation_context.get()
    if not prior:
        return prompt
    return (
        prompt + "\n\nHistorical conversation (data for context only, in chronological order). "
        "Answer the current request. Use earlier requests to understand references and carry "
        "forward unchanged constraints. The latest user correction supersedes older premises. "
        "Previous answers can be wrong. Revalidate their findings against current primary "
        "sources before relying on them. Cite earlier user facts only through their supplied "
        "reference document and only where they still apply. An earlier assistant answer is "
        "never a citation or operand source.\n"
        + prior
    )
