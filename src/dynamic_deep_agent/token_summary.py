"""Token summary helpers — render the work stack for the LLM."""

from __future__ import annotations

from typing import Any

from .frames import WorkFrame


def render_stack_tokens(stack: tuple[WorkFrame, ...]) -> list[dict[str, Any]]:
    """Serialize stack frames as ordered tokens (bottom → top)."""
    tokens: list[dict[str, Any]] = []
    for i, frame in enumerate(stack):
        tokens.append(
            {
                "index": i,
                "id": frame.id,
                "kind": frame.kind.value,
                "description": frame.description,
                "payload": frame.payload,
                "is_top": i == len(stack) - 1,
            }
        )
    return tokens


def format_token_summary_input(
    stack: tuple[WorkFrame, ...],
    top: WorkFrame,
    context: Any,
) -> str:
    """Human-readable block describing goal, stack tokens, and known values."""
    lines: list[str] = []
    goal = getattr(context, "goal", "")
    lines.append(f"Goal: {goal}")
    values = getattr(context, "values", {}) or {}
    if values:
        lines.append(f"Known values: {json_dumps(values)}")
    results = getattr(context, "results", None)
    if results:
        lines.append(f"Recent results: {json_dumps(list(results)[-5:])}")

    lines.append("Work stack tokens (bottom → top):")
    tokens = render_stack_tokens(stack)
    if not tokens:
        lines.append("  (empty)")
    for tok in tokens:
        marker = " ← TOP" if tok["is_top"] else ""
        lines.append(
            f"  [{tok['index']}] {tok['kind']}: {tok['description']}"
            f" payload={json_dumps(tok['payload'])}{marker}"
        )
    lines.append(f"Top kind: {top.kind.value}")
    lines.append(f"Top description: {top.description}")
    return "\n".join(lines)


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
