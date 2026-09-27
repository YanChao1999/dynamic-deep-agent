"""LLM-backed RPN controller with stack token summary.

Uses an OpenAI-style chat model to:

1. **Summarize** the current work-stack tokens
2. **Decide** ``pop_execute`` vs ``plan_push``
3. Optionally **emit frames** to push when planning
"""

from __future__ import annotations

import json
import re
from typing import Any

from .controller import Decision, Move
from .frames import FrameKind, WorkFrame, goal, op, value
from .llm import LLMClient, TokenUsage
from .token_summary import format_token_summary_input, render_stack_tokens

SYSTEM_PROMPT = """You are the controller of an RPN-style stack agent.
You see the work stack (program tokens) and its TOP.

Each step choose exactly one move:
- "pop_execute": top is ready (VALUE operand or OP operator) — pop and execute it.
- "plan_push": top is a GOAL (or needs decomposition) — summarize tokens, plan next tokens, push them.

Always produce a concise "summary" of the current stack tokens (what they mean for the goal).
When move is plan_push, also return "frames": a list of tokens to push (first runs next).
Each frame: {"kind": "goal"|"op"|"value", "description": "...", "payload": {...}}
For OP tools use payload like {"tool":"name","args":{...},"arity":0,"store_as":"x"}.
For VALUE use payload {"value": ...}.

Reply with ONLY valid JSON:
{
  "move": "pop_execute" | "plan_push",
  "summary": "token/stack summary",
  "reason": "short reason",
  "consume_top": true,
  "frames": []
}
"""


class LLMController:
    """Controller that asks an OpenAI-style LLM for summary + move."""

    def __init__(
        self,
        client: LLMClient,
        *,
        temperature: float = 0.0,
        system_prompt: str = SYSTEM_PROMPT,
        json_mode: bool = True,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_mode = json_mode

    def decide(
        self,
        stack: tuple[WorkFrame, ...],
        top: WorkFrame,
        context: Any,
    ) -> Decision:
        token_view = format_token_summary_input(stack, top, context)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Inspect these stack tokens and choose the next move.\n\n"
                    f"{token_view}\n\n"
                    f"Raw tokens JSON:\n{json.dumps(render_stack_tokens(stack), ensure_ascii=False)}"
                ),
            },
        ]
        response_format = {"type": "json_object"} if self.json_mode else None
        result = self.client.chat(
            messages,
            temperature=self.temperature,
            response_format=response_format,
        )
        _record_usage(context, result.usage)

        data = _parse_json_object(result.content)
        summary = str(data.get("summary") or "").strip()
        reason = str(data.get("reason") or "").strip()
        move_raw = str(data.get("move") or "").strip().lower()
        consume_top = bool(data.get("consume_top", True))
        frames = _frames_from_payload(data.get("frames") or [])

        # Always keep token summaries on the harness context.
        _append_summary(context, summary, stack=stack, top=top, usage=result.usage)

        if move_raw in ("pop_execute", "pop", "execute"):
            decision = Decision.pop_execute(reason=reason or "llm pop_execute")
            decision.summary = summary
            return decision
        if move_raw in ("plan_push", "plan", "push"):
            return Decision.plan_push(
                frames,
                summary=summary,
                consume_top=consume_top,
                reason=reason or "llm plan_push",
            )
        # Fallback: treat unknown move as plan_push if top looks like a goal.
        if top.kind is FrameKind.GOAL:
            return Decision.plan_push(
                frames,
                summary=summary or reason,
                consume_top=consume_top,
                reason=f"fallback plan_push (raw move={move_raw!r})",
            )
        return Decision.pop_execute(reason=f"fallback pop_execute (raw move={move_raw!r})")


class LLMPlanner:
    """Planner that asks the LLM to expand a GOAL given a token summary."""

    def __init__(
        self,
        client: LLMClient,
        *,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.json_mode = json_mode

    def plan(
        self,
        frame: WorkFrame,
        context: Any,
        *,
        summary: str = "",
        stack: tuple[WorkFrame, ...] = (),
    ) -> list[WorkFrame]:
        token_view = format_token_summary_input(stack or (frame,), frame, context)
        tools = []
        registry = getattr(context, "tools", None)
        if registry is not None and hasattr(registry, "schemas"):
            tools = registry.schemas()

        messages = [
            {
                "role": "system",
                "content": (
                    "You expand a GOAL into stack tokens for an RPN agent. "
                    "Return ONLY JSON: {\"summary\": \"...\", \"frames\": [...]} "
                    "where frames run first-to-last (first becomes TOP). "
                    "Each frame: {\"kind\":\"goal\"|\"op\"|\"value\",\"description\":str,\"payload\":object}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Prior token summary:\n{summary}\n\n"
                    f"Current stack:\n{token_view}\n\n"
                    f"Available tools:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
                    f"Expand GOAL: {frame.description!r} payload={json.dumps(frame.payload, ensure_ascii=False)}"
                ),
            },
        ]
        response_format = {"type": "json_object"} if self.json_mode else None
        result = self.client.chat(
            messages,
            temperature=self.temperature,
            response_format=response_format,
        )
        _record_usage(context, result.usage)
        data = _parse_json_object(result.content)
        plan_summary = str(data.get("summary") or summary).strip()
        _append_summary(
            context,
            plan_summary,
            stack=stack or (frame,),
            top=frame,
            usage=result.usage,
        )
        if plan_summary:
            context.last_summary = plan_summary
        return _frames_from_payload(data.get("frames") or [])


def _record_usage(context: Any, usage: TokenUsage) -> None:
    bucket = getattr(context, "token_usage", None)
    if bucket is None:
        return
    bucket.merge(usage)


def _append_summary(
    context: Any,
    summary: str,
    *,
    stack: tuple[WorkFrame, ...],
    top: WorkFrame,
    usage: TokenUsage,
) -> None:
    if not summary:
        return
    context.last_summary = summary
    summaries = getattr(context, "summaries", None)
    if summaries is None:
        return
    summaries.append(
        {
            "summary": summary,
            "top": {"kind": top.kind.value, "description": top.description, "id": top.id},
            "token_count": len(stack),
            "tokens": render_stack_tokens(stack),
            "usage": usage.as_dict(),
        }
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    raise ValueError(f"LLM did not return a JSON object: {text[:200]!r}")


def _frames_from_payload(items: Any) -> list[WorkFrame]:
    if not isinstance(items, list):
        return []
    frames: list[WorkFrame] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "op").lower()
        description = str(item.get("description") or kind)
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        # Allow flat tool fields on the item itself.
        flat = {
            k: v
            for k, v in item.items()
            if k not in {"kind", "description", "payload", "id", "parent_id"}
        }
        merged = {**flat, **payload}
        if kind in {"goal", "plan"}:
            frames.append(goal(description, **merged))
        elif kind in {"value", "val"}:
            data = merged.pop("value", merged.pop("data", None))
            frames.append(value(description, data=data, **merged))
        else:
            frames.append(op(description, **merged))
    return frames
