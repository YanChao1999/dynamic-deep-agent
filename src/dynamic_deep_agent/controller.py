"""RPN-style controller: look at stack + top, choose the next move.

Two moves, like reverse Polish evaluation:

* ``POP_EXECUTE`` — top is ready (operator / value) → pop and run
* ``PLAN_PUSH`` — top needs work → summarize situation, plan, push tokens
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .frames import FrameKind, WorkFrame


class Move(str, Enum):
    """What the agent chooses after inspecting the stack."""

    POP_EXECUTE = "pop_execute"
    PLAN_PUSH = "plan_push"


@dataclass
class Decision:
    """Controller output for one step."""

    move: Move
    summary: str = ""
    frames: list[WorkFrame] = field(default_factory=list)
    consume_top: bool = True
    reason: str = ""

    @classmethod
    def pop_execute(cls, *, reason: str = "") -> Decision:
        return cls(move=Move.POP_EXECUTE, reason=reason)

    @classmethod
    def plan_push(
        cls,
        frames: Sequence[WorkFrame] | None = None,
        *,
        summary: str = "",
        consume_top: bool = True,
        reason: str = "",
    ) -> Decision:
        return cls(
            move=Move.PLAN_PUSH,
            summary=summary,
            frames=list(frames or []),
            consume_top=consume_top,
            reason=reason,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "move": self.move.value,
            "summary": self.summary,
            "frames": [f.as_dict() for f in self.frames],
            "consume_top": self.consume_top,
            "reason": self.reason,
        }


class Controller(Protocol):
    """Inspects stack + top and returns the next RPN-style move."""

    def decide(
        self,
        stack: tuple[WorkFrame, ...],
        top: WorkFrame,
        context: Any,
    ) -> Decision:
        ...


class KindController:
    """Default policy: OP/VALUE → pop+execute, GOAL → summary+plan+push.

    A separate :class:`~dynamic_deep_agent.harness.Planner` may still fill in
    ``frames`` when the decision has an empty plan list.
    """

    def decide(
        self,
        stack: tuple[WorkFrame, ...],
        top: WorkFrame,
        context: Any,
    ) -> Decision:
        if top.kind is FrameKind.OP:
            return Decision.pop_execute(reason="top is OP")
        if top.kind is FrameKind.VALUE:
            return Decision.pop_execute(reason="top is VALUE")
        # GOAL (or unknown): summarize stack and expand.
        summary = _default_summary(stack, top, context)
        return Decision.plan_push(
            summary=summary,
            consume_top=True,
            reason="top is GOAL — summarize, plan, push",
        )


class CallableController:
    """Wrap a ``(stack, top, context) -> Decision`` callable."""

    def __init__(
        self,
        fn: Callable[[tuple[WorkFrame, ...], WorkFrame, Any], Decision],
    ) -> None:
        self.fn = fn

    def decide(
        self,
        stack: tuple[WorkFrame, ...],
        top: WorkFrame,
        context: Any,
    ) -> Decision:
        return self.fn(stack, top, context)


def _default_summary(
    stack: tuple[WorkFrame, ...],
    top: WorkFrame,
    context: Any,
) -> str:
    goal = getattr(context, "goal", "")
    depth = len(stack)
    values = getattr(context, "values", {}) or {}
    parts = [
        f"goal={goal!r}",
        f"depth={depth}",
        f"top={top.kind.value}:{top.description!r}",
    ]
    if values:
        parts.append(f"known={list(values)}")
    below = [f"{f.kind.value}:{f.description}" for f in stack[:-1]]
    if below:
        parts.append(f"below={below}")
    return "; ".join(parts)
