"""Built-in planners / actors / controllers for demos and tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .controller import Decision
from .frames import WorkFrame
from .harness import HarnessContext


class StaticPlanner:
    """Planner that always returns the same child list."""

    def __init__(self, children: list[WorkFrame]) -> None:
        self.children = children

    def plan(
        self,
        frame: WorkFrame,
        context: HarnessContext,
        *,
        summary: str = "",
        stack: tuple[WorkFrame, ...] = (),
    ) -> list[WorkFrame]:
        planned: list[WorkFrame] = []
        for child in self.children:
            planned.append(
                WorkFrame(
                    kind=child.kind,
                    description=child.description,
                    payload=dict(child.payload),
                    parent_id=frame.id,
                )
            )
        return planned


class EchoActor:
    """Actor that returns the frame description (useful in tests)."""

    def act(self, frame: WorkFrame, context: HarnessContext) -> Any:
        return frame.description


class RulePlanner:
    """Planner driven by a callable.

    Signature::

        (frame, context, *, summary, stack) -> list[WorkFrame]

    A 2-arg ``(frame, context)`` callable is also accepted.
    """

    def __init__(self, fn: Callable[..., list[WorkFrame]]) -> None:
        self.fn = fn

    def plan(
        self,
        frame: WorkFrame,
        context: HarnessContext,
        *,
        summary: str = "",
        stack: tuple[WorkFrame, ...] = (),
    ) -> list[WorkFrame]:
        try:
            return list(self.fn(frame, context, summary=summary, stack=stack))
        except TypeError:
            return list(self.fn(frame, context))


class RuleController:
    """Controller driven by ``(stack, top, context) -> Decision``."""

    def __init__(
        self, fn: Callable[[tuple[WorkFrame, ...], WorkFrame, Any], Decision]
    ) -> None:
        self.fn = fn

    def decide(
        self,
        stack: tuple[WorkFrame, ...],
        top: WorkFrame,
        context: Any,
    ) -> Decision:
        return self.fn(stack, top, context)
