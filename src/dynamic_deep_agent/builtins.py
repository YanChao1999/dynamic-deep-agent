"""Built-in planners / actors for demos and tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .frames import WorkFrame
from .harness import HarnessContext


class StaticPlanner:
    """Planner that always returns the same child list."""

    def __init__(self, children: list[WorkFrame]) -> None:
        self.children = children

    def plan(self, frame: WorkFrame, context: HarnessContext) -> list[WorkFrame]:
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
    """Planner driven by a callable ``(frame, context) -> list[WorkFrame]``."""

    def __init__(
        self, fn: Callable[[WorkFrame, HarnessContext], list[WorkFrame]]
    ) -> None:
        self.fn = fn

    def plan(self, frame: WorkFrame, context: HarnessContext) -> list[WorkFrame]:
        return list(self.fn(frame, context))
