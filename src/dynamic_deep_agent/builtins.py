"""Built-in planners / actors for demos and tests."""

from __future__ import annotations

from typing import Any

from .frames import FrameKind, WorkFrame
from .harness import HarnessContext


class StaticPlanner:
    """Planner that returns a fixed decomposition for any GOAL/PLAN."""

    def __init__(self, children: list[WorkFrame]) -> None:
        self.children = children

    def plan(self, frame: WorkFrame, context: HarnessContext) -> list[WorkFrame]:
        planned = []
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
    """Planner driven by a callable ``goal -> list[WorkFrame]``."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def plan(self, frame: WorkFrame, context: HarnessContext) -> list[WorkFrame]:
        return list(self.fn(frame, context))
