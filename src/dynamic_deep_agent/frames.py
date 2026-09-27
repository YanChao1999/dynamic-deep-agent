"""Work and tool frames that live on the harness stacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


def _new_id() -> str:
    return uuid4().hex[:12]


class FrameKind(str, Enum):
    """Kinds of work a frame can represent.

    Mirrors calculator semantics:

    * ``GOAL`` / ``PLAN`` — push a decomposition (like entering an expression)
    * ``ACTION`` — pop and run a concrete step (like an operator)
    * ``VALUE`` — intermediate result left for a later reduce (operand)
    * ``REDUCE`` — combine prior values into one result
    * ``DONE`` — sentinel that a sub-goal finished
    """

    GOAL = "goal"
    PLAN = "plan"
    ACTION = "action"
    VALUE = "value"
    REDUCE = "reduce"
    DONE = "done"


@dataclass
class WorkFrame:
    """One item on the agent work stack."""

    kind: FrameKind
    description: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    parent_id: str | None = None
    status: str = "pending"  # pending | running | done | failed
    result: Any = None

    def mark_running(self) -> None:
        self.status = "running"

    def mark_done(self, result: Any = None) -> None:
        self.status = "done"
        self.result = result

    def mark_failed(self, error: Any) -> None:
        self.status = "failed"
        self.result = error

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "description": self.description,
            "payload": self.payload,
            "parent_id": self.parent_id,
            "status": self.status,
            "result": self.result,
        }


@dataclass
class ToolFrame:
    """One in-flight tool call on the tool-call stack."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    parent_id: str | None = None
    status: str = "running"  # running | returned | failed
    result: Any = None

    def mark_returned(self, result: Any) -> None:
        self.status = "returned"
        self.result = result

    def mark_failed(self, error: Any) -> None:
        self.status = "failed"
        self.result = error

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "parent_id": self.parent_id,
            "status": self.status,
            "result": self.result,
        }
