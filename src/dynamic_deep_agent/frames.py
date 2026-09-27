"""Work and tool frames — RPN-style tokens on the work stack.

* ``GOAL`` — needs expansion (agent tends to summary + plan + push)
* ``OP`` — ready to run (agent tends to pop + execute, like an RPN operator)
* ``VALUE`` — operand / intermediate result sitting on the stack
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


def _new_id() -> str:
    return uuid4().hex[:12]


class FrameKind(str, Enum):
    """RPN token kinds on the agent work stack."""

    GOAL = "goal"
    OP = "op"
    VALUE = "value"


@dataclass
class WorkFrame:
    """One token on the agent work stack."""

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


def goal(description: str, **payload: Any) -> WorkFrame:
    """Shorthand for a GOAL frame (needs plan)."""
    return WorkFrame(kind=FrameKind.GOAL, description=description, payload=payload)


def op(description: str, **payload: Any) -> WorkFrame:
    """Shorthand for an OP frame (pop + execute)."""
    return WorkFrame(kind=FrameKind.OP, description=description, payload=payload)


def value(description: str, *, data: Any = None, **payload: Any) -> WorkFrame:
    """Shorthand for a VALUE frame (operand / result on the stack)."""
    body = dict(payload)
    if data is not None:
        body.setdefault("value", data)
    return WorkFrame(kind=FrameKind.VALUE, description=description, payload=body)


# Aliases matching the plan/do vocabulary.
plan = goal
do = op
