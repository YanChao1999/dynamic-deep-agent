"""Generic LIFO stacks used by the harness.

Two stacks play different roles:

* :class:`WorkStack` — agent planning / execution (push = plan, pop = do)
* :class:`ToolCallStack` — nested tool invocations (call / return)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from .frames import ToolFrame, WorkFrame

T = TypeVar("T")


class StackEmptyError(IndexError):
    """Raised when popping or peeking an empty stack."""


class Stack(Generic[T]):
    """Minimal LIFO stack with iterator and snapshot helpers."""

    def __init__(self, items: Iterable[T] | None = None) -> None:
        self._items: list[T] = list(items) if items is not None else []

    def push(self, item: T) -> None:
        self._items.append(item)

    def push_many(self, items: Iterable[T]) -> None:
        """Push items so the *first* given item ends up on top.

        Planning often produces an ordered list ``[step1, step2, step3]``
        that should run as step1 → step2 → step3. Because we pop from the
        top, we push them in reverse order.
        """
        for item in reversed(list(items)):
            self.push(item)

    def pop(self) -> T:
        if not self._items:
            raise StackEmptyError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise StackEmptyError("peek from empty stack")
        return self._items[-1]

    def clear(self) -> None:
        self._items.clear()

    @property
    def depth(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        return not self._items

    def snapshot(self) -> tuple[T, ...]:
        """Bottom → top view of the stack (immutable)."""
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._items)!r})"


class WorkStack(Stack["WorkFrame"]):
    """Stack of agent work frames. Empty ⇒ task finished."""


class ToolCallStack(Stack["ToolFrame"]):
    """Stack of in-flight tool calls (nested call / return)."""


# Late import for type hints only — avoid circular import at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frames import ToolFrame, WorkFrame
