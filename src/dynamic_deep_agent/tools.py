"""Tool registry and call-stack aware invocation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .frames import ToolFrame
from .stack import ToolCallStack


@dataclass
class Tool:
    """A named callable the agent can invoke."""

    name: str
    func: Callable[..., Any]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


def tool(
    name: str | None = None,
    *,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator that wraps a function as a :class:`Tool`."""

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or func.__name__,
            func=func,
            description=description or (func.__doc__ or "").strip(),
            parameters=parameters or {},
        )

    return decorator


class ToolRegistry:
    """Named tools plus a LIFO call stack for nested invocations."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.call_stack = ToolCallStack()

    def register(self, tool_obj: Tool) -> Tool:
        if tool_obj.name in self._tools:
            raise ValueError(f"tool already registered: {tool_obj.name}")
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def call(self, name: str, **arguments: Any) -> Any:
        """Push a tool frame, run the tool, then pop (call / return)."""
        tool_obj = self.get(name)
        parent_id = self.call_stack.peek().id if self.call_stack else None
        frame = ToolFrame(name=name, arguments=arguments, parent_id=parent_id)
        self.call_stack.push(frame)
        try:
            result = tool_obj(**arguments)
            frame.mark_returned(result)
            return result
        except Exception as exc:  # noqa: BLE001 — surface to agent loop
            frame.mark_failed(str(exc))
            raise
        finally:
            # Always pop so the call stack mirrors the real call/return path.
            if self.call_stack and self.call_stack.peek().id == frame.id:
                self.call_stack.pop()
