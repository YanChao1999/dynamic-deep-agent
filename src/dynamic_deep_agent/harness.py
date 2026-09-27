"""Stack-based agent harness.

Mental model (like a calculator):

1. **Push** a goal / plan onto the work stack.
2. **Pop** the top frame and execute it.
3. Execution may **push** more frames (decompose) or leave **values**.
4. When the work stack is **empty**, the task is finished.

Push and pop also drive the **tool-call stack** for nested tool invocations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .frames import FrameKind, WorkFrame
from .stack import WorkStack
from .tools import ToolRegistry


class Planner(Protocol):
    """Turns a GOAL/PLAN frame into ordered child work frames."""

    def plan(self, frame: WorkFrame, context: "HarnessContext") -> list[WorkFrame]:
        ...


class Actor(Protocol):
    """Executes an ACTION frame, optionally using tools."""

    def act(self, frame: WorkFrame, context: "HarnessContext") -> Any:
        ...


class Reducer(Protocol):
    """Combines VALUE frames referenced by a REDUCE frame."""

    def reduce(self, frame: WorkFrame, values: list[Any], context: "HarnessContext") -> Any:
        ...


@dataclass
class HarnessConfig:
    """Runtime knobs for the harness loop."""

    max_steps: int = 100
    stop_on_failure: bool = True
    record_trace: bool = True


@dataclass
class HarnessContext:
    """Shared state visible to planners, actors, and reducers."""

    goal: str
    tools: ToolRegistry
    values: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def store_value(self, key: str, value: Any) -> None:
        self.values[key] = value

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass
class HarnessResult:
    """Outcome of a finished (or aborted) harness run."""

    success: bool
    result: Any
    steps: int
    context: HarnessContext
    remaining: tuple[WorkFrame, ...] = ()
    error: str | None = None


class AgentHarness:
    """Plan-by-push / execute-by-pop agent loop."""

    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        planner: Planner | None = None,
        actor: Actor | None = None,
        reducer: Reducer | None = None,
        config: HarnessConfig | None = None,
        on_step: Callable[[WorkFrame, HarnessContext], None] | None = None,
    ) -> None:
        self.work = WorkStack()
        self.tools = tools or ToolRegistry()
        self.planner = planner
        self.actor = actor
        self.reducer = reducer
        self.config = config or HarnessConfig()
        self.on_step = on_step

    def push(self, frame: WorkFrame) -> None:
        """Push one work frame (agent planning / enqueue)."""
        self.work.push(frame)

    def push_plan(self, frames: Iterable[WorkFrame]) -> None:
        """Push an ordered plan so the first frame runs first."""
        self.work.push_many(frames)

    def pop(self) -> WorkFrame:
        """Pop the top work frame (agent starts doing it)."""
        return self.work.pop()

    def run(self, goal: str, *, initial_plan: Iterable[WorkFrame] | None = None) -> HarnessResult:
        """Push the goal (and optional plan), then pop/execute until empty."""
        context = HarnessContext(goal=goal, tools=self.tools)
        root = WorkFrame(kind=FrameKind.GOAL, description=goal)
        self.work.clear()
        self.push(root)
        if initial_plan is not None:
            # Replace GOAL with the provided plan (already decomposed).
            self.pop()
            self.push_plan(initial_plan)

        steps = 0
        last_result: Any = None

        while not self.work.empty():
            if steps >= self.config.max_steps:
                return HarnessResult(
                    success=False,
                    result=last_result,
                    steps=steps,
                    context=context,
                    remaining=self.work.snapshot(),
                    error=f"max_steps ({self.config.max_steps}) exceeded",
                )

            frame = self.pop()
            frame.mark_running()
            steps += 1
            if self.on_step:
                self.on_step(frame, context)

            try:
                last_result = self._dispatch(frame, context)
                frame.mark_done(last_result)
            except Exception as exc:  # noqa: BLE001
                frame.mark_failed(str(exc))
                if self.config.record_trace:
                    context.trace.append({"event": "error", "frame": frame.as_dict(), "error": str(exc)})
                if self.config.stop_on_failure:
                    return HarnessResult(
                        success=False,
                        result=None,
                        steps=steps,
                        context=context,
                        remaining=self.work.snapshot(),
                        error=str(exc),
                    )
                continue

            if self.config.record_trace:
                context.trace.append(
                    {
                        "event": "step",
                        "step": steps,
                        "frame": frame.as_dict(),
                        "stack_depth": self.work.depth,
                        "tool_depth": self.tools.call_stack.depth,
                    }
                )

        return HarnessResult(
            success=True,
            result=last_result,
            steps=steps,
            context=context,
            remaining=(),
        )

    def _dispatch(self, frame: WorkFrame, context: HarnessContext) -> Any:
        kind = frame.kind

        if kind in (FrameKind.GOAL, FrameKind.PLAN):
            return self._handle_plan(frame, context)
        if kind is FrameKind.ACTION:
            return self._handle_action(frame, context)
        if kind is FrameKind.VALUE:
            key = frame.payload.get("key", frame.id)
            context.store_value(key, frame.payload.get("value", frame.result))
            return context.get_value(key)
        if kind is FrameKind.REDUCE:
            return self._handle_reduce(frame, context)
        if kind is FrameKind.DONE:
            return frame.payload.get("value", frame.result)
        raise ValueError(f"unsupported frame kind: {kind}")

    def _handle_plan(self, frame: WorkFrame, context: HarnessContext) -> Any:
        if self.planner is None:
            raise RuntimeError("no planner configured for GOAL/PLAN frames")
        children = self.planner.plan(frame, context)
        for child in children:
            if child.parent_id is None:
                child.parent_id = frame.id
        if children:
            self.push_plan(children)
        return {"planned": [c.id for c in children]}

    def _handle_action(self, frame: WorkFrame, context: HarnessContext) -> Any:
        # Prefer an explicit tool name in the payload when present.
        tool_name = frame.payload.get("tool")
        if tool_name:
            args = dict(frame.payload.get("args") or {})
            result = context.tools.call(tool_name, **args)
            store_as = frame.payload.get("store_as")
            if store_as:
                context.store_value(store_as, result)
            return result
        if self.actor is None:
            raise RuntimeError(
                "ACTION frame has no 'tool' payload and no actor is configured"
            )
        result = self.actor.act(frame, context)
        store_as = frame.payload.get("store_as")
        if store_as:
            context.store_value(store_as, result)
        return result

    def _handle_reduce(self, frame: WorkFrame, context: HarnessContext) -> Any:
        keys = list(frame.payload.get("keys") or [])
        values = [context.get_value(k) for k in keys]
        if self.reducer is not None:
            result = self.reducer.reduce(frame, values, context)
        else:
            op = frame.payload.get("op", "identity")
            result = _default_reduce(op, values, frame.payload)
        store_as = frame.payload.get("store_as")
        if store_as:
            context.store_value(store_as, result)
        return result


def _default_reduce(op: str, values: list[Any], payload: dict[str, Any]) -> Any:
    if op == "identity":
        return values[0] if len(values) == 1 else values
    if op == "sum":
        return sum(values)
    if op == "product":
        out = 1
        for v in values:
            out *= v
        return out
    if op == "concat":
        sep = payload.get("sep", "")
        return sep.join(str(v) for v in values)
    if op == "dict":
        return dict(zip(payload.get("fields") or keys_as_fields(values), values, strict=False))
    raise ValueError(f"unknown reduce op: {op}")


def keys_as_fields(values: list[Any]) -> list[str]:
    return [f"v{i}" for i in range(len(values))]
