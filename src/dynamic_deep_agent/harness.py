"""Stack-based DFS agent harness.

Mental model:

1. **Push** a goal (PLAN) onto the work stack.
2. **Pop** the top frame.
3. If it is **PLAN** → decompose and **push** children (go deeper — DFS).
4. If it is **DO** → run the tool/actor and keep the result.
5. Repeat until the stack is **empty** → that last result is the answer.

Push/pop also drive the **tool-call stack** for nested tool invocations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .frames import FrameKind, WorkFrame, plan as plan_frame
from .stack import WorkStack
from .tools import ToolRegistry


class Planner(Protocol):
    """Turns a PLAN frame into ordered child frames (DFS: first child runs next)."""

    def plan(self, frame: WorkFrame, context: "HarnessContext") -> list[WorkFrame]:
        ...


class Actor(Protocol):
    """Executes a DO frame when no explicit tool is set in the payload."""

    def act(self, frame: WorkFrame, context: "HarnessContext") -> Any:
        ...


@dataclass
class HarnessConfig:
    """Runtime knobs for the harness loop."""

    max_steps: int = 100
    stop_on_failure: bool = True
    record_trace: bool = True


@dataclass
class HarnessContext:
    """Shared state visible to planners and actors."""

    goal: str
    tools: ToolRegistry
    values: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)
    results: list[Any] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def store_value(self, key: str, value: Any) -> None:
        self.values[key] = value

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def resolve_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve ``$name`` placeholders from ``context.values``."""
        out: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                out[key] = self.get_value(value[1:])
            elif isinstance(value, list):
                out[key] = [
                    self.get_value(item[1:], item)
                    if isinstance(item, str) and item.startswith("$")
                    else item
                    for item in value
                ]
            else:
                out[key] = value
        return out


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
    """DFS agent loop: plan → push, do → pop, until the stack is empty."""

    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        planner: Planner | None = None,
        actor: Actor | None = None,
        config: HarnessConfig | None = None,
        on_step: Callable[[WorkFrame, HarnessContext], None] | None = None,
    ) -> None:
        self.work = WorkStack()
        self.tools = tools or ToolRegistry()
        self.planner = planner
        self.actor = actor
        self.config = config or HarnessConfig()
        self.on_step = on_step

    def push(self, frame: WorkFrame) -> None:
        """Push one work frame (go deeper / enqueue)."""
        self.work.push(frame)

    def push_plan(self, frames: Iterable[WorkFrame]) -> None:
        """Push children so the *first* runs next (DFS order)."""
        self.work.push_many(frames)

    def pop(self) -> WorkFrame:
        """Pop the top work frame and start doing it."""
        return self.work.pop()

    def run(self, goal: str, *, initial_plan: Iterable[WorkFrame] | None = None) -> HarnessResult:
        """DFS until the work stack is empty; return the last DO result."""
        context = HarnessContext(goal=goal, tools=self.tools)
        self.work.clear()

        if initial_plan is not None:
            self.push_plan(initial_plan)
        else:
            self.push(plan_frame(goal))

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
                outcome = self._dispatch(frame, context)
                frame.mark_done(outcome)
                if frame.kind is FrameKind.DO:
                    last_result = outcome
                    context.results.append(outcome)
            except Exception as exc:  # noqa: BLE001
                frame.mark_failed(str(exc))
                if self.config.record_trace:
                    context.trace.append(
                        {"event": "error", "frame": frame.as_dict(), "error": str(exc)}
                    )
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
        if frame.kind is FrameKind.PLAN:
            return self._handle_plan(frame, context)
        if frame.kind is FrameKind.DO:
            return self._handle_do(frame, context)
        raise ValueError(f"unsupported frame kind: {frame.kind}")

    def _handle_plan(self, frame: WorkFrame, context: HarnessContext) -> Any:
        if self.planner is None:
            raise RuntimeError("no planner configured for PLAN frames")
        children = self.planner.plan(frame, context)
        for child in children:
            if child.parent_id is None:
                child.parent_id = frame.id
        if children:
            # First child on top → DFS goes deep before siblings.
            self.push_plan(children)
        return {"planned": [c.as_dict() for c in children]}

    def _handle_do(self, frame: WorkFrame, context: HarnessContext) -> Any:
        tool_name = frame.payload.get("tool")
        if tool_name:
            raw_args = dict(frame.payload.get("args") or {})
            args = context.resolve_args(raw_args)
            result = context.tools.call(tool_name, **args)
        elif self.actor is not None:
            result = self.actor.act(frame, context)
        else:
            raise RuntimeError(
                "DO frame has no 'tool' payload and no actor is configured"
            )

        store_as = frame.payload.get("store_as")
        if store_as:
            context.store_value(store_as, result)
        return result
