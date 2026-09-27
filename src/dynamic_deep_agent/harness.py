"""RPN-style stack agent harness.

Like reverse Polish notation, each step the agent looks at the **work stack**
and its **top**, then chooses:

* **pop + execute** — consume the top token
  - ``VALUE`` → move onto the **value stack** (operand)
  - ``OP`` → pop ``arity`` operands from the value stack, run, push result
* **summary + plan + push** — top is a ``GOAL`` that needs decomposition

When the work stack is empty, the top of the value stack is the answer.
Tool calls use a separate push/pop call stack.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .controller import Controller, Decision, KindController, Move
from .frames import FrameKind, WorkFrame, goal as goal_frame
from .stack import Stack, WorkStack
from .tools import ToolRegistry


class Planner(Protocol):
    """Builds frames to push when the controller chooses PLAN_PUSH."""

    def plan(
        self,
        frame: WorkFrame,
        context: "HarnessContext",
        *,
        summary: str,
        stack: tuple[WorkFrame, ...],
    ) -> list[WorkFrame]:
        ...


class Actor(Protocol):
    """Runs an OP when no explicit tool is set."""

    def act(self, frame: WorkFrame, context: "HarnessContext") -> Any:
        ...


@dataclass
class HarnessConfig:
    max_steps: int = 100
    stop_on_failure: bool = True
    record_trace: bool = True


@dataclass
class HarnessContext:
    goal: str
    tools: ToolRegistry
    values: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)
    results: list[Any] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    last_summary: str = ""

    def store_value(self, key: str, value: Any) -> None:
        self.values[key] = value

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def resolve_args(self, args: dict[str, Any]) -> dict[str, Any]:
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
    success: bool
    result: Any
    steps: int
    context: HarnessContext
    remaining: tuple[WorkFrame, ...] = ()
    value_stack: tuple[Any, ...] = ()
    error: str | None = None


class AgentHarness:
    """RPN agent: decide from work-stack + top → pop+execute or plan+push."""

    def __init__(
        self,
        *,
        tools: ToolRegistry | None = None,
        controller: Controller | None = None,
        planner: Planner | None = None,
        actor: Actor | None = None,
        config: HarnessConfig | None = None,
        on_step: Callable[[Decision, WorkFrame, HarnessContext], None] | None = None,
    ) -> None:
        self.work = WorkStack()
        self.values = Stack[Any]()
        self.tools = tools or ToolRegistry()
        self.controller = controller or KindController()
        self.planner = planner
        self.actor = actor
        self.config = config or HarnessConfig()
        self.on_step = on_step

    def push(self, frame: WorkFrame) -> None:
        self.work.push(frame)

    def push_plan(self, frames: Iterable[WorkFrame]) -> None:
        """Push so the *first* frame is on top (runs next) — DFS / RPN input order."""
        self.work.push_many(frames)

    def pop(self) -> WorkFrame:
        return self.work.pop()

    def peek(self) -> WorkFrame:
        return self.work.peek()

    def run(self, goal: str, *, initial_plan: Iterable[WorkFrame] | None = None) -> HarnessResult:
        context = HarnessContext(goal=goal, tools=self.tools)
        self.work.clear()
        self.values.clear()

        if initial_plan is not None:
            self.push_plan(initial_plan)
        else:
            self.push(goal_frame(goal))

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
                    value_stack=self.values.snapshot(),
                    error=f"max_steps ({self.config.max_steps}) exceeded",
                )

            stack_view = self.work.snapshot()
            top = self.peek()
            decision = self.controller.decide(stack_view, top, context)
            steps += 1

            if self.on_step:
                self.on_step(decision, top, context)

            try:
                if decision.move is Move.POP_EXECUTE:
                    last_result = self._pop_execute(context)
                elif decision.move is Move.PLAN_PUSH:
                    self._plan_push(decision, context, stack_view)
                else:
                    raise ValueError(f"unknown move: {decision.move}")
            except Exception as exc:  # noqa: BLE001
                if self.config.record_trace:
                    context.trace.append(
                        {
                            "event": "error",
                            "step": steps,
                            "decision": decision.as_dict(),
                            "top": top.as_dict(),
                            "error": str(exc),
                        }
                    )
                if self.config.stop_on_failure:
                    return HarnessResult(
                        success=False,
                        result=None,
                        steps=steps,
                        context=context,
                        remaining=self.work.snapshot(),
                        value_stack=self.values.snapshot(),
                        error=str(exc),
                    )
                continue

            if self.config.record_trace:
                context.trace.append(
                    {
                        "event": "step",
                        "step": steps,
                        "decision": decision.as_dict(),
                        "top": top.as_dict(),
                        "result": last_result if decision.move is Move.POP_EXECUTE else None,
                        "work_depth": self.work.depth,
                        "value_depth": self.values.depth,
                        "tool_depth": self.tools.call_stack.depth,
                    }
                )

        if not self.values.empty():
            last_result = self.values.peek()

        return HarnessResult(
            success=True,
            result=last_result,
            steps=steps,
            context=context,
            remaining=(),
            value_stack=self.values.snapshot(),
        )

    def _plan_push(
        self,
        decision: Decision,
        context: HarnessContext,
        stack_view: tuple[WorkFrame, ...],
    ) -> None:
        top = self.peek()
        summary = decision.summary or context.last_summary
        context.last_summary = summary

        frames = list(decision.frames)
        if not frames:
            if self.planner is None:
                raise RuntimeError(
                    "PLAN_PUSH has no frames and no planner is configured"
                )
            frames = self.planner.plan(
                top, context, summary=summary, stack=stack_view
            )

        parent = top
        if decision.consume_top:
            parent = self.pop()
            parent.mark_done({"summary": summary, "planned": [f.id for f in frames]})

        for child in frames:
            if child.parent_id is None:
                child.parent_id = parent.id
        if frames:
            self.push_plan(frames)

    def _pop_execute(self, context: HarnessContext) -> Any:
        frame = self.pop()
        frame.mark_running()

        if frame.kind is FrameKind.VALUE:
            result = frame.payload.get("value", frame.result)
            frame.mark_done(result)
            self.values.push(result)
            context.results.append(result)
            store_as = frame.payload.get("store_as")
            if store_as:
                context.store_value(store_as, result)
            return result

        if frame.kind is FrameKind.OP:
            result = self._execute_op(frame, context)
            frame.mark_done(result)
            self.values.push(result)
            context.results.append(result)
            store_as = frame.payload.get("store_as")
            if store_as:
                context.store_value(store_as, result)
            return result

        raise RuntimeError(
            f"POP_EXECUTE on unsupported kind {frame.kind.value!r}; "
            "controller should choose PLAN_PUSH for GOAL"
        )

    def _execute_op(self, frame: WorkFrame, context: HarnessContext) -> Any:
        arity = int(frame.payload.get("arity", 0))
        operands: list[Any] = []
        for _ in range(arity):
            if self.values.empty():
                raise RuntimeError(
                    f"OP {frame.description!r} needs arity={arity}, value stack too shallow"
                )
            operands.append(self.values.pop())
        # operands gathered top-first; reverse to left-to-right order
        ordered = list(reversed(operands))

        tool_name = frame.payload.get("tool")
        if tool_name:
            raw_args = dict(frame.payload.get("args") or {})
            arg_names = list(frame.payload.get("arg_names") or [])
            if arity and arg_names:
                for name, val in zip(arg_names, ordered, strict=False):
                    raw_args.setdefault(name, val)
            elif arity and not raw_args:
                raw_args = {"operands": ordered}
            args = context.resolve_args(raw_args)
            return context.tools.call(tool_name, **args)

        if self.actor is not None:
            context.scratch["operands"] = ordered
            return self.actor.act(frame, context)

        raise RuntimeError(
            "OP frame has no 'tool' payload and no actor is configured"
        )
