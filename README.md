# Dynamic Deep Agent

Stack-based agent **harness**: the agent plans by **pushing** work onto a stack, **pops** the top item to do it, and finishes when the stack is **empty**.

Push and pop mean two related things — the same LIFO idea as a calculator:

| Stack | Push means | Pop means |
|-------|------------|-----------|
| **Work stack** | Plan / enqueue a goal, subtask, or value | Start doing that work |
| **Tool-call stack** | Enter a tool invocation | Return from the tool |

```
goal
  └─ push PLAN ─────────────────────────────┐
       pop PLAN → push [A, B, REDUCE]       │  work stack
       pop A    → tool call (push/pop)      │
       pop B    → tool call (push/pop)      │
       pop REDUCE → combine values          │
  stack empty → task finished ──────────────┘
```

## Why a stack?

- **Depth-first** decomposition: the freshest subtask runs next (like a call stack).
- **Calculator-shaped** control flow: operands (`VALUE`), operators (`ACTION` / `REDUCE`), expression complete when nothing remains.
- **Nested tools** are honest: the tool-call stack mirrors real call/return, including failures that still pop.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from dynamic_deep_agent import AgentHarness, FrameKind, ToolRegistry, WorkFrame, tool

@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

registry = ToolRegistry()
registry.register(add)

plan = [
    WorkFrame(
        kind=FrameKind.ACTION,
        description="2+3",
        payload={"tool": "add", "args": {"a": 2, "b": 3}, "store_as": "sum"},
    ),
    WorkFrame(
        kind=FrameKind.REDUCE,
        description="return sum",
        payload={"op": "identity", "keys": ["sum"]},
    ),
]

harness = AgentHarness(tools=registry)
result = harness.run("2+3", initial_plan=plan)
assert result.success
assert result.context.get_value("sum") == 5
assert harness.work.empty()
```

## Frame kinds

| Kind | Role |
|------|------|
| `GOAL` | Root objective; planner pushes a decomposition |
| `PLAN` | Intermediate plan node; may push more children |
| `ACTION` | Concrete step — tool call or actor |
| `VALUE` | Store an operand / intermediate result |
| `REDUCE` | Combine stored values (sum, concat, …) |
| `DONE` | Explicit finish marker for a sub-goal |

## Loop

1. `run(goal)` pushes a `GOAL` (or an `initial_plan`).
2. While the work stack is not empty: **pop** → dispatch by kind.
3. `GOAL`/`PLAN` → planner **pushes** children (first child on top).
4. `ACTION` → tool registry **pushes/pops** the tool-call stack.
5. Empty work stack → `HarnessResult.success`.

## Examples

```bash
python examples/calculator_agent.py
python examples/research_agent.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Package layout

```
src/dynamic_deep_agent/
  stack.py      # WorkStack, ToolCallStack
  frames.py     # WorkFrame, ToolFrame, FrameKind
  tools.py      # ToolRegistry with call-stack push/pop
  harness.py    # AgentHarness main loop
  builtins.py   # Small planners/actors for demos
```
