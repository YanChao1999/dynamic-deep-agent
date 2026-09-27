# Dynamic Deep Agent

DFS stack agent **harness**: **plan** by pushing, **do** by popping, finish when the stack is empty and you have a result.

```
push GOAL (PLAN)
   │
   ▼
pop PLAN ──► push [child1, child2, ...]     ← go deeper (DFS)
   │
   ▼
pop DO    ──► run tool / actor → result
   │
   ▼
… repeat until stack empty → last DO result
```

Only two work kinds:

| Kind | Meaning |
|------|---------|
| `PLAN` | Pop → decompose → **push** children (first child runs next) |
| `DO` | Pop → execute tool/actor → keep result |

Push/pop also apply to the **tool-call stack** (enter tool / return).

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from dynamic_deep_agent import AgentHarness, ToolRegistry, do, tool
from dynamic_deep_agent.builtins import RulePlanner

@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

registry = ToolRegistry()
registry.register(add)

def plan_fn(frame, context):
    return [
        do("2+3", tool="add", args={"a": 2, "b": 3}, store_as="sum"),
        do("reuse", tool="add", args={"a": "$sum", "b": 0}),
    ]

harness = AgentHarness(tools=registry, planner=RulePlanner(plan_fn))
result = harness.run("2+3")
assert result.success and result.result == 5
assert harness.work.empty()
```

`$name` in tool args resolves from results stored with `store_as`.

## Loop

1. `run(goal)` pushes a root `PLAN` (or an `initial_plan`).
2. While the work stack is not empty: **pop**.
3. `PLAN` → planner **pushes** children (DFS: newest / first child first).
4. `DO` → tool registry **pushes/pops** the tool-call stack; result is kept.
5. Empty work stack → `HarnessResult` with the last `DO` result.

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
  frames.py     # PLAN / DO (+ tool frames)
  tools.py      # ToolRegistry with call-stack push/pop
  harness.py    # DFS AgentHarness loop
  builtins.py   # Small planners/actors for demos
```
