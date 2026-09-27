# Dynamic Deep Agent

RPN-style stack agent **harness**: each step the agent looks at the **work stack** and **top**, then chooses:

| Move | When | What happens |
|------|------|----------------|
| **pop + execute** | top is `VALUE` or `OP` | pop it; `VALUE` → value stack; `OP` → pop arity operands, run, push result |
| **summary + plan + push** | top is `GOAL` | summarize stack situation, plan tokens, push them |

When the work stack is empty, the **value stack** top is the answer.

```
work:  [ … | top ]          value: [ operands … ]
              │
              ▼
     controller.decide(stack, top)
         │              │
         │              └─ PLAN_PUSH → summary → plan → push
         └─ POP_EXECUTE → pop top → execute
```

## Tokens

| Kind | Role |
|------|------|
| `GOAL` | Needs expansion (plan) |
| `VALUE` | Operand / literal |
| `OP` | Operator / tool call (`arity`, `tool`, `arg_names`, …) |

`plan` / `do` are aliases for `goal` / `op`.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start (RPN calc)

```python
from dynamic_deep_agent import AgentHarness, ToolRegistry, op, tool, value

@tool()
def add(a: int, b: int) -> int:
    return a + b

reg = ToolRegistry()
reg.register(add)

# postfix: 2 3 +
tokens = [
    value("2", data=2),
    value("3", data=3),
    op("add", tool="add", arity=2, arg_names=["a", "b"]),
]
result = AgentHarness(tools=reg).run("2+3", initial_plan=tokens)
assert result.result == 5
```

## Custom controller

```python
from dynamic_deep_agent import Decision, FrameKind
from dynamic_deep_agent.builtins import RuleController

def decide(stack, top, context):
    if top.kind is FrameKind.GOAL:
        return Decision.plan_push(summary=f"depth={len(stack)}")
    return Decision.pop_execute()

harness = AgentHarness(controller=RuleController(decide), planner=...)
```

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
