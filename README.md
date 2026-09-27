# Dynamic Deep Agent

RPN-style stack agent **harness**: each step look at the **work stack** and **top**, then choose:

| Move | When | What happens |
|------|------|----------------|
| **pop + execute** | top is `VALUE` / `OP` | pop; `VALUE` → value stack; `OP` pops arity, runs, pushes result |
| **summary + plan + push** | top is `GOAL` | **token summary** of the stack → plan → push |

OpenAI-style LLM controllers produce the token summary and the move (and optional frames). Token usage is tracked on `context.token_usage`.

```
work:  [ … | top ]          value: [ operands … ]
              │
              ▼
     LLMController.decide(stack, top)
         │  1) summarize stack tokens
         │  2) choose move
         ├─ PLAN_PUSH → plan frames → push
         └─ POP_EXECUTE → pop + execute
```

## Install

```bash
pip install -e ".[dev]"
```

Env for live LLM:

```bash
export OPENAI_API_KEY=sk-...
# optional
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini
```

## OpenAI-style LLM + token summary

```python
from dynamic_deep_agent import (
    AgentHarness,
    LLMController,
    LLMPlanner,
    OpenAIChatClient,
    ToolRegistry,
    tool,
)

@tool(description="Add a and b")
def add(a: int, b: int) -> int:
    return a + b

client = OpenAIChatClient()  # reads OPENAI_* env
reg = ToolRegistry()
reg.register(add)

harness = AgentHarness(
    tools=reg,
    controller=LLMController(client),
    planner=LLMPlanner(client),
)
result = harness.run("Add 2 and 3 using tools")

print(result.result)
print(result.context.token_usage.summary())
for item in result.context.summaries:
    print(item["summary"], item["usage"])
```

`context.summaries` stores each stack **token summary** plus per-call usage.  
`context.token_usage` is the cumulative prompt/completion/total token count.

## Tokens

| Kind | Role |
|------|------|
| `GOAL` | Needs expansion (plan) |
| `VALUE` | Operand / literal |
| `OP` | Operator / tool (`arity`, `tool`, `arg_names`, …) |

`plan` / `do` are aliases for `goal` / `op`.

## Rule-based (no LLM)

```python
from dynamic_deep_agent import AgentHarness, ToolRegistry, op, tool, value

@tool()
def add(a: int, b: int) -> int:
    return a + b

reg = ToolRegistry()
reg.register(add)
tokens = [
    value("2", data=2),
    value("3", data=3),
    op("add", tool="add", arity=2, arg_names=["a", "b"]),
]
assert AgentHarness(tools=reg).run("2+3", initial_plan=tokens).result == 5
```

## Examples

```bash
python examples/calculator_agent.py
python examples/research_agent.py
OPENAI_API_KEY=... python examples/llm_agent.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
