# CFlow (Core Functional flow) 

A lightweight Python library for building and executing asynchronous workflow graphs with **immutable** state management.

## Features

- **Simple Graph Building**: Use intuitive `>>` and `-` operators to connect nodes
- **Async/Sync Support**: Automatically wraps synchronous functions for async execution
- **Immutable State**: Safe state management with deep copying
- **Conditional Routing**: Route execution based on state values
- **Retry Logic**: Built-in retry mechanisms with backoff
- **Parallel Execution**: Execute multiple flows concurrently
- **Type Safety**: Full type hints and Pydantic BaseModel support

### The Idea

- Create the simplest graph workflow orchestrator with functional programming-like state management.
- inspired by the need to enforce some high level determinism in Agentic workflows.
- the idea is simple: a graph with immutable state abstracted away from the user:
    - Node: IN: state, OUT: state
    - Every Node creates a copy of the incoming state, potentially modifies the copy, and returns it as the new state.
    - all of the deep copy logic is abstracted away from the user and is modifiable by monkey patching the Node.prep() method.

## Roadmap
- **v0.2.0**: Standardized Utility functions for LLM calls, DB Adaptors, Basic LLM Tools, Fan-Out Agentic Flows, support for LLM-Lite or OpenRouter.

## Quick Start

```python
import asyncio
from cflow import Node, Flow

# Define your workflow steps
async def start_process(state):
    state["step"] = "started"
    state["counter"] = state.get("counter", 0) + 1
    return state

def check_condition(state):  # Sync functions work too!
    if state.get("counter", 0) >= 5:
        state["next_action"] = "finish"
    else:
        state["next_action"] = "continue"
    return state

async def continue_process(state):
    state["step"] = "continuing"
    return state

async def finish_process(state):
    state["step"] = "finished"
    state["message"] = "Complete!"
    return state

# Build the workflow graph
start_node = Node(start_process)
check_node = Node(check_condition)
continue_node = Node(continue_process)
finish_node = Node(finish_process)

# Connect nodes with simple operators
start_node >> check_node
check_node - ("next_action", "continue") >> continue_node
check_node - ("next_action", "finish") >> finish_node
continue_node >> start_node  # Create a loop

# Execute the workflow
async def main():
    flow = Flow()
    flow.set_start(start_node)
    
    initial_state = {"message": "Hello World"}
    result = await flow.run(initial_state)
    print(f"Final result: {result}")

asyncio.run(main())
```

## Core Concepts

### Nodes

Nodes wrap your functions and handle execution, retries, and state management:

```python
# Basic node
node = Node(my_function)

# Node with retry logic
node = Node(my_function, retries=3, backoff=1.0)

# Nodes automatically handle sync/async functions
sync_node = Node(lambda state: {"result": "sync"})
async_node = Node(lambda state: asyncio.sleep(0.1) or {"result": "async"})
```

### State Management

State is automatically copied to ensure immutability:

```python
# Works with dictionaries
initial_state = {"key": "value"}

# Works with Pydantic models
from pydantic import BaseModel

class MyState(BaseModel):
    key: str
    value: int = 0

initial_state = MyState(key="value")
```

### Graph Building

Use operators to connect nodes:

```python
# Unconditional edge
node_a >> node_b

# Conditional edges
node_a - ("status", "success") >> success_node
node_a - ("status", "error") >> error_node

# Mix conditional and unconditional
node_a - ("retry", True) >> retry_node
node_a >> default_node  # Default when no conditions match
```

### Parallel Execution

Execute multiple flows concurrently:

```python
from cflow import ParallelNode

# Create separate flows
flow1 = Flow()
flow1.set_start(process_data_node)

flow2 = Flow()
flow2.set_start(validate_input_node)

# Execute in parallel
parallel_node = ParallelNode([flow1, flow2])

# Custom merge function (optional)
def custom_merge(states):
    return {"combined": [s for s in states]}

parallel_node = ParallelNode([flow1, flow2], merge_fn=custom_merge)
```

## Advanced Usage

### Custom Node Lifecycle

Override node methods (monkey patch) for custom behavior:

```python
class LoggingNode(Node):
    def post(self, state, result):
        print(f"Node {self.id} completed: {result}")
        return super().post(state, result)

node = LoggingNode(my_function)
```

### Error Handling

```python
# Retry with backoff
risky_node = Node(might_fail_function, retries=3, backoff=2.0)

try:
    result = await flow.run(initial_state)
except RuntimeError as e:
    print(f"Workflow failed: {e}")
```

### Complex Routing

```python
def router(state):
    priority = state.get("priority", "normal")
    if priority == "high":
        state["route"] = "express"
    elif priority == "low":
        state["route"] = "batch"
    else:
        state["route"] = "standard"
    return state

router_node = Node(router)
router_node - ("route", "express") >> express_node
router_node - ("route", "batch") >> batch_node
router_node - ("route", "standard") >> standard_node
```

## API Reference

### Node

```python
Node(func, retries=0, backoff=0.0)
```

- `func`: Function to execute (sync or async)
- `retries`: Number of retry attempts on failure
- `backoff`: Delay between retries in seconds

### Flow

```python
Flow()
```

Main workflow execution engine.

**Methods:**
- `set_start(node)`: Set the starting node
- `run(state)`: Execute the workflow

### ParallelNode

```python
ParallelNode(flows, merge_fn=None)
```

- `flows`: List of Flow objects to execute in parallel
- `merge_fn`: Optional function to merge results

### Rules.md
 - includes a Rules.md file to help Cursor, Claude, other IDE Agentic code builders leverage the capability of the library directly.

## Requirements

- Python 3.11+
- pydantic
- asyncio (built-in)

## License

Apache License 2.0 - see LICENSE file for details.

## Contributing

Contributions / Feedback welcome!

## Examples

Check out the `examples/` directory for more complex workflow patterns.
