import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union  # noqa: F401
from pydantic import BaseModel
import copy

# Type definitions
State = Union[Dict[str, Any], BaseModel]
StepFunc = Callable[[State], Union[State, Awaitable[State]]]


# EVENT_LOG = []
# EVENT_LOG_COUNTER = 0


class ConditionalTransition:
    """Helper class to handle conditional transitions syntax: node - (key, value) >> target"""

    def __init__(self, source_node: "Node", key: str, value: Any):
        self.source_node = source_node
        self.key = key
        self.value = value

    def __rshift__(self, target_node: "Node") -> "Node":
        """Complete the conditional transition: - (key, value) >> target"""
        self.source_node._add_conditional_edge(self.key, self.value, target_node)
        return target_node


class Node:
    """Lightweight node wrapper around user functions with immutable state management"""

    def __init__(self, func: StepFunc, retries: int = 0, backoff: float = 0.0):
        self.func = func
        self.id = func.__name__
        self.retries = retries
        self.backoff = backoff

        # Graph edges
        self.default_edge: Optional["Node"] = None
        self.conditional_edges: Dict[Tuple[str, Any], "Node"] = {}

        # Auto-wrap sync functions
        if not inspect.iscoroutinefunction(func):
            self.func = self._wrap_sync_function(func)

    def _wrap_sync_function(
        self, func: StepFunc
    ) -> Callable[[State], Awaitable[State]]:
        """Convert sync function to async"""

        async def wrapped(state: State) -> State:
            return func(state)

        wrapped.__name__ = func.__name__
        return wrapped

    def prep(self, state: State) -> State:
        """Prepare state by creating immutable copy"""
        if isinstance(state, BaseModel):
            return state.model_copy(deep=True)
        elif isinstance(state, dict):
            return copy.deepcopy(state)
        else:
            raise TypeError(
                f"Unsupported state type: {type(state)}. Must be Dict or BaseModel"
            )

    async def execute(self, state: State) -> State:
        """Execute the user function with retry logic"""
        for attempt in range(self.retries + 1):
            try:
                return await self.func(state)
            except Exception as e:
                if attempt == self.retries:
                    raise RuntimeError(
                        f"Node '{self.id}' failed after {self.retries + 1} attempts"
                    ) from e
                if self.backoff > 0:
                    await asyncio.sleep(self.backoff)

    def post(self, state: State, result: State) -> State:
        """Post-processing hook for logging/instrumentation"""
        # Default implementation - can be overridden by users if needed

        # global EVENT_LOG_COUNTER
        # event_to_add = (EVENT_LOG_COUNTER, result)
        # EVENT_LOG.append(event_to_add)
        # EVENT_LOG_COUNTER+=1

        return result

    async def run(self, state: State) -> State:
        """Run the complete prep -> execute -> post cycle"""
        prepped_state = self.prep(state)
        result = await self.execute(prepped_state)
        return self.post(state, result)

    def get_next_node(self, result_state: State) -> Optional["Node"]:
        """Determine next node based on result state"""
        # Check conditional edges first
        for (key, value), target_node in self.conditional_edges.items():
            if isinstance(result_state, BaseModel):
                state_value = getattr(result_state, key, None)
            else:
                state_value = result_state.get(key)

            if state_value == value:
                return target_node

        # Fall back to default edge
        return self.default_edge

    def _add_conditional_edge(self, key: str, value: Any, target_node: "Node"):
        """Add conditional edge (used by ConditionalTransition)"""
        self.conditional_edges[(key, value)] = target_node

    def __rshift__(self, other: "Node") -> "Node":
        """Implement >> operator for unconditional edges"""
        if self.default_edge is not None:
            raise ValueError(
                f"Node '{self.id}' already has a default edge. Use conditional transitions for multiple edges."
            )
        self.default_edge = other
        return other

    def __sub__(self, condition: Tuple[str, Any]) -> ConditionalTransition:
        """Implement - operator for conditional transitions: node - (key, value)"""
        if not isinstance(condition, tuple) or len(condition) != 2:
            raise ValueError("Condition must be a tuple of (key, value)")
        key, value = condition
        return ConditionalTransition(self, key, value)


class Flow:
    """Graph execution engine with immutable state management"""

    def __init__(self):
        self.start_node: Optional[Node] = None

    def set_start(self, node: Node) -> None:
        """Set the starting node for execution"""
        self.start_node = node

    async def run(self, state: State) -> State:
        """Execute the graph starting from start_node"""
        if self.start_node is None:
            raise ValueError("No start node set. Call set_start() first.")

        if not isinstance(state, (dict, BaseModel)):
            raise TypeError(
                f"Initial state must be Dict or BaseModel, got {type(state)}"
            )

        current_node = self.start_node
        current_state = state

        while current_node is not None:
            # Run the current node
            current_state = await current_node.run(current_state)

            # Get the next node
            next_node = current_node.get_next_node(current_state)

            if next_node is None:
                # End of graph
                break

            current_node = next_node

        return current_state


class ParallelNode(Node):
   """Node that executes multiple flows in parallel and merges results"""
   
   def __init__(self, flows: List[Flow], merge_fn: Optional[Callable[[List[State]], State]] = None):
        self.flows = flows
        self.merge_fn = merge_fn
        self.id = "parallel_node"
        
        # Initialize Node attributes that >> operator needs
        self.default_edge: Optional[Node] = None
        self.conditional_edges: Dict[Tuple[str, Any], Node] = {}
        
        # We don't have a function, so no need for func-related attributes
   
   def _copy_state(self, state: State) -> State:
       """Create immutable copy of state for parallel execution"""
       if isinstance(state, BaseModel):
           return state.model_copy(deep=True)
       elif isinstance(state, dict):
           return copy.deepcopy(state)
       else:
           raise TypeError(f"Unsupported state type: {type(state)}. Must be Dict or BaseModel")
   
   def _default_merge(self, states: List[State]) -> State:
       """Default merge function for parallel execution results"""
       if not states:
           raise ValueError("No states to merge")
       
       if isinstance(states[0], BaseModel):
           # BaseModel merge
           base = states[0].model_copy()
           for state in states[1:]:
               updates = {k: v for k, v in state.model_dump().items() 
                         if v is not None and v != ""}
               if updates:
                   base = base.model_copy(update=updates)
           return base
       else:
           # Dict merge
           merged = copy.deepcopy(states[0])
           for state in states[1:]:
               for key, value in state.items():
                   if value is not None and value != "":
                       merged[key] = value
           return merged
   
   async def execute(self, state: State) -> State:
       """Execute all flows in parallel and merge results"""
       if not self.flows:
           raise ValueError("No flows to execute")
       
       # Create tasks for parallel execution
       tasks = [flow.run(self._copy_state(state)) for flow in self.flows]
       
       # Run all flows in parallel
       results = await asyncio.gather(*tasks)
       
       # Use custom merge function or default
       merge_func = self.merge_fn or self._default_merge
       return merge_func(results)




# Example usage and testing
if __name__ == "__main__":
    
    # Example functions
    async def start_process(state):
        print(f"Starting process with: {state}")
        state["step"] = "started"
        state["counter"] = state.get("counter", 0) + 1
        return state

    def check_condition(state):
        # Sync function - will be auto-wrapped
        print(f"Checking condition: {state}")
        if state.get("counter", 0) >= 5:
            state["next_action"] = "finish"
        else:
            state["next_action"] = "continue"
        return state

    async def continue_process(state):
        print(f"Continuing process: {state}")
        state["step"] = "continuing"
        return state

    async def finish_process(state):
        print(f"Finishing process: {state}")
        state["step"] = "finished"
        state["message"] = "We are Done!"
        return state

    # Create nodes
    start_node = Node(start_process)
    check_node = Node(check_condition)
    continue_node = Node(continue_process)
    finish_node = Node(finish_process)

    # Build graph with conditional loop
    start_node >> check_node
    check_node - ("next_action", "continue") >> continue_node
    check_node - ("next_action", "finish") >> finish_node
    continue_node >> start_node  # Loop back

    # Create and run flow
    async def main():
        flow = Flow()
        flow.set_start(start_node)
        initial_state = {"message": "Hello World"}
        result = await flow.run(initial_state)

        print(f"Final result: {result}")

        # print(f"EL\n{EVENT_LOG}")

    # Run example
    asyncio.run(main())
