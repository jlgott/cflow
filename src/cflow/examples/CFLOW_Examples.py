#CFLOW_Examples

from cflow import Node, Flow  # noqa: F401
from pydantic import BaseModel, Field  # noqa: F401
import asyncio


class MyState(BaseModel):
    user_input: int
    user_name: str
    input_check: str
    path:str = "default path"
    loops: int
    loop_limit:int
    message:str


def start(state):
    return state


def check_input(state):
    if state.loops >= state.loop_limit:
        state.path = "default"
        state.message = "LOOPS EXCEEDED"
        state.input_check = "default"
    elif state.user_input > 5:
        state.input_check = "optional"
        state.path = "optional path"
    else:
        state.input_check = "default"
    return state


def reduce_input(state):
    state.user_input = state.user_input // 2
    state.loops = state.loops + 1
    return state

def finish(state):
    return state


async def main():
    # Build Nodes
    start_node = Node(start)
    check_node = Node(check_input)
    reduce_node = Node(reduce_input)
    finish_node = Node(finish)

    #* Monkey Patch a new POST method into the Node class (advanced)
    def print_post(self, state, result):
        print(f"Result: {result}, {self.id}")
        return result
    
    #apply to all nodes...
    for node in [start_node, check_node, reduce_node, finish_node]:
        node.post = print_post.__get__(node, Node)


    # Build Graph
    flow = Flow()
    flow.set_start(start_node)

    # Set Edges
    start_node >> check_node
    check_node - ("input_check", "optional") >> reduce_node
    reduce_node >> check_node
    check_node - ("input_check", "default") >> finish_node

    # Define Initial State
    initial_state = MyState(
        user_input=1_000,  # Example value > 5
        user_name="Test User",
        input_check="",
        loops=0,
        loop_limit=5,
        message="Starting test"
        )

    # run graph
    result = await flow.run(initial_state)

    print(result)
    return result


asyncio.run(main())

