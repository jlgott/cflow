from CFLOW import Flow, Node, ParallelNode  # noqa: F401
from pydantic import BaseModel, Field  # noqa: F401
import asyncio
import random  # noqa: F401
from typing import List, Dict, Optional  # noqa: F401
import time


# Example usage and testing
if __name__ == "__main__":

    def start(state):
        return state

    # Same processing functions for both demos
    async def process_a(state):
        print("Process A starting...")
        await asyncio.sleep(.5)
        print("Process A completed!")
        state["result_a"] = "data_from_a"
        return state

    async def process_b(state):
        print("Process B starting...")
        await asyncio.sleep(.4)
        print("Process B completed!")
        state["result_b"] = "data_from_b"
        return state

    async def process_c(state):
        print("Process C starting...")
        await asyncio.sleep(.6)
        print("Process C completed!")
        state["result_c"] = "data_from_c"
        return state

    async def end(state):
        return state


    # DEFINE NODES
    
    start_node = Node(start)
    pstart_node = Node(start)
    node_a = Node(process_a)
    node_b = Node(process_b)
    node_c = Node(process_c)
    end_node = Node(end)

    parallel_node = ParallelNode([node_a, node_b, node_c])

    initial_state = {
        "input_data": "test_data",
        "result_a": None,
        "result_b": None, 
        "result_c": None
        }


    async def demo_sequential():
        print("=== Sequential Demo ===")
        start_time = time.time()
        print(initial_state)
        
        sflow = Flow()
        sflow.set_start(start_node)

        start_node >> node_a >> node_b >> node_c >> end_node
        
        result = await sflow.run(initial_state)
        print(result)
        print(f"Sequential took: {time.time() - start_time:.2f}s")  # Should be ~6s
        return result

    async def demo_parallel():
        print("=== Parallel Demo ===")
        start_time = time.time()
        print(initial_state)
        
        pflow = Flow()
        pflow.set_start(pstart_node)

        pstart_node >> parallel_node
        parallel_node >> end_node
                
        result = await pflow.run(initial_state)
        print(result)
        print(f"Parallel took: {time.time() - start_time:.2f}s")   # Should be ~3s
        return result
            

# Run both demos
async def main():
    await demo_parallel()
    await demo_sequential()
    
# Run example
asyncio.run(main())