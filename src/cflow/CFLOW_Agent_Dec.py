import asyncio
import functools
import time
from typing import Any, Dict, List, Optional, Union  # noqa: F401

from agents import (  # noqa: F401
    Agent,
    RunConfig,
    RunContextWrapper,
    Runner,
    function_tool,
    set_default_openai_api,
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field  # noqa: F401

from CFLOW import Flow, Node, ParallelNode  # noqa: F401

load_dotenv()
set_default_openai_api("chat_completions")


"""
NOTES: 
# TODO Need to figure out a way for ContextInjection here.  UserContext and RunContextWrapper for OpenAI Agent Class.

"""


class LLMState(BaseModel):
    user_input: Union[str, List[str]] = None
    flow_path: str = "sequential"
    llm_responses: Union[str, List[str], BaseModel, Dict[str, Any]] = None
    agent_config: Dict[str, Any] = {}


def call_LLM(
    name: str = "DefaultDecoratorAgent",
    model: str = "gpt-4.1-mini",
    system_prompt: str = "",
    output_type: Union[BaseModel, Dict[str, Any]] = None,
    tools=[],
    timeout: int = 30,
    max_turns: int = 3,
    max_concurrent:int = 5,
    tracing_disabled: bool = False
    ):
    def fn_decorator(user_func):
        @functools.wraps(user_func)
        async def wrapper(*args, **kwargs):
            # Get user queries
            user_output = user_func(*args, **kwargs)

            # Define Funcs for Flow Graph
            async def start(state: LLMState) -> LLMState:
                state.user_input = user_output
                state.agent_config = {
                    "name": name,
                    "model": model,
                    "instructions": system_prompt,
                    "tools": tools or [],
                    "output_type": output_type or None,
                }
                if isinstance(user_output, list):
                    state.flow_path = "parallel"
                return state

            async def sequential_llm(state: LLMState) -> LLMState:
                agent = Agent(**state.agent_config)
                result = await Runner.run(
                    agent,
                    state.user_input,
                    max_turns=max_turns,
                    run_config=RunConfig(tracing_disabled=tracing_disabled),
                )
                state.llm_responses = result.final_output
                return state

            async def parallel_llm(state: LLMState) -> LLMState:
                agent = Agent(**state.agent_config)
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def run_with_semaphore(query):
                    async with semaphore:
                        return await Runner.run(
                            agent,
                            query,
                            max_turns=max_turns,
                            run_config=RunConfig(tracing_disabled=tracing_disabled),
                        )
                
                tasks = [run_with_semaphore(query) for query in state.user_input]
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
                state.llm_responses = [r.final_output for r in results]
                return state

            async def finish(state: LLMState):
                print(f"State from Finish: {state}")
                return state

            # Create nodes
            start_node = Node(start)
            sequential_node = Node(sequential_llm)
            parallel_node = Node(parallel_llm)
            finish_node = Node(finish)

            # Build graph
            start_node - ("flow_path", "sequential") >> sequential_node
            start_node - ("flow_path", "parallel") >> parallel_node
            parallel_node >> finish_node
            sequential_node >> finish_node

            # Run flow
            flow = Flow()
            flow.set_start(start_node)

            initial_state = LLMState()
            result_state = await flow.run(initial_state)

            return result_state.llm_responses

        return wrapper

    return fn_decorator


# Example usage
if __name__ == "__main__":
    print("\n")

    class DemoOutput(BaseModel):
        model_output:str


    async def sequential_example():
        ss = time.time()

        @call_LLM(system_prompt="Be helpful and brief", name="seq1")
        def single_query():
            return "What is the capital of France?"

        @call_LLM(system_prompt="Be helpful and brief", name="seq2", output_type=DemoOutput)
        def single_query2():
            return "What is the capital of Spain?"

        result1 = await single_query()
        result2 = await single_query2()
        print(f"Sequential Time: {time.time() - ss:.3f}")
        print(f"Results: {result1}, {result2}")
        print(f"Result1 type: {type(result1)}")
        print(f"Result2 type: {type(result2)}")



    async def parallel_example():
        sp = time.time()

        @call_LLM(system_prompt="Be helpful and brief", name="parallel_test", output_type=DemoOutput)
        def multiple_queries():
            return ["What is the capital of France?", "What is the capital of Spain?"]

        result = await multiple_queries()
        print(f"Parallel Time: {time.time() - sp:.3f}")
        print(f"Results: {result}")


    async def test():
        await sequential_example()
        print("\n\n")
        await parallel_example()

    asyncio.run(test())
