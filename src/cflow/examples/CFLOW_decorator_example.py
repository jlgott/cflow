import asyncio
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

from cflow import Flow, Node, ParallelNode, call_LLM  # noqa: F401

load_dotenv()
set_default_openai_api("chat_completions")


# Example usage
if __name__ == "__main__":
    print("\n")

    # * create structured output via pydantic BaseModel
    class DemoOutput(BaseModel):
        model_output: str

    # * call two sequential examples and time the response generation
    async def sequential_example():
        ss = time.time()

        @call_LLM(
            system_prompt="Be helpful and brief", name="seq1", model="gpt-4.1-mini"
        )
        def single_query():
            return "What is the capital of France?"

        @call_LLM(
            system_prompt="Be helpful and brief",
            name="seq2",
            model="gpt-4.1-mini",
            output_type=DemoOutput,
        )
        def single_query2():
            return "What is the capital of Spain?"

        result1 = await single_query()
        result2 = await single_query2()
        print(f"Sequential Time: {time.time() - ss:.3f}")
        print(f"Results: {result1}, {result2}")
        print(f"Result1 type: {type(result1)}")
        print(f"Result2 type: {type(result2)}")

    # * call a parallel example (list input)
    # * remember all functions that return a list of strings to the llm decorator go parallel by default.
    async def parallel_example():
        sp = time.time()

        @call_LLM(
            system_prompt="Be helpful and brief",
            name="parallel_test",
            model="gpt-4.1-mini",
            output_type=DemoOutput,
        )
        def multiple_queries():
            return ["What is the capital of France?", "What is the capital of Spain?"]

        result = await multiple_queries()
        print(f"Parallel Time: {time.time() - sp:.3f}")
        print(f"Results: {result}")

    #* run both tests
    async def test():
        await sequential_example()
        print("\n\n")
        await parallel_example()

    #* call them async -> library is async first!
    asyncio.run(test())
