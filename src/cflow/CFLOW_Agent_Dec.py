import asyncio
import functools
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

from cflow import Flow, Node, ParallelNode  # noqa: F401

load_dotenv()
set_default_openai_api("chat_completions")


"""
NOTES: 
# TODO Need to figure out an easy way for ContextInjection here.  UserContext and RunContextWrapper for OpenAI Agent Class.

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
    timeout: int = 30,  # async timeout
    max_turns: int = 3,  # openai Agent limiter
    max_concurrent:int = 5,  # semaphore
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
                # print(f"State from Finish: {state}") #* UNCOMMENT FOR DEBUGGING
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
