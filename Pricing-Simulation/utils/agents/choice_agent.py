import os
import instructor
import openai
from pydantic import Field

from atomic_agents import BaseIOSchema, AgentConfig, AtomicAgent
from atomic_agents.context import ChatHistory, BaseDynamicContextProvider, SystemPromptGenerator

from tools.utils import ChatConfig

"""
The input schema will be the user question or initial prompt, along with a description of what we are looking for (decision_type).
So for example, whether we want to either perform a search, do deeper analysis, etc.
The agent is purely for generic decision making, so search_type defines what decisions we are making.

The output schema will have the boolean Ture/False value for the decision.
Will also have a response detailing the reasoning behind making the decision the agent made.
"""

class ChoiceAgentInputSchema(BaseIOSchema):
    user_input: str = Field(..., description='User\'s last message or question')
    decision_type: str = Field(..., description='Type of decision to be made')
    
class ChoiceAgentOutputSchema(BaseIOSchema):
    make_decision: bool = Field(..., description='Whether the agent has decided to make decision, True/False')
    reasoning: str = Field(..., description='The reasoning behind the agent\'s choice')
    
choice_agent = AtomicAgent[ChoiceAgentInputSchema, ChoiceAgentOutputSchema](
    config=AgentConfig(
        client=instructor.from_openai(openai.OpenAI(api_key=ChatConfig.api_key)),
        model=ChatConfig.model,
        model_api_parameters={'reasoning_effort': ChatConfig.reasoning_effort, 'temperature': 0.1},
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a decision-making agent that determines whether a new web search is needed to answer the user's question.",
                "Your primary role is to analyze whether the existing context contains sufficient, up-to-date information to answer the question.",
                "You must output a clear TRUE/FALSE decision - TRUE if a new search is needed, FALSE if existing context is sufficient.",
            ],
            steps=[
                "1. Analyze the user's question to determine whether or not an answer warrants a new search",
                "2. Review the available web search results",
                "3. Determine if existing information is sufficient and relevant",
                "4. Make a binary decision: TRUE for new search, FALSE for using existing context",
            ],
            output_instructions=[
                "Your reasoning must clearly state WHY you need or don't need new information",
                "If the web search context is empty or irrelevant, always decide TRUE for new search",
                "If the question is time-sensitive, check the current date to ensure context is recent",
                "For ambiguous cases, prefer to gather fresh information",
                "Your decision must match your reasoning - don't contradict yourself",
            ],
        ),
    )
)

if __name__ == "__main__":
    # Example usage for search decision
    search_example = choice_agent.run(
        ChoiceAgentInputSchema(user_message="Who won the nobel prize in physics in 2024?", decision_type="needs_search")
    )
    print(search_example)