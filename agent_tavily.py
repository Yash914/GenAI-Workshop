import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

# Tavily Tool
search = TavilySearchResults(max_results=3)
tools = [search]

# Prompt Template
template = """
Answer the following question as best you can.

You have access to the following tools:
{tools}

Use the following format:

Question: {input}
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: input to the action
Observation: result of the action
... (repeat if needed)

Thought: I now know the final answer
Final Answer: the answer to the question
"""

prompt = PromptTemplate.from_template(template)

# Create agent
agent = create_react_agent(llm, tools, prompt)

# Agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run agent
response = agent_executor.invoke({
    "input": "What are the latest AI news today?"
})

print(response["output"])