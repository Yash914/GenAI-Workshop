from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=200
)

parser = StrOutputParser()

# Prompt 1
fact_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Give me 5 interesting facts about {topic}.")
])

# Prompt 2
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain the topic {topic} in a short paragraph.")
])

# Parallel Chain
parallel_chain = RunnableParallel(
    facts=fact_prompt | model | parser,
    summary=summary_prompt | model | parser
)

# Run
result = parallel_chain.invoke({"topic": "Unemployment in India"})

print("Facts:\n", result["facts"])
print("\nSummary:\n", result["summary"])

# Graph
parallel_chain.get_graph().print_ascii()