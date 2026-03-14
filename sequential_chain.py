from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompt 1
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Give me 5 interesting facts about: {topic}.")
])

# Prompt 2
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Summarize the following text:\n{text}")
])

# Model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=200
)

# Output Parser
parser = StrOutputParser()

# Chain
chain = prompt1 | model | parser | prompt2 | model | parser

# Run Chain
result = chain.invoke({"topic": "Unemployment in India"})

print(result)

# Print graph
chain.get_graph().print_ascii()