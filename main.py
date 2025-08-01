import os
import json
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import tool
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

# ------------------ Load API KEY ------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ------------------ Load Data ------------------
DATA_DIR = "data"

# Load flights.json
with open(os.path.join(DATA_DIR, "flights.json"), "r") as f:
    mock_flights = json.load(f)

# Load visa_rules.md
with open(os.path.join(DATA_DIR, "visa_rules.md"), "r") as f:
    docs = [f.read()]

# ------------------ LLM-Based Travel Parser ------------------

class TravelQuery(BaseModel):
    origin: Optional[str] = Field(description="Departure city or airport")
    destination: Optional[str] = Field(description="Arrival city or airport")
    departure_month: Optional[str] = Field(description="Month of departure if exact date not specified")
    departure_date: Optional[str] = Field(description="Exact departure date if specified, in YYYY-MM-DD format")
    return_date: Optional[str] = Field(description="Return date for round-trip, in YYYY-MM-DD format")
    airline_alliance: Optional[str] = Field(description="Preferred airline alliance (e.g., Star Alliance)")
    avoid_overnight_layovers: Optional[bool] = Field(description="Whether the user wants to avoid overnight layovers")

def parse_query_llm(query: str) -> dict:
    """Use OpenAI LLM to parse structured flight search intent from user query."""
    parser = PydanticOutputParser(pydantic_object=TravelQuery)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful travel assistant that extracts structured travel preferences from user messages."),
        ("human", "Extract travel preferences from this message:\n\n{query}\n\n{format_instructions}")
    ])

    chain = prompt | ChatOpenAI() | parser

    result = chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
    return result.dict()

# ------------------ Flight Search Tool ------------------

def filter_flights(flights, filters):
    """Filter mock flight listings based on user preferences."""
    filtered = []
    for flight in flights:
        if filters.get("origin") and filters["origin"].lower() != flight["from"].lower():
            continue
        if filters.get("destination") and filters["destination"].lower() != flight["to"].lower():
            continue
        if filters.get("airline_alliance") and filters["airline_alliance"].lower() != flight["alliance"].lower():
            continue
        filtered.append(flight)
    return filtered

@tool
def search_flights(query: str) -> str:
    """Search flights using a structured query parsed by LLM."""
    parsed = parse_query_llm(query)
    results = filter_flights(mock_flights, parsed)
    if not results:
        return "No flights found matching your criteria."
    return json.dumps(results, indent=2)

# ------------------ Knowledge Base RAG Setup ------------------
splitter = CharacterTextSplitter(chunk_size=300)
texts = splitter.create_documents(docs)
vectorstore = Chroma.from_documents(texts, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

@tool
def visa_info(query: str) -> str:
    """Answers visa-related questions using ChromaDB RAG."""
    return rag_chain.run(query)

@tool
def refund_policy(query: str) -> str:
    """Answers refund-related questions using ChromaDB RAG."""
    return rag_chain.run(query)

# ------------------ LangChain Agent Setup ------------------
tools = [
    Tool(name="FlightSearch", func=search_flights, description="Find and filter flights based on user query"),
    Tool(name="VisaRAG", func=visa_info, description="Provide visa requirements for countries"),
    Tool(name="PolicyRAG", func=refund_policy, description="Provide refund and cancellation policies")
]

agent = initialize_agent(tools, ChatOpenAI(), agent="zero-shot-react-description", verbose=True)

# --- Exposed function for Gradio ---
def travel_assistant_response(user_query: str) -> str:
    """Interface for Gradio to interact with the agent."""
    return agent.run(user_query)

# ------------------ Sample Execution ------------------
# Note: this segment of code is just for testing, the app can be launched from gradio_app.py
def main():
    user_query_1 = "Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers."
    user_query_2 = "Can UAE citizens travel to Japan without a visa?"
    user_query_3 = "Earliest flight to Saudi Arabia?"

    print("\n--- Flight Search ---")
    response_1 = agent.run(user_query_1)
    print(response_1)

    print("\n--- Visa Question ---")
    response_2 = agent.run(user_query_2)
    print(response_2)

    print("\n--- Flight Search ---")
    response_3 = agent.run(user_query_3)
    print(response_3)

if __name__ == "__main__":
    main()