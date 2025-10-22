import os
import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
from datetime import datetime

# Load API keys
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
except FileNotFoundError:
    from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not all([OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    st.error("❌ Missing API keys!")
    st.stop()

from langchain_openai import ChatOpenAI # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceEmbeddings # pyright: ignore[reportMissingImports]
from langchain_qdrant import QdrantVectorStore # pyright: ignore[reportMissingImports]
from langchain.tools import tool # pyright: ignore[reportMissingImports]
from langgraph.prebuilt import create_react_agent # pyright: ignore[reportMissingImports]
from langchain_core.messages import ToolMessage, HumanMessage # pyright: ignore[reportMissingImports]

# Initialize LLM (only uses OpenAI for chat, not embedding)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.7
)

# Initialize FREE local embeddings (no API cost!)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings = load_embeddings()

# Initialize Qdrant Vector Store
@st.cache_resource
def load_qdrant():
    try:
        collection_name = "imdb_movies"
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False
        )
        return qdrant
    except Exception as e:
        st.error(f"❌ Failed to connect to Qdrant: {str(e)}")
        st.info("Make sure you ran: python scripts/create_vector_db.py")
        st.stop()

qdrant = load_qdrant()

# Define RAG Tools
@tool
def search_movies_by_title(query: str):
    """Search for movies by title, genre, or director."""
    results = qdrant.similarity_search(query, k=5)
    return results

@tool
def search_movies_by_rating(query: str):
    """Search for highly-rated movies by IMDB rating."""
    results = qdrant.similarity_search(f"IMDB rating {query}", k=5)
    return results

@tool
def search_movies_by_actor(query: str):
    """Search for movies by actor or star name."""
    results = qdrant.similarity_search(f"actor star {query}", k=5)
    return results

@tool
def search_movies_by_year(query: str):
    """Search for movies released in a specific year."""
    results = qdrant.similarity_search(f"released year {query}", k=5)
    return results

@tool
def search_movies_by_genre(query: str):
    """Search for movies by genre like Action, Drama, Comedy."""
    results = qdrant.similarity_search(f"genre {query}", k=5)
    return results

tools = [
    search_movies_by_title,
    search_movies_by_rating,
    search_movies_by_actor,
    search_movies_by_year,
    search_movies_by_genre
]

def create_agent_prompt():
    return """You are an expert IMDB movie recommendation assistant.

Guidelines:
1. Use the provided tools to search for relevant movies
2. Provide recommendations with clear reasoning
3. Use appropriate tools for different query types
4. Include: title, year, IMDB rating, genre, director, stars, overview
5. Be conversational and helpful
6. Cite movie details from retrieved documents

Focus on providing accurate IMDB dataset information."""

def chat_movie_expert(question):
    """Process user query with agent"""
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=create_agent_prompt()
    )
    
    try:
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        answer = result["messages"][-1].content
        
        # Calculate token usage
        total_input_tokens = 0
        total_output_tokens = 0
        
        for message in result["messages"]:
            if hasattr(message, "response_metadata"):
                if "usage_metadata" in message.response_metadata:
                    total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
                    total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
                elif "token_usage" in message.response_metadata:
                    total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
                    total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)
        
        usd_price = (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000
        idr_price = usd_price * 17_000
        
        tool_messages = []
        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                tool_messages.append(message.content)
        
        return {
            "answer": answer,
            "idr_price": idr_price,
            "usd_price": usd_price,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tool_messages": tool_messages,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "idr_price": 0,
            "usd_price": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "tool_messages": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Streamlit UI
st.set_page_config(
    page_title="IMDB Movie Assistant",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 IMDB Movie Recommendation Agent")
st.markdown("AI-powered movie recommendations using RAG Agent")

with st.sidebar:
    st.header("About")
    st.info("""
    **Tech Stack:**
    - LLM: GPT-4o Mini
    - Embeddings: HuggingFace (FREE, local)
    - Vector DB: Qdrant Cloud
    - Framework: LangChain + LanGraph
    - Dataset: IMDB Top 1000 Movies
    """)
    st.success("✅ Connected to Qdrant Cloud")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about movies, actors, directors, genres..."):
    with st.chat_message("Human"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    with st.chat_message("AI"):
        with st.spinner("Searching movies..."):
            response = chat_movie_expert(prompt)
        st.markdown(response["answer"])
    st.session_state.messages.append({"role": "AI", "content": response["answer"]})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("🔧 Tool Calls"):
            if response["tool_messages"]:
                for i, tool_msg in enumerate(response["tool_messages"], 1):
                    st.text_area(f"Result {i}", value=str(tool_msg), height=100, disabled=True)
            else:
                st.info("No tools called")
    
    with col2:
        with st.expander("📊 Token Usage"):
            st.metric("Input Tokens", response["total_input_tokens"])
            st.metric("Output Tokens", response["total_output_tokens"])
    
    with col3:
        with st.expander("💰 Cost"):
            st.metric("Cost (IDR)", f"Rp {response['idr_price']:.2f}")
            st.metric("Cost (USD)", f"${response['usd_price']:.6f}")

st.markdown("---")
st.caption("IMDB Movie RAG Agent | Powered by LangChain + Qdrant")