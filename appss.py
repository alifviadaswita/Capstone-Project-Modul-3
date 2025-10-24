import os
import streamlit as st  
import pandas as pd  
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_qdrant import QdrantVectorStore  
from langchain.tools import tool 
from langchain.agents import create_agent 
from langchain_core.messages import ToolMessage, HumanMessage  
from qdrant_client.http import models


st.set_page_config(
    page_title="RESUME Assistant",
    page_icon="🧾",
    layout="wide"
)
# Load API keys
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
except Exception:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not all([OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    st.error("❌ Missing API keys! Please provide OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY.")
    st.stop()


# Initialize LLM 
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0
)

# Initialize FREE local embeddings 
@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY)

embeddings = load_embeddings()

# Initialize Qdrant Vector Store
@st.cache_resource
def load_qdrant():
    try:
        collection_name = "resume_documents"
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
        st.info("Make sure you ran: python scripts/main.py")
        st.stop()
        return None
    
qdrant = load_qdrant()
if qdrant is None:
    st.stop()

retriever = qdrant.as_retriever(search_kwargs={"k": 5})

def _clean_category(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()

# Define RAG Tools
# 1️⃣ Searh resume by Category
@tool
def search_resumes_by_category(category: str):
    """Search resumes by Category (e.g., HR, Finance, IT)."""
    try:
        cat = _clean_category(category)
        results = qdrant.similarity_search(
            query=cat,
            k=5
        )
    except Exception as e:
        return {"error": str(e)}

    if not results:
        return f"Tidak ada data untuk kategori '{category}'."   
    
    out = []
    for doc in results:
        out.append({
            "ID": doc.metadata.get("id"),
            "Category": doc.metadata.get("category"),
            "Snippet": (doc.page_content[:300] + "...") if doc.page_content else ""
        })
    return out

# 2️⃣ Search Resumes by Skills
@tool
def search_resumes_by_skills(skills: str):
    """Search resumes by free-text query (skills, keywords)."""
    try:
        query = f"Candidates skilled in {skills}"
        results = qdrant.similarity_search(
            query=query, 
            k=5)
    except Exception as e:
        return {"error": str(e)}

    if not results:
        return f"Tidak ditemukan kandidat dengan keterampilan '{skills}'."

    return [
        {
            "ID": doc.metadata.get("id"),
            "Category": doc.metadata.get("Category"),
            "Snippet": (doc.page_content[:300] + "...") if doc.page_content else ""
        } for doc in results
    ]


# 3️⃣ Search resumes by Category + query (filtered)
@tool
def search_resumes_by_category_and_query(category: str, query: str):
    """Search resumes by category and query (combined filter)."""
    try:
        cat = _clean_category(category)
        combined_query = f"{query} professional in {cat}"
        results = qdrant.similarity_search(
            query=combined_query,
            k=5,
        )
    except Exception as e:
        return {"error": str(e)}

    if not results:
        return f"Tidak ditemukan hasil untuk kategori '{category}' dengan query '{query}'."

    return [
        {
            "ID": doc.metadata.get("id"),
            "Category": doc.metadata.get("category"),
            "Excerpt": (doc.page_content[:300] + "...") if doc.page_content else ""
        } for doc in results
    ]

# 4️⃣ Search Resumes by Prompt

@tool
def get_resume_by_prompt(query_prompt: str):
    """Get the most relevant resume for a given query prompt."""
    try:
        results = qdrant.similarity_search(
            query=f"Resume about {query_prompt}", 
            k=1)
    except Exception as e:
        return {"error": str(e)}

    if not results:
        return f"Tidak ditemukan resume relevan dengan '{query_prompt}'."

    doc = results[0]
    return {
        "ID": doc.metadata.get("id"),
        "Category": doc.metadata.get("Category"),
        "Resume": (doc.page_content[:1000] + "...") if doc.page_content else ""
    }

# 5️⃣ Search Recommed Similar Candidates
@tool
def recommend_similar_candidates(query_prompt: str):
    """Recommend similar candidates based on a query prompt."""
    try:
        results = qdrant.similarity_search(
            query=f"Similar candidates to {query_prompt}",
            k=6)
    except Exception as e:
        return {"error": str(e)}

    if not results:
        return f"Tidak ada kandidat serupa untuk '{query_prompt}'."

    return [
        {
            "ID": doc.metadata.get("id"),
            "Category": doc.metadata.get("Category"),
            "Snippet": (doc.page_content[:300] + "...") if doc.page_content else ""
        } for doc in results
    ]


tools = [
    search_resumes_by_category,
    search_resumes_by_skills,
    search_resumes_by_category_and_query,
    get_resume_by_prompt,
    recommend_similar_candidates

]

def detect_language(text: str) -> str:
    """Detect whether text is Indonesian or English (simple heuristic)."""
    text_lower = text.lower()
    if any(word in text_lower for word in ["apa", "bagaimana", "mengapa", "siapa", "dimana", "kapan", "tidak", "dengan", "dan", "yang"]):
        return "indonesian"
    return "english"


# 🧩 AGENT CREATION
def create_agent_prompt(language: str):
    base_prompt = """
You are **Resume Intelligence Agent**, an expert assistant designed to answer questions about job candidates based on their resume documents stored in the Qdrant vector database.

🎯 **Objective:**
Provide accurate, concise, and evidence-based answers derived from the most relevant resumes.

🧩 **Information Source:**
You have access to these tools:
- search_resumes_by_category
- search_resumes_by_skills
- search_resumes_by_category_and_query
- get_resume_by_prompt
- recommend_similar_candidates

Use them to retrieve data and answer the user’s questions.

🧭 **Reasoning Process:**
1. First, analyze the user’s question carefully.  
2. Determine if you need to retrieve resume information using the "Resume Retriever" tool.  
3. If yes, call the tool with appropriate keywords (e.g., job title, department, or skills).  
4. Summarize and synthesize retrieved resume data into a clear, factual answer.  
5. If the question is unrelated to the resume content, politely state that the information is not available.

🧭 **Behavior Rules for Counting:**
1. When the user asks "how many", "berapa", or "jumlah", you must:
   - Use one of the search tools above to get the data.
   - Count the number of returned results.
   - Respond with the total count and a short explanation.

2. Always provide clear answers based on resume data.  
3. Never respond “information not available” unless there are truly no results.  
4. Maintain a professional and factual tone.  
5. If the user requests examples (e.g., “tampilkan 5 kandidat di kategori IT”), provide up to 5 sample candidates.

🧾 **Output Format:**
Your response must follow this structure:
- **Main Answer:** concise and factual explanation (≤5 sentences)
- **Key Summary:** bullet points of main skills, roles, or experiences
- **Sources:** candidate IDs and categories used (if available)

🧠 **Additional Rules:**
- Do NOT invent or assume information beyond what is found in resumes.  
- Base all answers strictly on retrieved data.  
- If the requested information is not found in any resume, respond clearly that the data is not available.  
- Maintain a professional, helpful, and factual tone at all times.
"""
    if language == "indonesian":
        base_prompt += "\n\n🗣️ Semua hasil dan penjelasan ditulis dalam Bahasa Indonesia"
    else:
        base_prompt += "\n\n🗣️ Respond entirely in English."

    return base_prompt

def resume_expert(question: str):
    """Process user query with agent"""
    language = detect_language(question)

    # if language not in ["indonesian", "english"]:
    #     language = "indonesian"

    system_prompt = create_agent_prompt(language) 
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    try:
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
        if not messages:
            answer_text = str(result)
        else:
            answer_text = messages[-1].content

        # Calculate token usage (best-effort)
        total_input_tokens = 0
        total_output_tokens = 0
        for message in messages:
            meta = getattr(message, "response_metadata", None)
            if meta:
                if "usage_metadata" in meta:
                    um = meta["usage_metadata"]
                    total_input_tokens += um.get("input_tokens", 0)
                    total_output_tokens += um.get("output_tokens", 0)
                elif "token_usage" in meta:
                    tu = meta["token_usage"]
                    total_input_tokens += tu.get("prompt_tokens", 0)
                    total_output_tokens += tu.get("completion_tokens", 0)

        usd_price = (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000
        idr_price = usd_price * 17_000

        tool_messages = []
        for message in messages:
            if isinstance(message, ToolMessage):
                tool_messages.append(message.content)

        return {
            "answer": answer_text,
            "idr_price": idr_price,
            "usd_price": usd_price,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tool_messages": tool_messages,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language": language
        }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "idr_price": 0,
            "usd_price": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "tool_messages": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language": language
        }

# Streamlit UI

st.title("🧠 RESUME Recommendation Agent")
st.markdown("AI-powered resume recommendation using RAG Agent")

with st.sidebar:
    st.header("About")
    st.info("""
    **Tech Stack:**
    - LLM: GPT-4o Mini (OpenAI)
    - Embeddings: OpenAI text-embedding-3-small
    - Vector DB: Qdrant Cloud
    - Framework: LangChain + LangGraph
    - Dataset: RESUME
    """)
    st.success("✅ Connected to Qdrant Cloud")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about resumes, skills, categories, or candidate IDs..."):
    with st.chat_message("Human"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "Human", "content": prompt})

    with st.chat_message("AI"):
        with st.spinner("Processing..."):
            response = resume_expert(prompt)
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
st.caption("RESUME RAG Agent | Powered by LangChain + Qdrant")