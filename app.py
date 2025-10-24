#⚙️ Blok 1: Import Library 
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
from qdrant_client.models import Filter, FieldCondition, MatchValue
import base64

st.set_page_config(
    page_title="RESUME Assistant",
    page_icon="🧾",
    layout="wide"
)
# 🔐 Blok 2: Load API keys
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


# 🧠 Blok 3:  Initialize LLM 
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

# 🗃️ Blok 4:  Initialize Qdrant Vector Store
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

retriever = qdrant.as_retriever(search_kwargs={"k": 20})

def _clean_category(value: str) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()

# 🧩 Blok 5: Define RAG Tools
# 1️⃣ Searh resume by Category
@tool
def search_resumes_by_category(category: str):
    """Search resumes by Category (e.g., HR, Finance, IT)."""
    try:
        cat = _clean_category(category)
        results = qdrant.similarity_search(
            query=cat,
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
        results = qdrant.similarity_search(query=query, k=100)
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

# 4️⃣ Search resumes by Prompt
@tool
def get_resume_by_prompt(query_prompt: str):
    """Get the most relevant resume for a given query prompt."""
    try:
        results = qdrant.similarity_search(query=f"Resume about {query_prompt}", k=100)
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

# 5️⃣ Search resumes similar candidates
@tool
def recommend_similar_candidates(query_prompt: str):
    """Recommend similar candidates based on a query prompt."""
    try:
        results = qdrant.similarity_search(query=f"Similar candidates to {query_prompt}", k=100)
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

# 6️⃣ Get Resume by ID
@tool
def get_resume_by_id(id: str):
    """Ambil resume lengkap berdasarkan ID kandidat."""
    try:
        results = qdrant.similarity_search_with_score(
            query="resume",
            k=3,
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="id",
                    match=models.MatchValue(value=str(id))
                )]
            )
        )
    except Exception as e:
        return {"error": f"Gagal mengambil resume: {str(e)}"}

    if not results:
        return f"Maaf, tidak ditemukan resume dengan ID {id}."

    doc, score = results[0]

    return {
        "ID": doc.metadata.get("id"),
        "Category": doc.metadata.get("category"),
        "Resume": (doc.page_content[:1500] + "...") if doc.page_content else "",
        "RelevanceScore": score
    }


tools = [
    search_resumes_by_category,
    search_resumes_by_skills,
    search_resumes_by_category_and_query,
    get_resume_by_prompt,
    recommend_similar_candidates,
    get_resume_by_id
]

# 🧠 Blok 6: 🧩 AGENT CREATION
def create_agent_prompt():
    return """
Anda adalah **Resume Intelligence Agent**, asisten ahli yang dirancang untuk menjawab pertanyaan tentang kandidat kerja berdasarkan dokumen resume yang disimpan di database vektor Qdrant.

🎯 Tujuan:
Berikan jawaban yang akurat, ringkas, dan berbasis bukti yang diambil dari resume kandidat yang paling relevan.

🧩 Sumber Informasi:
Anda memiliki akses ke tool bernama **"Resume Retriever"** yang dapat mencari dan mengekstrak informasi faktual dari resume (mis. pengalaman kerja, pendidikan, sertifikasi, keterampilan). 
Gunakan tool ini setiap kali perlu data dari resume.

🧭 Proses Penalaran:
1. Analisis pertanyaan pengguna dengan saksama.
2. Tentukan apakah perlu memanggil **Resume Retriever** untuk mengambil informasi dari resume.
3. Bila perlu, panggil tool dengan kata kunci yang tepat (mis. judul pekerjaan, departemen, keterampilan, nama kandidat).
4. Rangkum dan sintesis data yang diambil menjadi jawaban yang jelas dan faktual.
5. Jika pertanyaan tidak berhubungan dengan isi resume, nyatakan dengan sopan bahwa informasi tersebut tidak tersedia di resume.

🧾 Format Keluaran (WAJIB):
- **Main Answer:** penjelasan singkat dan faktual (maksimum 5 kalimat).
- **Key Summary:** poin-poin (bullet) yang memuat keterampilan utama, peran, atau pengalaman.
- **Sources:** ID kandidat dan kategori dokumen yang digunakan (jika tersedia).

🧠 Aturan Tambahan (WAJIB):
- Jangan membuat atau mengasumsikan informasi di luar apa yang ditemukan pada resume.
- Semua jawaban harus berdasar secara ketat pada data yang diambil.
- Jika informasi yang diminta tidak ditemukan di resume mana pun, jawab dengan jelas bahwa data tidak tersedia.
- Gunakan nada profesional, membantu, dan faktual.
- **Selalu** jawab dalam bahasa Indonesia.

Instruksi teknis tambahan untuk integrasi agen:
- Ketika memanggil Resume Retriever, sertakan kata kunci pencarian dan batas jumlah dokumen yang ingin diambil (mis. top 3).
- Jika Resume Retriever mengembalikan banyak dokumen, prioritaskan bukti yang paling relevan dan terbaru.
- Sertakan kutipan singkat (mis. baris atau fragmen) dari resume bila perlu untuk mendukung klaim — tetapi hanya bila tool menyediakan fragmen tersebut.

Contoh panggilan tool (pseudo):
ResumeRetriever.search(query="data scientist python", top_k=3)

Gunakan prompt ini sebagai pedoman perilaku agen setiap kali menjawab pertanyaan tentang kandidat.
"""
#💬 Blok 7: Function resume expert
def resume_expert(question: str, history: str = ""):
    """Process user query with agent"""
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=create_agent_prompt()
    )

    prompt_with_context = f"""
    This is the previous chat history:
    {history}

    Now, answer this new question:
    {question}
    """

    try:
        result = agent.invoke({"messages": [HumanMessage(content=prompt_with_context)]})

        # Ambil pesan hasil agent
        messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
        messages_history = st.session_state.get("messages", [])[-20:]
        history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

        # Ambil hasil jawaban
        if not messages:
            answer_text = str(result)
        else:
            answer_text = messages[-1].content

        # Hitung token usage
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

        # Estimasi biaya
        usd_price = (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000
        idr_price = usd_price * 17_000

        # Ambil tool message (jika ada)
        tool_messages = [m.content for m in messages if isinstance(m, ToolMessage)]

        return {
            "answer": answer_text,
            "idr_price": idr_price,
            "usd_price": usd_price,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tool_messages": tool_messages,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": history
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
            "history": history
        }

# 🧭 Blok 8: Streamlit UI
# Header
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        data = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{data}"

# Ganti 'image.png' sesuai nama file kamu
img_base64 = get_base64_image("image.png")

# Header banner
st.markdown(
    f"""
    <div style='
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 30px;
    '>
        <img src="{img_base64}" style='width:60%; max-width:600px; border-radius:10px; margin-bottom:20px;'>
        <h1 style='font-size: 42px; margin-bottom: 10px;'>🧠 RESUME Recommendation Agent</h1>
        <p style='font-size: 18px; color: #555;'>
            AI-powered resume recommendation using <b>RAG Agent</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
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

    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages])

    with st.chat_message("AI"):
        with st.spinner("Processing..."):
            response = resume_expert(prompt,history)
        st.markdown(response["answer"])
    st.session_state.messages.append({"role": "AI", "content": response["answer"]})

    with st.expander("History Chat"):
        for chat in st.session_state.messages:
            st.markdown(f"**{chat['role']}** : {chat['content']}")

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