import os, re, time
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from langchain.agents import initialize_agent, AgentType, Tool

# ENVIRONMENT CONFIG 

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
qdrant_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
os.environ["OPENAI_API_KEY"] = openai_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LOAD & CLEAN DATA 

data_path = r"D:\Training AI\Capstone Project Modul 3\Capstone 3\Dataset\RESUME\Resume.csv"
df = pd.read_csv(data_path)

# Hapus null & duplikat
data = df.dropna(subset=['ID', 'Resume_str', 'Resume_html', 'Category']).reset_index(drop=True)
data = data.drop_duplicates(subset=['ID', 'Resume_str', 'Resume_html', 'Category']).reset_index(drop=True)

# Fungsi pembersihan
def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stopwords = {"the", "and", "or", "a", "an", "in", "on", "for", "to", "of", "is", "are", "at", "this", "that"}
    return " ".join([w for w in text.split() if w not in stopwords])

clean_data = data.copy()
for col in ['Resume_str', 'Resume_html', 'Category']:
    clean_data[col] = clean_data[col].apply(clean_text)

# Simpan hasil ke file baru 
output_path = os.path.join(os.path.dirname(data_path), "Resume_Clean.csv")
clean_data.to_csv(output_path, index=False, encoding='utf-8')

# BUAT DOCUMENTS
documents = []
for i, row in clean_data.iterrows():
    doc = Document(
        page_content=f"{row['Resume_str']}\nKategori: {row['Category']}",
        metadata={"id": str(row.get('ID', i)), "category": str(row['Category'])}
    )
    documents.append(doc)

# KONEKSI QDRANT 
client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)
collection_name = "resume_documents"
client = QdrantClient(url=qdrant_url, timeout=60.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

collections = [c.name for c in client.get_collections().collections]

collection_name = "resume_documents"

client = None
try:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)
    collections = [c.name for c in client.get_collections().collections]
    print("✅ Terhubung ke Qdrant. Koleksi yang tersedia:", collections)

if collection_name not in collections:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print(f"✅ Koleksi '{collection_name}' dibuat.")

except UnexpectedResponse as e:
    print("❌ Gagal mengakses Qdrant: Forbidden (403). Periksa API Key atau izin akses!")
    print("Pesan:", e)
    client = None
except Exception as e:
    print("⚠️ Tidak dapat terhubung ke Qdrant:", e)
    client = None

# Jika gagal koneksi, lanjut dengan mode fallback tanpa Qdrant
if client is not None:
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        try:
            QdrantVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                url=qdrant_url,
                api_key=qdrant_key,
                collection_name=collection_name
            )
            print(f"✅ Batch {i//batch_size + 1} upload sukses ({len(batch)} dokumen)")
        except Exception as e:
            print(f"⚠️ Batch {i//batch_size + 1} gagal: {e}")
else:
    print("⚠️ Lewati upload batch karena Qdrant tidak aktif.")

# RAG dengan fallback

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def retrieve_from_qdrant(query: str):
    """Ambil data dari Qdrant, Jika kosong atau error → fallback ke LLM."""
    if client is None:
        print("⚠️ Qdrant tidak aktif. Jawaban diambil langsung dari LLM.")
        response = llm.invoke(f"Jawab pertanyaan ini secara umum: {query}")
        return response.content

    try:
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
        docs = vectorstore.similarity_search(query, k=3)

        if not docs:
            print("⚠️ Tidak ada hasil dari Qdrant, fallback ke LLM.")
            response = llm.invoke(
                f"Tidak ada jawaban dalam database. Jawab pertanyaan ini secara umum: {query}"
            )
            return response.content

        results = [f"Result {i+1}: {doc.page_content[:300]}..." for i, doc in enumerate(docs)]
        return "\n\n".join(results)

    except Exception as e:
        print("⚠️ Error saat query ke Qdrant:", e)
        response = llm.invoke(
            f"Gagal mengakses database Qdrant. Jawab pertanyaan ini secara umum: {query}"
        )
        return response.content
    
# Buat Agent

rag_tool = Tool(
    name="Qdrant_RAG_Search",
    func=retrieve_from_qdrant,
    description="Gunakan ini untuk mencari jawaban dari database Qdrant. Akan fallback ke ChatGPT jika tidak ada hasil."
)

agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

print("🚀 Agent siap digunakan!")


# Tes

if __name__ == "__main__":
    while True:
        query = input("\n💬 Masukkan pertanyaan (atau ketik 'exit'): ")
        if query.lower() == 'exit':
            break
        try:
            response = agent.run(query)
            print("\n🧠 Jawaban Agent:\n", response)
        except Exception as e:
            print(f"⚠️ Terjadi error: {e}")