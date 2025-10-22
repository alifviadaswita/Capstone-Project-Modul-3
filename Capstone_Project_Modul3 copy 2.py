import pandas as pd
import os
import re
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain.chains import RetrievalQA
from langchain.schema import Document as LangchainDocument 
# ========== Load environment ==========
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
qdrant_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

os.environ["OPENAI_API_KEY"] = openai_key

# ========== Setup embeddings & LLM ==========
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========== Load dataset ==========
data_path = r"D:\Training AI\Capstone Project Modul 3\Capstone 3\Dataset\RESUME\Resume.csv"
df = pd.read_csv(data_path)

# ========== Data cleaning dasar ==========
# Hapus data kosong dan duplikat
data = df.dropna(subset=['ID', 'Resume_str', 'Resume_html', 'Category']).reset_index(drop=True)
data = data.drop_duplicates(subset=['ID', 'Resume_str', 'Resume_html', 'Category']).reset_index(drop=True)

# ========== Fungsi pembersihan teks ==========

def clean_text(text):
    # Hapus tag HTML
    text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ")
    # Ubah ke huruf kecil
    text = text.lower()
    # Hapus karakter aneh tapi pertahankan tanda baca dasar
    text = re.sub(r'[^a-z0-9.,!?%&()\-+\s]', ' ', text)
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Masukkan ke Variabel baru
clean_data = data.copy()
clean_data["Resume_str"] = clean_data["Resume_str"].apply(clean_text)
clean_data["Category"] = clean_data["Category"].apply(clean_text)

# ==========  Simpan hasil ke file baru ==========
output_path = os.path.join(os.path.dirname(data_path), "Resume_Clean.csv")
clean_data.to_csv(output_path, index=False, encoding='utf-8')


# ========== Konversi ke Document ==========
documents = []
for i in range(clean_data.shape[0]):
    doc = Document(
        page_content=f"{clean_data.loc[i, 'Resume_str']}\nKategori: {clean_data.loc[i, 'Category']}",
        metadata={
            "id": str(clean_data.loc[i, 'ID']),
            "category": str(clean_data.loc[i, 'Category'])
        },
    )
    documents.append(doc)

# ========== Koneksi ke Qdrant ==========
client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

collection_name = "resume_documents"
# Hapus collection lama agar tidak ada duplikasi data
try:
    client.delete_collection(collection_name=collection_name)
    print("🗑️ Koleksi lama 'resume_documents' dihapus dari Qdrant.")
except Exception as e:
    print(f"⚠️ Tidak ada koleksi lama untuk dihapus atau terjadi error: {e}")

# Buat collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
 


####Vector Database Process
# ========== Simpan embeddings ke Qdrant ==========
vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    url=qdrant_url,
    api_key=qdrant_key,
    collection_name=collection_name,
    timeout=120
)

# print(f"✅ {len(documents)} dokumen berhasil disimpan ke Qdrant!")

# # ========== Cek koleksi di Qdrant ==========
collections_response = client.get_collections()
print("\n📚 Koleksi yang tersedia di Qdrant:")
for col in collections_response.collections:
    print("-", col.name)

#### 2. RAG TOOL
# ==========  retriever & chain RetrievalQA ==========
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#  chain RetrievalQA (LLM + retriever)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def ask_query(query: str) -> str:
    """
    Menjalankan RetrievalQA untuk menjawab query dari vectorstore.
    Returns: jawaban string dari chain.
    """
    if not isinstance(query, str) or query.strip() == "":
        return "Masukkan pertanyaan yang valid."
    try:
        result = qa_chain.run(query)
        return result
    except Exception as e:
        return f"Terjadi error saat query: {e}"


if __name__ == "__main__":
    print("🚀 RAG RetrievalQA siap! (ketik 'exit' untuk keluar)\n")
    while True:
        q = input("Masukkan pertanyaan: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Keluar. Sampai jumpa!")
            break
        answer = ask_query(q)
        print("\n💬 Jawaban:\n", answer, "\n" + "-"*60 + "\n")
