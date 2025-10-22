import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from uuid import uuid4
import traceback

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool

# try:
#     from langchain.output_parsers import StructuredOutputParser, ResponseSchema
#     STRUCTURED_PARSER_AVAILABLE = True
# except Exception:
#     STRUCTURED_PARSER_AVAILABLE = False

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
qdrant_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

if not openai_key or not qdrant_key or not qdrant_url:
    raise EnvironmentError("Pastikan OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_URL sudah diset di environment")

os.environ["OPENAI_API_KEY"] = openai_key

# ========== Setup embeddings & LLM ==========
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========== Load dataset ==========
data_path = r"D:\Training AI\Capstone Project Modul 3\Capstone 3\Dataset\RESUME\Resume.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File tidak ditemukan: {data_path}")

df = pd.read_csv(data_path, dtype=str).fillna("")

data = df.dropna(subset=['ID']).reset_index(drop=True)
data = data.drop_duplicates(subset=['ID']).reset_index(drop=True)

# ========== Data cleaning ==========

def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ")     # Hapus tag HTML
    text = text.lower()   # Ubah ke huruf kecil
    text = re.sub(r'[^a-z0-9.,!?%&()\-+\s]', ' ', text)  # Hapus karakter aneh tapi pertahankan tanda baca dasar
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    return text

# Masukkan ke Variabel baru
clean_data = data.copy()
clean_data["Resume_str"] = clean_data["Resume_str"].apply(clean_text)
clean_data["Category"] = clean_data["Category"].apply(clean_text)

output_path = os.path.join(os.path.dirname(data_path), "Resume_Clean.csv")
clean_data.to_csv(output_path, index=False, encoding="utf-8")
print(f"✅ Data dibersihkan dan disimpan ke: {output_path}")

# ========== Konversi ke Document ==========
documents = []
for i, row in clean_data.iterrows():
    content = f"{row['Resume_str']}\n\nKategori: {row['Category']}"
    metadata = {
        "id": str(row['ID']),
        "category": row.get('Category', "")
    }

    metadata["html_length"] = len(str(row.get("Resume_html","")))
    documents.append(Document(page_content=content, metadata=metadata))

print(f"🔎 Total dokumen yang akan di-index: {len(documents)}")
if len(documents) == 0:
    raise ValueError("Tidak ada dokumen untuk di-index. Periksa proses pembersihan.")

# ========== Koneksi ke Qdrant ==========
client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
collection_name = "resume_documents"

# HAPUS collection lama jika ada (agar tidak duplikat)
try:
    client.delete_collection(collection_name=collection_name)
    print(f"🗑️ Koleksi lama '{collection_name}' telah dihapus.")
except Exception as e:
    # Jika tidak ada koleksi, qdrant dapat throw error —   agar script tetap lanjut
    print(f"⚠️ Tidak dapat menghapus koleksi lama (mungkin tidak ada): {e}")

# Buat collection baru
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print(f"✅ Koleksi '{collection_name}' dibuat (vector size=1536).")
except Exception as e:
    print("❌ Gagal membuat koleksi baru:", e)
    raise

# ========== Simpan embeddings ke Qdrant ==========
print("↳ Mengindeks dokumen ke Qdrant. Proses ini mungkin memakan waktu beberapa menit jika dataset besar...")
vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    url=qdrant_url,
    api_key=qdrant_key,
    collection_name=collection_name,
    timeout=120
)
print("✅ Dokumen berhasil diindeks ke Qdrant (via QdrantVectorStore).")

try:
    count_info = None
    try:
        count_info = client.count(collection_name=collection_name)
        print(f"📦 Jumlah points di koleksi: {count_info.count if hasattr(count_info, 'count') else count_info}")
    except Exception:
        pts = client.scroll(collection_name=collection_name, limit=1)
        if pts:
            print("📦 Koleksi terisi (minimal 1 point ditemukan).")
except Exception as e:
    print("⚠️ Tidak bisa mengecek jumlah points:", e)


retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    output_key="answer"   
)

print("🧠 ConversationalRetrievalChain siap (dengan memory).")

# ========== Utility: fungsi tanya ==========
def ask_query_with_sources(question: str):
    if not question or not str(question).strip():
        return {"answer": "Masukkan pertanyaan yang valid.", "source_documents": []}
    try:
        resp = conv_chain.invoke({"question": question})
        answer = resp.get("answer") or resp.get("result") or str(resp)
        sources = resp.get("source_documents") or []
        return {"answer": answer, "source_documents": sources}
    except Exception as e:
        traceback.print_exc()
        return {"answer": f"Terjadi error saat query: {e}", "source_documents": []}


# Daftarkan retriever sebagai tool
tools = [
    Tool(
        name="Resume Retriever",
        func=lambda q: ask_query_with_sources(q)["answer"],
        description="Gunakan untuk mencari informasi dari resume kandidat berdasarkan pertanyaan pengguna."
    )
]

# Prompt sistem 
system_prompt = """
Kamu adalah Resume Intelligence Agent yang bertugas menjawab pertanyaan tentang kandidat berdasarkan isi dokumen resume yang tersimpan dalam basis data (Qdrant vector store).

🎯 Tujuan:
Berikan jawaban yang akurat, ringkas, dan berbasis bukti dari resume kandidat yang relevan.

🧩 Sumber informasi:
Kamu dapat menggunakan alat bernama "Resume Retriever" untuk mencari informasi dari resume kandidat. 
Gunakan tool ini setiap kali kamu membutuhkan data faktual dari resume (misalnya pengalaman kerja, pendidikan, sertifikasi, atau keahlian).

🧭 Panduan berpikir:
1. Analisis pertanyaan pengguna terlebih dahulu.
2. Tentukan apakah kamu perlu mencari informasi di resume menggunakan "Resume Retriever".
3. Jika ya, panggil tool tersebut dengan kata kunci yang sesuai (misalnya nama posisi, bidang pekerjaan, keterampilan).
4. Gabungkan hasil pencarian resume menjadi jawaban final.
5. Jika pertanyaan tidak berkaitan dengan isi resume, jawab secara sopan bahwa data tidak tersedia.

🧾 Format keluaran:
Berikan jawaban dengan struktur berikut:
- **Jawaban utama:** penjelasan singkat dan jelas (maksimum 5 kalimat)
- **Rangkuman kunci:** poin-poin keterampilan, posisi, atau pengalaman utama
- **Sumber:** ID dan kategori dokumen resume yang digunakan (jika tersedia)

🧠 Aturan tambahan:
- Jangan berasumsi di luar isi resume.
- Gunakan data yang ditemukan dari retriever sebagai dasar utama jawaban.
- Jika resume tidak memuat informasi tersebut, katakan dengan sopan bahwa informasinya tidak tercantum.
"""

# Buat agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    system_message=system_prompt
)

print("🤖 Resume Agent siap digunakan!\n")

# if __name__ == "__main__":
#     print("🚀 Resume Agent siap! (ketik 'exit' untuk keluar)\n")
#     while True:
#         q = input("❓ Pertanyaan: ").strip()
#         if q.lower() in {"exit", "quit"}:
#             print("👋 Keluar dari program. Sampai jumpa!")
#             break
#         response = agent.run(q)
#         print("\n💬 Jawaban:\n", response)
#         print("-" * 80)



#### Streamlit UI
st.set_page_config(page_title="Resume Intelligence Agent", page_icon="🧠")
st.title("🧠 Resume Intelligence Agent")
st.markdown("""
Masukkan pertanyaan tentang **resume kandidat**.  
Agent akan menjawab berdasarkan informasi yang tersimpan di Qdrant.
""")

user_query = st.text_input("🗣️ Pertanyaan kamu:")

if user_query:
    with st.spinner("Sedang mencari jawaban..."):
        response = agent.run(user_query)
        st.markdown("### ✅ Jawaban")
        st.success(response)

        # tampilkan sumber
        result = ask_query_with_sources(user_query)
        docs = result["source_documents"]
        if docs:
            st.markdown("### 📚 Sumber Dokumen:")
            for i, doc in enumerate(docs, 1):
                meta = getattr(doc, "metadata", {})
                snippet = doc.page_content[:400].replace("\n", " ")
                st.markdown(f"**{i}.** `{meta.get('id', 'tanpa id')}` — {snippet}...")
        else:
            st.info("Tidak ada dokumen sumber yang relevan ditemukan.")