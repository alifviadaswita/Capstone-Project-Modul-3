import pandas as pd
import os
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from bs4 import BeautifulSoup

### 1. Vector Database
# ========== 1. Load environment ==========
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
qdrant_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

os.environ["OPENAI_API_KEY"] = openai_key

# ========== 2. Setup embeddings & LLM ==========
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========== 3. Load dataset ==========
data_path = r"D:\Training AI\Capstone Project Modul 3\Capstone 3\Dataset\RESUME\Resume.csv"
df = pd.read_csv(data_path)

# print(f"✅ Dataset berhasil dimuat dengan shape {df.shape}")
# print(df.head())

# ========== 4. Data cleaning ==========
data = df.dropna(subset=['ID', 'Resume_str', 'Resume_html', 'Category']).reset_index(drop=True)
data = data.drop_duplicates(subset=['ID', 'Resume_str', 'Resume_html', 'Category']).reset_index(drop=True)
# print(f"✅ Setelah pembersihan: {data.shape}")

def remove_html_tags_bs(data):
    return BeautifulSoup(str(data), "html.parser").get_text()

df['Resume_clean'] = df['Resume_html'].apply(remove_html_tags_bs)


print (df)
# # ========== 5. Konversi ke Document ==========
# documents = []
# for i in range(data.shape[0]):
#     doc = Document(
#         page_content=f"{data['Resume_str'][i]}\nKategori: {data['Category'][i]}",
#         metadata={
#             "id": str(data['ID'][i]),
#             "category": str(data['Category'][i])
#         },
#     )
#     documents.append(doc)

# # print(f"📄 Total dokumen: {len(documents)}")

# # ========== 6. Koneksi ke Qdrant ==========
# client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

# # Buat collection kalau belum ada
# collection_name = "resume_documents"
# if collection_name not in [col.name for col in client.get_collections().collections]:
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
#     )
#     # print(f"🆕 Koleksi '{collection_name}' berhasil dibuat di Qdrant!")


# ####Vector Database Process
# # ========== 7. Simpan embeddings ke Qdrant ==========
# vectorstore = QdrantVectorStore.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     url=qdrant_url,
#     api_key=qdrant_key,
#     collection_name=collection_name,
# )

# # print(f"✅ {len(documents)} dokumen berhasil disimpan ke Qdrant!")

# # # ========== 8. Cek koleksi di Qdrant ==========
# # collections_response = client.get_collections()
# # print("\n📚 Koleksi yang tersedia di Qdrant:")
# # for col in collections_response.collections:
# #     print("-", col.name)




# ###### 2. RAG TOOL
# # --- Define retrieval function ---
# def retrieve_from_qdrant(query: str):
#     docs = vectorstore.similarity_search(query, k=3)
#     results = [f"Result {i+1}: {doc.page_content[:300]}..." for i, doc in enumerate(docs)]
#     return "\n\n".join(results)
# # --- Define RAG tool ---
# rag_tool = Tool(
#     name="RAG_QdrantRetriever",
#     func=retrieve_from_qdrant,
#     description="Gunakan tool ini untuk mencari informasi dari vector database Qdrant. Input berupa pertanyaan natural language."
# )

# # --- Build agent ---
# tools = [rag_tool]
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
# )

# # --- Example query ---
# if __name__ == "__main__":
#     print("🚀 Agent siap digunakan!\n")
#     query = input("Masukkan pertanyaan: ")
#     response = agent.run(query)
#     print("\n💬 Jawaban Agent:\n", response)
