# 📂 Blok 1: Import Library
import os, re, pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# 🔑 Blok 2: Load Environment Variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not all([OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise EnvironmentError("❌ Missing API keys. Please check .env file.")

# 🧠 Blok 3: Inisialisasi Model Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# 📑 Blok 4: Load Dataset
print("📂 Loading Resume dataset...")
data_path = "Dataset/RESUME/Resume.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

df = pd.read_csv(data_path, dtype=str).fillna("")
data = df.dropna(subset=['ID']).drop_duplicates(subset=['ID']).reset_index(drop=True)

# 🧹 Blok 5: Data cleaning
required_cols = ["ID", "Resume_str", "Category"]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise ValueError(f"❌ Missing required columns in dataset: {missing}")

def clean_text(text: str) -> str:
    """Remove HTML tags, lowercasing, and keep alphanumeric punctuation."""
    text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ")
    text = re.sub(r'[^a-z0-9.,!?%&()\-+\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

data["Resume_str"] = data["Resume_str"].apply(clean_text)
data["Category"] = data["Category"].apply(clean_text)

output_path = os.path.join(os.path.dirname(data_path), "Resume_Clean.csv")
data.to_csv(output_path, index=False, encoding="utf-8")
print(f"✅ Cleaned dataset saved to: {output_path}")


# 📄 Blok 6: Convert rows into LangChain Documents
documents = [
    Document(
        page_content=f"{row['Resume_str']}\n\nCategory: {row['Category']}",
        metadata={
            "id": str(row["ID"]).strip(),
            "category": str(row["Category"]).strip(),
            "html_length": len(str(row.get("Resume_html", "")))
        }
    )
    for _, row in data.iterrows()
]

if not documents:
    raise ValueError("❌ No documents to index!")

print(f"🔎 Total documents prepared: {len(documents)}")
print("📘 Sample document:", documents[0].metadata)


# 🔗 Blok 7:  Connect to Qdrant
collection_name = "resume_documents"

print("🔗 Connecting to Qdrant Cloud...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

# Create a new collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
print(f"✅ Collection '{collection_name}' created (dimension: 1536).")


#🚀 Blok 8:  Index embeddings into Qdrant
print("🚀 Indexing documents into Qdrant (this may take a few minutes)...")
vector_store = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    location=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=collection_name,
    prefer_grpc=False,
    batch_size=20,  
    timeout=60    
)
print("✅ Documents successfully uploaded to Qdrant!")

# 🧭 Blok 9: Create Payload Index
for field in ["category", "id"]:
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema="keyword"
        )
        print(f"✅ Index created for field '{field}'")
    except Exception as e:
        print(f"⚠️ Could not create index for '{field}': {e}")


# 🔍 Cek info koleksi
try:
    collection_info = client.get_collection(collection_name)
    vector_size = getattr(
        getattr(collection_info.config.params.vectors, "size", None),
        "size",
        1536
    )

    print("\n📊 Collection Info:")
    print(f"  - Name: {collection_name}")
    print(f"  - Vector Count: {collection_info.points_count}")
    print(f"  - Vector Dimension: {vector_size}")

    # hits, _ = client.scroll(collection_name=collection_name, limit=3)
    # print("\n🔎 Sample payloads:")
    # for point in hits:
    #     print(f"   - ID: {point.payload.get('id', '<no id>')}, Category: {point.payload.get('category', '<no category>')}")

except Exception as e:
    print(f"❌ Error creating vector database: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check if QDRANT_URL is correct (should start with https://)")
    print("2. Check if QDRANT_API_KEY is valid")
    print("3. Check if the dataset file exists at Dataset/RESUME/Resume.csv")
    print("4. Check if your OpenAI API key is valid and has sufficient credits")
    exit(1)


# Done
print("\n" + "="*50)
print("✅ SETUP COMPLETE!")
print("="*50)
print("\nYou can now run the main Streamlit app using:")
print("  streamlit run app.py")
