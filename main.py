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

# Validate API keys
if not all([OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    print("❌ Error: Missing API keys in .env file")
    print("Make sure you have:")
    print("  - OPENAI_API_KEY")
    print("  - QDRANT_URL")
    print("  - QDRANT_API_KEY")
    exit(1)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# 🧠 Blok 3: Inisialisasi Model Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 📑 Blok 4: Load Dataset
print("📂 Loading Resume dataset...")
data_path = "Dataset/RESUME/Resume.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

df = pd.read_csv(data_path, dtype=str).fillna("")
data = df.dropna(subset=['ID']).reset_index(drop=True)
data = data.drop_duplicates(subset=['ID']).reset_index(drop=True)


# 🧹 Blok 5: Data cleaning
required_cols = ["ID", "Resume_str", "Category"]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise ValueError(f"❌ Missing required columns in dataset: {missing}")

def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ")  
    text = text.lower()  
    text = re.sub(r'[^a-z0-9.,!?%&()\-+\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

# Create cleaned version
clean_data = data.copy()
clean_data["Resume_str"] = clean_data["Resume_str"].apply(clean_text)
clean_data["Category"] = clean_data["Category"].apply(clean_text)

output_path = os.path.join(os.path.dirname(data_path), "Resume_Clean.csv")
clean_data.to_csv(output_path, index=False, encoding="utf-8")
print(f"✅ Data cleaned and saved to: {output_path}")


# 📄 Blok 6: Convert rows into LangChain Documents
documents = []
for i, row in clean_data.iterrows():
    content = f"{row['Resume_str']}\n\nCategory: {row['Category']}"
    metadata = {
        "id": str(row['ID']),
        "category": row.get('Category', "")
    }

    metadata["html_length"] = len(str(row.get("Resume_html", "")))
    documents.append(Document(page_content=content, metadata=metadata))

print(f"🔎 Total documents to index: {len(documents)}")
if len(documents) == 0:
    raise ValueError("No documents available for indexing. Please check the cleaning process.")


# 🔗 Blok 7:  Connect to Qdrant
collection_name = "resume_documents"

print("🔗 Connecting to Qdrant Cloud...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

# Delete old collection 
try:
    client.delete_collection(collection_name)
    print(f"⚠️ Old collection '{collection_name}' deleted.")
except:
    print("ℹ️ No existing collection found, continuing...")

# Create a new collection
client.create_collection(
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
try:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="category",
        field_schema="keyword"
    )
except Exception as e:
    
    # Verify collection info
    collection_info = client.get_collection(collection_name)
    print(f"\n📊 Collection Statistics:")
    print(f"  - Collection Name: {collection_name}")
    print(f"  - Vector Count: {collection_info.points_count}")

    # Handle various vector config structures
    try:
        if hasattr(collection_info.config.params.vectors, 'size'):
            vector_size = collection_info.config.params.vectors.size
        elif isinstance(collection_info.config.params.vectors, dict):
            vector_size = collection_info.config.params.vectors.get('size', 'unknown')
        else:
            vector_size = collection_info.config.params.vectors.vectors.size if hasattr(collection_info.config.params.vectors, 'vectors') else 'unknown'
        print(f"  - Vector Dimension: {vector_size}")
    except Exception as e:
        print("  - Vector Dimension: 1536 (default)")

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
