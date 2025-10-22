import pandas as pd # pyright: ignore[reportMissingModuleSource]
import os
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from langchain_openai import OpenAIEmbeddings # pyright: ignore[reportMissingImports]
from langchain_qdrant import QdrantVectorStore # pyright: ignore[reportMissingImports]
from langchain_core.documents import Document # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient # pyright: ignore[reportMissingImports]
from qdrant_client.http.models import Distance, VectorParams # pyright: ignore[reportMissingImports]

# Load environment variables
load_dotenv()

# Get API keys from .env file
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

print("✅ API keys loaded successfully\n")

# Initialize Qdrant client
print("Connecting to Qdrant Cloud...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

collection_name = "Resume"

# Check if collection already exists
try:
    existing_collection = client.get_collection(collection_name)
    print(f"⚠️  Collection '{collection_name}' already exists")
    user_input = input("Do you want to delete and recreate it? (yes/no): ").lower()
    
    if user_input == 'yes':
        print(f"Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name)
        print("✅ Collection deleted")
    else:
        print("Aborting operation")
        exit(0)
except Exception as e:
    print(f"Collection doesn't exist yet. Will create new one.\n")

# Initialize embeddings
print("Initializing OpenAI embeddings...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
    dimensions=1536
)

# Load IMDB dataset
print("Loading IMDB dataset...")
try:
    df = pd.read_csv("data/imdb_top_1000.csv")
    print(f"✅ Dataset loaded: {len(df)} movies")
except FileNotFoundError:
    print("❌ Error: data/imdb_top_1000.csv not found")
    print("Please download the dataset and place it in the data/ folder")
    exit(1)

print("\nDataset columns:", df.columns.tolist())
print("First row sample:")
print(df.iloc[0])

# Prepare documents for embedding
print("\n📝 Preparing documents for embedding...")
documents = []

for idx, row in df.iterrows():
    # Handle NaN values
    title = str(row.get('Series_Title', 'N/A'))
    year = str(row.get('Released_Year', 'N/A'))
    genre = str(row.get('Genre', 'N/A'))
    rating = str(row.get('IMDB_Rating', 'N/A'))
    director = str(row.get('Director', 'N/A'))
    overview = str(row.get('Overview', 'N/A'))
    runtime = str(row.get('Runtime', 'N/A'))
    certificate = str(row.get('Certificate', 'N/A'))
    
    # Get stars
    stars = []
    for i in range(1, 5):
        star = row.get(f'Star{i}', None)
        if pd.notna(star):
            stars.append(str(star))
    stars_str = ", ".join(stars) if stars else "N/A"
    
    # Combine all information for embedding
    content = f"""
Movie Title: {title}
Release Year: {year}
Genre: {genre}
IMDB Rating: {rating}
Director: {director}
Stars: {stars_str}
Runtime: {runtime}
Certificate: {certificate}
Overview: {overview}
    """.strip()
    
    # Create document with metadata
    doc = Document(
        page_content=content,
        metadata={
            "title": title,
            "year": int(year) if year.isdigit() else 0,
            "rating": float(rating) if rating.replace('.', '').isdigit() else 0.0,
            "genre": genre,
            "director": director,
            "stars": stars_str,
            "source": "IMDB"
        }
    )
    documents.append(doc)
    
    # Progress indicator
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(df)} documents")

print(f"✅ Total documents prepared: {len(documents)}\n")

# Create vector store and upload documents
print("🚀 Creating vector database in Qdrant Cloud...")
print("This may take a few minutes for 1000 documents...\n")

try:
    vector_store = QdrantVectorStore.from_documents(
        documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        prefer_grpc=False,
        batch_size=50  # Process in batches to avoid timeout
    )
    
    print("✅ Vector database created successfully!")
    
    # Verify collection
    collection_info = client.get_collection(collection_name)
    print(f"\n📊 Collection Statistics:")
    print(f"  - Collection Name: {collection_name}")
    print(f"  - Vector Count: {collection_info.points_count}")
    
    # Handle different attribute structures
    try:
        if hasattr(collection_info.config.params.vectors, 'size'):
            vector_size = collection_info.config.params.vectors.size
        elif isinstance(collection_info.config.params.vectors, dict):
            vector_size = collection_info.config.params.vectors.get('size', 'unknown')
        else:
            vector_size = collection_info.config.params.vectors.vectors.size if hasattr(collection_info.config.params.vectors, 'vectors') else 'unknown'
        print(f"  - Vector Dimension: {vector_size}")
    except Exception as e:
        print(f"  - Vector Dimension: 1536 (default)")
    
except Exception as e:
    print(f"❌ Error creating vector database: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check if QDRANT_URL is correct (should have https://)")
    print("2. Check if QDRANT_API_KEY is valid")
    print("3. Check if dataset file exists at data/imdb_top_1000.csv")
    print("4. Check OpenAI API key is valid and has enough credits")
    exit(1)

print("\n" + "="*50)
print("✅ SETUP COMPLETE!")
print("="*50)
print("\nYou can now run the main Streamlit app:")
print("  streamlit run main.py")