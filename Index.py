'''
# Run once (or whenever you add new documents to docs/)
python3 step1_index_docs.py

# Run to chat
python3 step3_rag_with_memory_and_search.py
'''
import os
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading embedding model (first run downloads ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!\n")

DOCS_DIR = "docs"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def read_documents(directory):
    """Read all .txt and .md files from a directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                text = f.read()
            documents.append({"filename": filename, "text": text})
            print(f"  Read: {filename} ({len(text)} chars)")
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def index_documents():
    """Read docs, chunk them, embed them, store in ChromaDB."""

    print("=" * 60)
    print("STEP 1: Indexing your documents")
    print("=" * 60)
    print()

    # Read all documents
    print("Reading documents...")
    docs = read_documents(DOCS_DIR)
    if not docs:
        print(f"No .txt or .md files found in '{DOCS_DIR}/' folder!")
        print("Add some text files there and run again.")
        return

    # Chunk all documents
    print("\nChunking documents...")
    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc['filename']}_chunk_{i}")
            all_metadata.append({"source": doc["filename"], "chunk_index": i})
        print(f"  {doc['filename']} → {len(chunks)} chunks")

    # Create embeddings and store in ChromaDB
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks).tolist()

    # Create/connect to ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete old collection if it exists, then create fresh
    try:
        client.delete_collection("my_documents")
    except:
        pass
    
    collection = client.create_collection(
        name="my_documents",
        metadata={"hnsw:space": "cosine"}
    )

    # Add everything to the collection
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadata,
    )

    print(f"\nDone! Stored {len(all_chunks)} chunks in ChromaDB.")
    print(f"Database saved to ./chroma_db/")

    # Test it with a sample query
    print("\n" + "=" * 60)
    print("TESTING: searching for 'what is an agent'")
    print("=" * 60)

    query = "what is an agent"
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    for i, (doc, score) in enumerate(
        zip(results["documents"][0], results["distances"][0])
    ):
        print(f"\nResult {i+1} (distance: {score:.4f}):")
        print(f"  {doc[:150]}...")


if __name__ == "__main__":
    index_documents()