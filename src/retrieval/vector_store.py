import json
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,        # how we measure similarity between vectors
    VectorParams,    # configuration for our vectors
    PointStruct,     # one single record in Qdrant (id + vector + metadata)
    Filter,          # for filtering results by company/date
    FieldCondition,  # one condition in a filter e.g. company == "apple"
    MatchValue       # the value to match e.g. "apple"
)

load_dotenv()

# Name of our collection inside Qdrant
# A collection is like a table in a normal database
# All our 1427 chunks will live in this one collection
COLLECTION_NAME = "finrag_sec_filings"

# The dimension of our vectors — must match the embedding model
# our model "all-MiniLM-L6-v2" produces 384-dimensional vectors
VECTOR_SIZE = 384


def create_qdrant_client() -> QdrantClient:
    """
    Create a connection to Qdrant.
    
    We are using Qdrant in LOCAL mode — it runs entirely on your 
    laptop, no internet needed, no account needed, completely free.
    
    It saves data to a folder called "qdrant_storage" on your disk
    so your vectors persist even after you close the program.
    
    In production at a real company, you'd point this to a 
    cloud Qdrant instance instead — just change the URL.
    """
    
    # This folder is where Qdrant will store all vector data on disk
    storage_path = "./qdrant_storage"
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    
    # QdrantClient in local mode — stores everything on your laptop
    client = QdrantClient(path=storage_path)
    print(f"✅ Qdrant client created (local mode, storage: {storage_path})")
    return client


def create_collection(client: QdrantClient):
    """
    Create a collection in Qdrant — like creating a table in SQL.
    
    We only need to do this ONCE. If it already exists, we skip it.
    
    We tell Qdrant two things:
    1. How big are our vectors? → 384 dimensions
    2. How do we measure similarity? → Cosine similarity
    
    Cosine similarity measures the ANGLE between two vectors.
    Vectors pointing in the same direction = similar meaning.
    This is the best choice for text embeddings.
    """
    
    # Get list of existing collections
    existing = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME in existing:
        # Collection already exists — no need to recreate
        print(f"✅ Collection '{COLLECTION_NAME}' already exists, skipping creation")
        return
    
    # Create a brand new collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,        # each vector has 384 numbers
            distance=Distance.COSINE # measure similarity using cosine
        )
    )
    print(f"✅ Created collection '{COLLECTION_NAME}' "
          f"(size={VECTOR_SIZE}, distance=cosine)")


def upload_chunks_to_qdrant(client: QdrantClient,
                             embedded_chunks: List[Dict],
                             batch_size: int = 100):
    """
    Upload all 1427 embedded chunks into Qdrant.
    
    Each chunk becomes one "Point" in Qdrant:
    - id: unique number identifying this point
    - vector: the 384 numbers representing meaning
    - payload: metadata (text, company, date etc.)
    
    We upload in batches of 100 to avoid memory issues.
    Think of it like moving house — you don't carry 
    everything at once, you use boxes (batches).
    """
    
    total = len(embedded_chunks)
    print(f"\nUploading {total} chunks to Qdrant in batches of {batch_size}...")
    
    # Process chunks in batches
    for batch_start in range(0, total, batch_size):
        # Get the current batch
        batch_end = min(batch_start + batch_size, total)
        batch = embedded_chunks[batch_start:batch_end]
        
        # Convert each chunk into a Qdrant PointStruct
        points = []
        for i, chunk in enumerate(batch):
            point = PointStruct(
                # Unique integer ID for this point
                # We use the global index across all chunks
                id=batch_start + i,
                
                # The 384-dimensional vector — this is what Qdrant
                # uses to find similar chunks during search
                vector=chunk["embedding"],
                
                # Payload = all the metadata we want to store alongside
                # the vector so we can return it with search results
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],          # actual text content
                    "company": chunk["company"],    # e.g. "apple"
                    "date": chunk["date"],          # e.g. "2024-11-01"
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "word_count": chunk["word_count"]
                }
            )
            points.append(point)
        
        # Upload this batch to Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        print(f"  Uploaded batch {batch_start//batch_size + 1} "
              f"({batch_end}/{total} chunks)")
    
    print(f"\n✅ All {total} chunks uploaded to Qdrant!")


def search_similar_chunks(client: QdrantClient,
                          query_vector: List[float],
                          top_k: int = 5,
                          company_filter: str = None) -> List[Dict]:
    """
    Search Qdrant for the most similar chunks to a query vector.
    Uses the new query_points API (Qdrant v1.7+)
    """
    
    # Build optional company filter
    search_filter = None
    if company_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="company",
                    match=MatchValue(value=company_filter)
                )
            ]
        )
    
    # New Qdrant API uses query_points instead of search
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,          # the question's 384-number vector
        limit=top_k,                 # return top K most similar chunks
        query_filter=search_filter,  # optional company filter
        with_payload=True            # include text and metadata in results
    ).points                         # .points extracts the list from response
    
    # Format results into clean dictionaries
    formatted_results = []
    for result in results:
        formatted_results.append({
            "score": result.score,
            "text": result.payload["text"],
            "company": result.payload["company"],
            "date": result.payload["date"],
            "chunk_id": result.payload["chunk_id"]
        })
    
    return formatted_results


def get_collection_stats(client: QdrantClient):
    """
    Print stats about our Qdrant collection.
    Shows how many vectors are stored and collection config.
    """
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n--- QDRANT COLLECTION STATS ---")
    print(f"Collection name : {COLLECTION_NAME}")
    print(f"Total vectors   : {info.points_count}")
    print(f"Vector size     : {info.config.params.vectors.size}")
    print(f"Distance metric : {info.config.params.vectors.distance}")
    print(f"Index status    : {info.status}")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from sentence_transformers import SentenceTransformer

    # Step 1: Load our embedded chunks from disk
    print("=== STEP 1: LOADING EMBEDDED CHUNKS ===")
    with open("data/processed/embedded_chunks.json", "r") as f:
        embedded_chunks = json.load(f)
    print(f"✅ Loaded {len(embedded_chunks)} embedded chunks")

    # Step 2: Connect to Qdrant
    print("\n=== STEP 2: CONNECTING TO QDRANT ===")
    client = create_qdrant_client()

    # Step 3: Create the collection (like creating a table)
    print("\n=== STEP 3: CREATING COLLECTION ===")
    create_collection(client)

    # Step 4: Upload all 1427 chunks into Qdrant
    print("\n=== STEP 4: UPLOADING CHUNKS ===")
    upload_chunks_to_qdrant(client, embedded_chunks)

    # Step 5: Print collection stats
    print("\n=== STEP 5: COLLECTION STATS ===")
    get_collection_stats(client)

    # Step 6: Test search — prove Qdrant works
    print("\n=== STEP 6: TEST SEARCH ===")
    
    # Load the embedding model to embed our test question
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Test question
    test_question = "What are Apple's main revenue sources?"
    print(f"Test question: '{test_question}'")
    
    # Convert question to vector
    question_vector = model.encode([test_question])[0].tolist()
    
    # Search Qdrant for top 3 most relevant chunks
    results = search_similar_chunks(client, question_vector, top_k=3)
    
    print(f"\nTop 3 results:")
    for i, result in enumerate(results):
        print(f"\n  Rank {i+1} | Score: {result['score']:.4f}")
        print(f"  Company: {result['company']} | Date: {result['date']}")
        print(f"  Text: {result['text'][:200]}...")