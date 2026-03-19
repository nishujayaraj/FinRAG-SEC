import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# We are using a free, open-source embedding model from HuggingFace
# "all-MiniLM-L6-v2" is a popular, lightweight model that:
# - Converts any text into a 384-dimensional vector
# - Is fast enough to run on CPU (no GPU needed)
# - Is specifically trained to capture MEANING, not just keywords
# - Is used in production by many companies
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model from HuggingFace.
    First time: downloads the model (~80MB) and caches it locally.
    After that: loads from cache instantly.
    """
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(chunks: List[Dict], 
                 model: SentenceTransformer,
                 batch_size: int = 64) -> List[Dict]:
    """
    Convert every chunk's text into a vector (embedding).
    
    What is a vector? It's just a list of numbers.
    e.g. "Apple makes iphones" → [0.23, -0.11, 0.87, 0.45, ...]
                                   (384 numbers total)
    
    Two chunks that are SIMILAR IN MEANING will have vectors
    that are mathematically close to each other.
    Two chunks that are DIFFERENT will have vectors far apart.
    
    This is how we do meaning-based search instead of keyword search!
    
    batch_size=64 means we process 64 chunks at a time
    instead of one by one — much faster!
    """
    
    print(f"\nEmbedding {len(chunks)} chunks in batches of {batch_size}...")
    
    # Extract just the text from each chunk
    # We only embed the text, not the metadata
    texts = [chunk["text"] for chunk in chunks]
    
    # This is the main embedding step
    # tqdm shows a nice progress bar so you can see it working
    # show_progress_bar=True shows progress inside the model too
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True  # return as numpy arrays (easier to work with)
    )
    
    # Now attach each embedding back to its chunk
    # So each chunk now has: text + metadata + its vector
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        embedded_chunk = {
            # Keep all original chunk data
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "company": chunk["company"],
            "date": chunk["date"],
            "source_file": chunk["source_file"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "word_count": chunk["word_count"],
            
            # Add the embedding vector as a list
            # (JSON can't store numpy arrays, so we convert to list)
            "embedding": embeddings[i].tolist()
        }
        embedded_chunks.append(embedded_chunk)
    
    print(f"✅ Embedded {len(embedded_chunks)} chunks successfully!")
    print(f"   Each chunk now has a {len(embeddings[0])}-dimensional vector")
    return embedded_chunks


def save_embeddings(embedded_chunks: List[Dict],
                    save_dir: str = "data/processed") -> str:
    """
    Save all embedded chunks to disk as a JSON file.
    
    Why save? Embedding is slow — we don't want to redo it every time.
    Next time we run the project, we just load this file directly.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(save_dir) / "embedded_chunks.json"
    
    print(f"\nSaving embeddings to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f)
    
    # Calculate file size to show user
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved! File size: {size_mb:.1f} MB")
    return str(output_path)


def verify_similarity(embedded_chunks: List[Dict], 
                      model: SentenceTransformer):
    """
    Quick sanity check — prove that meaning-based search works.
    
    We embed a test question and find which chunk is most similar.
    If the result is relevant to the question, our embeddings work!
    
    How similarity works:
    - We use "cosine similarity" — measures the angle between two vectors
    - Score of 1.0 = identical meaning
    - Score of 0.0 = completely unrelated
    - Score of -1.0 = opposite meaning
    """
    
    print("\n--- SIMILARITY VERIFICATION ---")
    test_question = "What are the biggest risks facing Apple?"
    print(f"Test question: '{test_question}'")
    
    # Embed the question using the same model
    # This converts the question into a 384-dimensional vector
    question_embedding = model.encode([test_question], convert_to_numpy=True)[0]
    
    # Convert all chunk embeddings from lists back to numpy arrays
    chunk_embeddings = np.array([c["embedding"] for c in embedded_chunks])
    
    # Calculate cosine similarity between question and ALL chunks at once
    # Cosine similarity = dot product / (magnitude of A * magnitude of B)
    # numpy makes this fast — computes 1427 similarities in milliseconds
    dot_products = np.dot(chunk_embeddings, question_embedding)
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    similarities = dot_products / norms
    
    # Get the top 3 most similar chunks
    top_3_indices = np.argsort(similarities)[::-1][:3]
    
    print(f"\nTop 3 most relevant chunks:")
    for rank, idx in enumerate(top_3_indices):
        chunk = embedded_chunks[idx]
        score = similarities[idx]
        print(f"\n  Rank {rank+1} | Score: {score:.4f}")
        print(f"  Company: {chunk['company']} | Date: {chunk['date']}")
        print(f"  Text preview: {chunk['text'][:200]}...")


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    # Step 1: Load the chunks we already created
    print("=== STEP 1: LOADING CHUNKS ===")
    chunks_path = "data/processed/all_chunks.json"
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks from {chunks_path}")

    # Step 2: Load the embedding model
    print("\n=== STEP 2: LOADING MODEL ===")
    model = load_embedding_model()

    # Step 3: Embed all chunks
    print("\n=== STEP 3: EMBEDDING CHUNKS ===")
    embedded_chunks = embed_chunks(chunks, model)

    # Step 4: Save embeddings to disk
    print("\n=== STEP 4: SAVING ===")
    save_embeddings(embedded_chunks)

    # Step 5: Verify it works with a test question
    print("\n=== STEP 5: VERIFICATION ===")
    verify_similarity(embedded_chunks, model)