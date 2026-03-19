import json
from pathlib import Path
from typing import List, Dict


def chunk_text(text: str, 
               chunk_size: int = 500, 
               overlap: int = 100) -> List[str]:
    """
    Break a long text into smaller overlapping chunks.
    
    Why overlapping? Because important sentences can fall at the 
    boundary between two chunks. Overlap ensures nothing is missed.
    
    Example with chunk_size=500, overlap=100:
    chunk 1: words 0-500
    chunk 2: words 400-900   (starts 100 words before chunk 1 ended)
    chunk 3: words 800-1300  (starts 100 words before chunk 2 ended)
    """
    
    # Split the entire text into individual words
    words = text.split()
    
    # If the text is shorter than one chunk, just return it as is
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0  # starting word index for each chunk
    
    while start < len(words):
        # Calculate end index for this chunk
        end = start + chunk_size
        
        # Grab the words for this chunk and join them back into a string
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move start forward by (chunk_size - overlap)
        # This creates the overlap with the next chunk
        start += chunk_size - overlap
    
    return chunks


def chunk_document(doc: dict,
                   chunk_size: int = 500,
                   overlap: int = 100) -> List[Dict]:
    """
    Take a single parsed document and split it into chunks.
    
    Each chunk keeps metadata (company, date) attached to it
    so we always know WHERE a chunk came from.
    This is critical for RAG — when we find a relevant chunk,
    we need to tell the user which company and year it's from.
    """
    
    # Get the chunks of raw text
    text_chunks = chunk_text(doc["text"], chunk_size, overlap)
    
    chunks = []
    for i, chunk_text_content in enumerate(text_chunks):
        chunks.append({
            # The actual text content of this chunk
            "text": chunk_text_content,
            
            # Metadata — always know where this chunk came from
            "company": doc["company"],
            "date": doc["date"],
            "source_file": doc["filepath"],
            
            # Chunk position — useful for debugging and context
            "chunk_index": i,
            "total_chunks": len(text_chunks),
            
            # Unique ID for this chunk — company + date + chunk number
            # e.g. "apple_2024-11-01_chunk_42"
            "chunk_id": f"{doc['company']}_{doc['date']}_chunk_{i}",
            
            # Word count of just this chunk
            "word_count": len(chunk_text_content.split())
        })
    
    return chunks


def chunk_all_documents(docs: List[Dict],
                        chunk_size: int = 500,
                        overlap: int = 100,
                        save_dir: str = "data/processed") -> List[Dict]:
    """
    Chunk ALL parsed documents and optionally save to disk.
    
    This is the main function that processes all 15 filings
    and returns one big flat list of all chunks combined.
    """
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    all_chunks = []
    
    for doc in docs:
        print(f"Chunking {doc['company']} | {doc['date']}...")
        
        # Chunk this single document
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        
        print(f"  ✅ {len(chunks)} chunks created "
              f"(avg {doc['word_count'] // max(len(chunks),1)} words/chunk)")
    
    print(f"\n✅ Total chunks across all documents: {len(all_chunks)}")
    
    # Save all chunks to a JSON file so we don't have to rechunk every time
    output_path = Path(save_dir) / "all_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to {output_path}")
    return all_chunks


if __name__ == "__main__":
    # Import the parser we already built
    import sys
    sys.path.append(".")
    from src.ingestion.parser import parse_all_filings
    
    # Step 1: Parse all 15 raw HTML filings into clean text
    print("=== STEP 1: PARSING ===")
    docs = parse_all_filings()
    
    # Step 2: Chunk all documents
    print("\n=== STEP 2: CHUNKING ===")
    chunks = chunk_all_documents(docs, chunk_size=500, overlap=100)
    
    # Step 3: Show a sample chunk so we can verify it looks right
    print("\n--- SAMPLE CHUNK ---")
    sample = chunks[100]  # pick chunk number 100 as example
    print(f"ID      : {sample['chunk_id']}")
    print(f"Company : {sample['company']}")
    print(f"Date    : {sample['date']}")
    print(f"Words   : {sample['word_count']}")
    print(f"Text    :\n{sample['text'][:500]}")