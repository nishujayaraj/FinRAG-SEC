import os
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
# This gives us access to GROQ_API_KEY
load_dotenv()

os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN", "")
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN", "")

def create_groq_client() -> Groq:
    """
    Create a connection to Groq API.
    
    Groq is a cloud service that runs LLMs (like Llama 3) 
    at extremely fast speeds using special hardware called LPUs.
    Think of it as a super fast brain we can send questions to.
    
    We authenticate using the API key from our .env file.
    """
    
    # Read the Groq API key from .env file
    # If it's missing, raise a clear error message
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file!")
    
    # Create and return the Groq client using our API key
    # This is our connection to Groq's servers
    client = Groq(api_key=api_key)
    print("✅ Groq client created!")
    return client


def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build the prompt we send to the LLM.
    
    This is PROMPT ENGINEERING — carefully crafting the message
    so the LLM gives accurate, grounded, cited answers.
    
    The prompt has 3 parts:
    1. Instructions — tell LLM exactly how to behave
    2. Context     — the relevant chunks retrieved from Qdrant
    3. Question    — the user's actual question
    
    Why do we give the LLM the chunks?
    Because without them, the LLM answers from its training data
    which could be outdated or hallucinated. By giving it the
    actual SEC filing text, answers are accurate and traceable.
    This pattern is called RAG — Retrieval Augmented Generation.
    """
    
    # Build the context section by formatting each retrieved chunk
    # We label each chunk with its source company and date
    # so the LLM knows exactly where each piece of info came from
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            # Format: [Source 1: APPLE | 2024-11-01]
            # followed by the actual chunk text
            f"[Source {i+1}: {chunk['company'].upper()} | {chunk['date']}]\n"
            f"{chunk['text']}"
        )
    
    # Join all chunks into one big context block
    # separated by blank lines for readability
    context = "\n\n".join(context_parts)
    
    # Build the final prompt with 3 clear sections:
    # 1. Role + instructions at the top
    # 2. Context (the SEC filing excerpts)
    # 3. The actual question at the bottom
    prompt = f"""You are a financial analyst assistant specializing in SEC filings analysis.
You have been given relevant excerpts from official SEC 10-K annual reports.

Your job is to answer the user's question based ONLY on the provided context.

Rules:
- Only use information from the provided context below
- Always mention which company and year your information comes from
- If the context doesn't contain enough information, say so clearly
- Be precise and professional like a real financial analyst
- Keep your answer focused and well structured

CONTEXT FROM SEC FILINGS:
{context}

QUESTION: {question}

ANSWER:"""
    
    return prompt


def generate_answer(client: Groq,
                    question: str,
                    retrieved_chunks: List[Dict],
                    model: str = "llama-3.1-8b-instant") -> Dict:
    """
    Generate a final answer using Groq LLM.
    
    Flow:
    1. Build a prompt combining question + retrieved chunks
    2. Send prompt to Groq's Llama 3 model
    3. Get back a natural language answer
    4. Return the answer + source metadata
    
    About the model "llama3-8b-8192":
    - llama3    = Meta's open source LLM (like a free version of GPT-4)
    - 8b        = 8 billion parameters (the size/intelligence of the model)
    - 8192      = can read up to 8192 tokens (~6000 words) at once
    """
    
    # Step 1: Build the prompt using our chunks and question
    # This combines everything into one carefully crafted message
    prompt = build_prompt(question, retrieved_chunks)
    
    # Step 2: Send the prompt to Groq and get a response
    # This is a standard API call — we send messages, get a reply
    response = client.chat.completions.create(
        
        # Which LLM model to use
        model=model,
        
        # Messages follow a conversation format:
        # "system" = sets the overall behavior/personality of the LLM
        # "user"   = the actual message we're sending (our prompt)
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise financial analyst. "
                    "Answer questions based only on provided SEC filing excerpts. "
                    "Always cite your sources by company name and year."
                )
            },
            {
                # This is where our full prompt goes —
                # instructions + context chunks + question
                "role": "user",
                "content": prompt
            }
        ],
        
        # Maximum length of the response
        # 1 token ≈ 0.75 words, so 1024 tokens ≈ ~750 words
        # Enough for a detailed financial answer
        max_tokens=1024,
        
        # Temperature controls creativity vs accuracy:
        # 0.0 = completely deterministic, always same answer
        # 0.1 = very focused and factual (perfect for finance)
        # 1.0 = very creative and random (bad for facts)
        # We use 0.1 because we want accurate financial analysis
        temperature=0.1
    )
    
    # Step 3: Extract just the text answer from Groq's response object
    # response.choices[0] = first (and only) response
    # .message.content = the actual text the LLM generated
    answer = response.choices[0].message.content
    
    # Step 4: Package and return everything together:
    # - the original question
    # - the generated answer
    # - which sources (chunks) were used to generate it
    # - how many tokens were used (for cost monitoring)
    return {
        "question": question,
        "answer": answer,
        
        # Sources = metadata about each chunk used
        # Useful for showing citations to the user
        "sources": [
            {
                "company": chunk["company"],
                "date": chunk["date"],
                "chunk_id": chunk["chunk_id"],
                "score": chunk["score"]  # similarity score from Qdrant
            }
            for chunk in retrieved_chunks
        ],
        
        # Total tokens used in this request
        # input tokens (prompt) + output tokens (answer)
        "tokens_used": response.usage.total_tokens
    }


def pretty_print_result(result: Dict):
    """
    Print the final Q&A result in a clean, readable format.
    
    Shows:
    - The original question
    - The LLM generated answer
    - Which sources (company + date) were used
    - How many tokens were consumed
    """
    
    print("\n" + "="*60)
    print(f"QUESTION: {result['question']}")
    print("="*60)
    
    # Print the full answer from the LLM
    print(f"\nANSWER:\n{result['answer']}")
    
    print("\n" + "-"*60)
    print("SOURCES USED:")
    
    # Print each source with company, date and similarity score
    for i, source in enumerate(result["sources"]):
        print(f"  {i+1}. {source['company'].upper()} | "
              f"{source['date']} | "
              f"Similarity Score: {source['score']:.4f}")
    
    # Print token usage for cost awareness
    print(f"\nTokens used: {result['tokens_used']}")
    print("="*60)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from sentence_transformers import SentenceTransformer
    from src.retrieval.vector_store import (
        create_qdrant_client,
        search_similar_chunks
    )

    # Step 1: Connect to Groq API
    print("=== STEP 1: CONNECTING TO GROQ ===")
    groq_client = create_groq_client()

    # Step 2: Load the same embedding model we used before
    # IMPORTANT: must use the same model as during embedding
    # because the vector space must be identical
    # Using a different model would give wrong search results
    print("\n=== STEP 2: LOADING EMBEDDING MODEL ===")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding model loaded!")

    # Step 3: Connect to our local Qdrant database
    print("\n=== STEP 3: CONNECTING TO QDRANT ===")
    qdrant_client = create_qdrant_client()

    # Step 4: Test the full RAG pipeline with 3 real questions
    # One question per company to show cross-company capability
    print("\n=== STEP 4: TESTING Q&A ===")
    test_questions = [
        "What are Apple's main sources of revenue?",
        "What risks does Tesla mention related to electric vehicles?",
        "How does Microsoft describe its cloud computing business?"
    ]
    
    for question in test_questions:
        
        # Convert the question into a 384-number vector
        # using the same embedding model we used for the chunks
        question_vector = embed_model.encode([question])[0].tolist()
        
        # Search Qdrant for the top 3 chunks most similar
        # in meaning to the question
        # These chunks temporarily live in RAM as a Python list
        retrieved_chunks = search_similar_chunks(
            qdrant_client,
            question_vector,
            top_k=3  # get top 3 most relevant chunks
        )
        
        # Feed the question + retrieved chunks to Groq LLM
        # and get back a natural language answer with citations
        result = generate_answer(groq_client, question, retrieved_chunks)
        
        # Print the final result nicely
        pretty_print_result(result)