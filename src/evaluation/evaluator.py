import os
import json
import mlflow
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN", "")
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN", "")


def get_test_questions() -> List[Dict]:
    """
    Our evaluation dataset — 10 questions with known expected answers.
    Each question targets a specific company so we can filter
    Qdrant search to only that company's filings.
    """
    return [
        {
            "question": "What are Apple's main sources of revenue?",
            "ground_truth": "Apple generates revenue from iPhone sales, Mac computers, iPad, wearables, and services including App Store, Apple Music, and iCloud.",
            "company": "apple"
        },
        {
            "question": "What risks does Apple mention about its supply chain?",
            "ground_truth": "Apple relies on single source suppliers for some components and faces risks from supplier concentration and geopolitical issues.",
            "company": "apple"
        },
        {
            "question": "How does Apple describe its services segment?",
            "ground_truth": "Apple services segment includes App Store, Apple Music, iCloud, Apple Pay, Apple TV+ and generates recurring subscription revenue.",
            "company": "apple"
        },
        {
            "question": "What risks does Tesla mention related to electric vehicles?",
            "ground_truth": "Tesla mentions risks including range anxiety, charging infrastructure, competition from automakers, and battery technology limitations.",
            "company": "tesla"
        },
        {
            "question": "How does Tesla describe its manufacturing operations?",
            "ground_truth": "Tesla manufactures vehicles at Gigafactories and faces risks related to production ramp up and supply chain disruptions.",
            "company": "tesla"
        },
        {
            "question": "How does Microsoft describe its cloud computing business?",
            "ground_truth": "Microsoft describes Azure as its cloud platform offering computing, storage, and AI services with economies of scale advantages.",
            "company": "microsoft"
        },
        {
            "question": "What are Microsoft's main business segments?",
            "ground_truth": "Microsoft operates through Productivity and Business Processes, Intelligent Cloud, and More Personal Computing segments.",
            "company": "microsoft"
        },
        {
            "question": "What risks does Amazon mention about its AWS business?",
            "ground_truth": "Amazon mentions risks to AWS including competition from Microsoft Azure and Google Cloud, security breaches, and service outages.",
            "company": "amazon"
        },
        {
            "question": "How does Amazon describe its retail business model?",
            "ground_truth": "Amazon operates as both a direct retailer and marketplace platform where third party sellers can list products.",
            "company": "amazon"
        },
        {
            "question": "What does Google say about its advertising revenue?",
            "ground_truth": "Google generates most revenue from advertising through Google Search, YouTube ads, and Google Network advertising products.",
            "company": "google"
        }
    ]


def compute_context_relevancy(question: str, contexts: List[str]) -> float:
    """
    Measure how relevant the retrieved chunks are to the question.
    
    Simple but effective approach:
    - Extract keywords from the question (words longer than 4 chars)
    - Check what % of those keywords appear in the retrieved chunks
    - Score = keywords found / total keywords
    
    Example:
    Question: "What are Apple's revenue sources?"
    Keywords: ["Apple", "revenue", "sources"]
    If all 3 appear in chunks → score = 1.0
    If only 2 appear → score = 0.67
    """
    
    # Extract meaningful keywords from question
    # Filter out short words like "what", "are", "the", "its"
    question_words = set([
        word.lower().strip("?.,")
        for word in question.split()
        if len(word) > 4
    ])
    
    if not question_words:
        return 0.0
    
    # Combine all retrieved chunks into one big text
    combined_context = " ".join(contexts).lower()
    
    # Count how many question keywords appear in the chunks
    found = sum(1 for word in question_words if word in combined_context)
    
    # Score = fraction of keywords found
    score = found / len(question_words)
    return round(score, 4)


def compute_answer_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Measure how faithful the answer is to the retrieved chunks.
    
    Faithfulness = did the LLM answer using the chunks 
                   or did it make things up?
    
    Approach:
    - Extract keywords from the answer
    - Check what % of those keywords appear in the chunks
    - High score = answer closely follows the source material
    - Low score = answer contains things not in the chunks (hallucination)
    """
    
    # Extract meaningful words from the answer
    answer_words = set([
        word.lower().strip("?.,;:()")
        for word in answer.split()
        if len(word) > 4
    ])
    
    if not answer_words:
        return 0.0
    
    # Combine all chunks into one text
    combined_context = " ".join(contexts).lower()
    
    # Count how many answer words appear in the source chunks
    found = sum(1 for word in answer_words if word in combined_context)
    
    # Score = fraction of answer words that came from the chunks
    score = found / len(answer_words)
    return round(score, 4)


def compute_answer_relevancy(question: str,
                              answer: str,
                              embed_model: SentenceTransformer) -> float:
    """
    Measure how relevant the answer is to the question.
    
    We use semantic similarity here — the same technique
    we use for retrieval. If the question and answer are
    about the same topic, their vectors will be close.
    
    Score close to 1.0 = answer is very relevant to question
    Score close to 0.0 = answer is about something different
    """
    
    # Embed both the question and the answer
    embeddings = embed_model.encode([question, answer])
    question_vec = embeddings[0]
    answer_vec = embeddings[1]
    
    # Compute cosine similarity between question and answer vectors
    dot_product = np.dot(question_vec, answer_vec)
    norms = np.linalg.norm(question_vec) * np.linalg.norm(answer_vec)
    similarity = dot_product / norms
    
    return round(float(similarity), 4)


def run_rag_pipeline(question: str,
                     company: str,
                     embed_model: SentenceTransformer,
                     groq_client: Groq,
                     qdrant_client,
                     top_k: int = 3) -> Dict:
    """
    Run our complete RAG pipeline for one question.
    Combines embedding → Qdrant search → Groq generation.
    Returns question, answer, and retrieved contexts.
    """
    import sys
    sys.path.append(".")
    from src.retrieval.vector_store import search_similar_chunks
    from src.generation.generator import generate_answer

    # Convert question to vector
    question_vector = embed_model.encode([question])[0].tolist()

    # Search Qdrant — only search this company's filings
    retrieved_chunks = search_similar_chunks(
        qdrant_client,
        question_vector,
        top_k=top_k,
        company_filter=company
    )

    # Generate answer with Groq
    result = generate_answer(groq_client, question, retrieved_chunks)

    return {
        "question": question,
        "answer": result["answer"],
        "contexts": [chunk["text"] for chunk in retrieved_chunks],
        "retrieval_scores": [chunk["score"] for chunk in retrieved_chunks]
    }


def evaluate_rag_system(embed_model: SentenceTransformer,
                        groq_client: Groq,
                        qdrant_client) -> tuple:
    """
    Run full evaluation of our RAG system.
    
    For each of 10 test questions:
    1. Run the full RAG pipeline
    2. Compute 3 evaluation metrics
    3. Collect all results
    4. Return average scores + detailed dataframe
    """

    print("=== STARTING RAG EVALUATION ===\n")

    test_questions = get_test_questions()
    print(f"Evaluating {len(test_questions)} test questions...\n")

    # Store results for each question
    all_results = []

    for i, test_item in enumerate(test_questions):
        print(f"Question {i+1}/{len(test_questions)}: "
              f"{test_item['question'][:60]}...")

        # Run full RAG pipeline
        result = run_rag_pipeline(
            question=test_item["question"],
            company=test_item["company"],
            embed_model=embed_model,
            groq_client=groq_client,
            qdrant_client=qdrant_client
        )

        # Compute our 3 evaluation metrics
        context_rel = compute_context_relevancy(
            result["question"],
            result["contexts"]
        )

        faithfulness = compute_answer_faithfulness(
            result["answer"],
            result["contexts"]
        )

        answer_rel = compute_answer_relevancy(
            result["question"],
            result["answer"],
            embed_model
        )

        # Average retrieval score from Qdrant
        avg_retrieval_score = round(
            sum(result["retrieval_scores"]) / len(result["retrieval_scores"]), 4
        )

        # Overall score for this question = average of all metrics
        overall = round(
            (context_rel + faithfulness + answer_rel + avg_retrieval_score) / 4, 4
        )

        # Store this question's results
        all_results.append({
            "question": result["question"],
            "company": test_item["company"],
            "answer": result["answer"][:200] + "...",  # truncate for CSV
            "ground_truth": test_item["ground_truth"],
            "context_relevancy": context_rel,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_rel,
            "avg_retrieval_score": avg_retrieval_score,
            "overall_score": overall
        })

        print(f"  ✅ Context Rel: {context_rel:.3f} | "
              f"Faithfulness: {faithfulness:.3f} | "
              f"Answer Rel: {answer_rel:.3f}")

    # Build results dataframe
    results_df = pd.DataFrame(all_results)

    # Compute average scores across all questions
    scores = {
        "context_relevancy": round(
            float(results_df["context_relevancy"].mean()), 4),
        "faithfulness": round(
            float(results_df["faithfulness"].mean()), 4),
        "answer_relevancy": round(
            float(results_df["answer_relevancy"].mean()), 4),
        "avg_retrieval_score": round(
            float(results_df["avg_retrieval_score"].mean()), 4),
        "overall_score": round(
            float(results_df["overall_score"].mean()), 4)
    }

    return scores, results_df


def log_to_mlflow(scores: Dict,
                  results_df: pd.DataFrame,
                  run_name: str = "baseline"):
    """
    Log all results to MLflow for experiment tracking.
    
    MLflow records parameters + metrics + artifacts so you can:
    - Compare different experiments side by side
    - See which settings gave the best scores
    - Download detailed per-question results
    
    View in browser: run 'mlflow ui' then open http://localhost:5000
    """

    # Group all FinRAG runs under one experiment name
    mlflow.set_experiment("FinRAG_Evaluation")

    with mlflow.start_run(run_name=run_name):

        # Log the settings used for this experiment
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("llm_model", "llama-3.1-8b-instant")
        mlflow.log_param("chunk_size", 500)
        mlflow.log_param("chunk_overlap", 100)
        mlflow.log_param("top_k", 3)
        mlflow.log_param("num_test_questions", 10)
        mlflow.log_param("vector_db", "qdrant")
        mlflow.log_param("distance_metric", "cosine")

        # Log the evaluation scores
        mlflow.log_metric("context_relevancy", scores["context_relevancy"])
        mlflow.log_metric("faithfulness", scores["faithfulness"])
        mlflow.log_metric("answer_relevancy", scores["answer_relevancy"])
        mlflow.log_metric("avg_retrieval_score", scores["avg_retrieval_score"])
        mlflow.log_metric("overall_score", scores["overall_score"])

        # Save detailed per-question CSV as an artifact
        results_path = "data/processed/evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)

        print(f"✅ Logged to MLflow run: '{run_name}'")


def print_evaluation_summary(scores: Dict):
    """Print a clean summary table of all evaluation scores."""

    print("\n" + "="*60)
    print("       FINRAG EVALUATION SUMMARY")
    print("="*60)
    print(f"  Context Relevancy   : {scores['context_relevancy']:.4f} / 1.0")
    print(f"  Faithfulness        : {scores['faithfulness']:.4f} / 1.0")
    print(f"  Answer Relevancy    : {scores['answer_relevancy']:.4f} / 1.0")
    print(f"  Avg Retrieval Score : {scores['avg_retrieval_score']:.4f} / 1.0")
    print("-"*60)
    print(f"  Overall Score       : {scores['overall_score']:.4f} / 1.0")
    print("="*60)

    overall = scores["overall_score"]
    if overall >= 0.8:
        grade = "Excellent ✅"
    elif overall >= 0.6:
        grade = "Good 👍"
    elif overall >= 0.4:
        grade = "Needs Improvement ⚠️"
    else:
        grade = "Poor ❌"

    print(f"\n  Grade: {grade}")
    print("="*60)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.retrieval.vector_store import create_qdrant_client

    # Load all components
    print("=== LOADING COMPONENTS ===\n")

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding model loaded")

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("✅ Groq client created")

    qdrant_client = create_qdrant_client()
    print("✅ Qdrant client created")

    # Run evaluation
    scores, results_df = evaluate_rag_system(
        embed_model=embed_model,
        groq_client=groq_client,
        qdrant_client=qdrant_client
    )

    # Print summary
    print_evaluation_summary(scores)

    # Log to MLflow
    print("\n=== LOGGING TO MLFLOW ===")
    log_to_mlflow(scores, results_df, run_name="baseline_chunk500_top3")

    # Instructions to view MLflow UI
    print("\n✅ All done!")
    print("   To view results run: mlflow ui")
    print("   Then open: http://localhost:5000")