import json
import torch
from sentence_transformers import SentenceTransformer, util

# ===========================
# 1. Load the JSON dataset
# ===========================
reviews = []
with open("Software_5.json", "r") as f:
    for i, line in enumerate(f):
        if i >= 1000:  # limit to first 10k reviews for speed
            break
        data = json.loads(line)
        text = data.get("reviewText", "").strip()
        data["combined"] = text  # only use reviewText for embedding
        reviews.append(data)

print(f"✅ Loaded {len(reviews)} reviews.")
print("Example keys:", list(reviews[0].keys()))

# ===========================
# 2. Load SentenceTransformer
# ===========================
model_name = "all-MiniLM-L6-v2"  # good tradeoff between speed and quality
model = SentenceTransformer(model_name)

# ===========================
# 3. Encode all reviews
# ===========================
corpus = [r["combined"] for r in reviews]
print("🔍 Encoding all reviews...")
corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

# ===========================
# 4. Define semantic search function
# ===========================
def search_reviews(query, top_k=5):
    """Return top_k most semantically similar reviews to the query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        review = reviews[hit["corpus_id"]]
        results.append({
            "score": float(hit["score"]),
            "reviewText": review.get("reviewText", ""),
            "overall": review.get("overall", None),
            "reviewerName": review.get("reviewerName", ""),
            "asin": review.get("asin", "")
        })
    return results

# ===========================
# 5. Example queries
# ===========================
if __name__ == "__main__":
    queries = [
        "Is Microsoft Office 365 any good?",
        "Tell me about the reviews for Photoshop",
        "Does Paint Shop work well?"
    ]

    for q in queries:
        print(f"\n🔎 Query: {q}")
        results = search_reviews(q, top_k=10)
        for i, r in enumerate(results):
            print(f"  {i+1}. [Score: {r['score']:.3f}] (Rating: {r['overall']}) by {r['reviewerName']}")
            print(f"     {r['reviewText'][:200]}...\n")
