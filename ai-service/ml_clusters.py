from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import numpy as np

# =====================================================
# Embedding Model
# =====================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================================
# FLAN-T5 Cluster Naming Model
# =====================================================

cluster_namer = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# =====================================================
# Generate Cluster Name
# =====================================================

def generate_cluster_name(items):

    reviews_text = "\n".join([
        r.get("text", "")[:150]
        for r in items[:5]
    ])

    prompt = f"""
Analyze these customer reviews and generate:

Name: short business-friendly cluster name (max 5 words)

Summary: one sentence

Reviews:
{reviews_text}
"""

    try:

        response = cluster_namer(
            prompt,
            max_new_tokens=50
        )

        text = response[0]["generated_text"].strip()

        lines = [
            l.strip()
            for l in text.split("\n")
            if l.strip()
        ]

        name = "Customer Segment"
        summary = ""

        for line in lines:

            lower = line.lower()

            if lower.startswith("name:"):
                name = line.split(":", 1)[1].strip()

            elif lower.startswith("summary:"):
                summary = line.split(":", 1)[1].strip()

        # fallback if FLAN returns plain text
        if name == "Customer Segment" and lines:
            name = lines[0][:60]

        return name, summary

    except Exception as e:

        print("Cluster naming error:", e)

        return (
            "Customer Segment",
            ""
        )

# =====================================================
# Main Clustering Function
# =====================================================

def cluster_issues(reviews, k=5):

    if not reviews:
        return []

    texts = [
        r.get("text", "")
        for r in reviews
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    k = min(k, len(texts))

    if k <= 1:
        return [{
            "clusterId": 0,
            "clusterName": "All Reviews",
            "clusterSummary": "",
            "size": len(reviews),
            "sampleText": reviews[0].get("text", "")
        }]

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(embeddings)

    clusters = {}

    for i, label in enumerate(labels):

        label = int(label)

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(
            reviews[i]
        )

    result = []

    for cid, items in clusters.items():

        cluster_name, cluster_summary = (
            generate_cluster_name(items)
        )

        result.append({
            "clusterId": cid,
            "clusterName": cluster_name,
            "clusterSummary": cluster_summary,
            "size": len(items),
            "sampleText": items[0].get("text", ""),
            "reviews": items[:3]
        })

    result.sort(
        key=lambda x: x["size"],
        reverse=True
    )

    return result