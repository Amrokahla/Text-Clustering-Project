import os
import pandas as pd
from sklearn.metrics import silhouette_score

def evaluate_clustering(embeddings, labels):
    """Calculate silhouette score for clustering."""
    return silhouette_score(embeddings, labels)

def save_topic_results(lda_topics, filename="results/lda_topics.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(lda_topics).to_csv(filename, index=False)
    print(f"✅ Topic distributions saved to {filename}")