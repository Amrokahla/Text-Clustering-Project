import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

def kmeans_clustering(embeddings, n_clusters=3):
    """Apply K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

def lda_topic_modeling(text_features, n_topics=5):
    """Apply LDA topic modeling."""
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_distribution = lda.fit_transform(text_features)
    topic_labels = topic_distribution.argmax(axis=1)
    return topic_labels

def save_cluster_results(df, labels, filename="results/kmeans_clusters.csv"):
    df["Cluster"] = labels
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Cluster results saved to {filename}")