import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.download_data import download_data
from src.preprocessing import load_data, preprocess_data
from src.feature_extraction import get_sbert_embeddings, get_lda_features
from src.clustering import kmeans_clustering, lda_topic_modeling, save_cluster_results
from src.evaluation import evaluate_clustering, save_topic_results
from src.visualization import plot_clusters, save_visualization


download_data()
file_path = "data\people_wiki.csv"
df = load_data(file_path)
print("data loaded")
df = preprocess_data(df)
print("data preprocessed")

embeddings = get_sbert_embeddings(df["text"].tolist())
text_features, _ = get_lda_features(df["text"].tolist())
print("embeddings generated")

kmeans_labels = kmeans_clustering(embeddings, n_clusters=3)
print("kmeans clustering completed")
lda_labels = lda_topic_modeling(text_features, n_topics=5)
print("LDA completed")

silhouette = evaluate_clustering(embeddings, kmeans_labels)
print(f"Silhouette Score (KMeans): {silhouette:.4f}")

plot_clusters(embeddings, kmeans_labels, title="K-Means Clusters")
plot_clusters(embeddings, lda_labels, title="LDA Topics")

save_cluster_results(df, kmeans_labels)
save_topic_results(lda_labels)
save_visualization(embeddings, kmeans_labels, "K-Means Clusters", "results/kmeans_clusters.png")



