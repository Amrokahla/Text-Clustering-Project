import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.download_data import download_data
from src.preprocessing import load_data, preprocess_data
from src.feature_extraction import get_sbert_embeddings, get_tfidf_features, get_word2vec_embeddings
from src.clustering import kmeans_clustering, lda_topic_modeling, save_cluster_results
from src.evaluation import evaluate_clustering
from src.visualization import plot_clusters, save_visualization

def main(args):
    if args.download:
        download_data()
    
    df = load_data(args.input_file)
    df = preprocess_data(df)
    
    embedding_methods = {
        'sbert': get_sbert_embeddings,
        'tfidf': get_tfidf_features,
        'word2vec': get_word2vec_embeddings}
    
    if args.embedding not in embedding_methods:
        raise ValueError(f"Unknown embedding type: {args.embedding}")

    if args.embedding == 'tfidf':
        feature_matrix, _ = embedding_methods[args.embedding](df["text"].tolist(), max_features=args.max_features)
        embeddings = feature_matrix.toarray()
    elif args.embedding in ['word2vec', 'sbert']:
        embeddings = embedding_methods[args.embedding](
            df["text"].tolist(), 
            model_name=args.embedding_model if args.embedding != 'sbert' else args.sbert_model)
    
    if args.method == 'kmeans':
        labels = kmeans_clustering(embeddings, n_clusters=args.n_clusters)
        title = f"K-Means Clustering ({args.n_clusters} clusters)"
    elif args.method == 'lda':
        labels = lda_topic_modeling(df["text"].tolist(), n_topics=args.n_topics)
        title = f"LDA Topic Modeling ({args.n_topics} topics)"
    else:
        raise ValueError(f"Unknown clustering method: {args.method}")
    
    silhouette = evaluate_clustering(embeddings, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    plot_clusters(embeddings, labels, title=title)
    
    save_cluster_results(df, labels)
    save_visualization(embeddings, labels, title, f"results/{args.method}_clusters.png")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamic Text Clustering")
    
    parser.add_argument('--input-file', default='data/people_wiki.csv', help='Path to input CSV file')
    parser.add_argument('--download', action='store_true', help='Download the default dataset')
    
    parser.add_argument('--embedding', choices=['sbert', 'tfidf', 'word2vec'], default='sbert', help='Embedding technique to use')
    parser.add_argument('--sbert-model', default='nli-roberta-base-v2', help='SBERT model to use for embeddings')
    parser.add_argument('--embedding-model', default='word2vec-google-news-300', help='Specific pre-trained embedding model')
    parser.add_argument('--max-features', type=int, default=5000, help='Maximum number of features for TF-IDF')
    
    parser.add_argument('--method', choices=['kmeans', 'lda'], default='kmeans', help='Clustering method to use')
    parser.add_argument('--n-clusters', type=int, default=3, help='Number of clusters for K-Means')
    parser.add_argument('--n-topics', type=int, default=5, help='Number of topics for LDA')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)