import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_clusters(embeddings, labels, title="Cluster Visualization"):
    """Plot clusters in 2D space using PCA."""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="viridis")
    plt.title(title)
    plt.show()

def save_visualization(pca_embeddings, labels, title, filename):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], hue=labels, palette="viridis")
    plt.title(title)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"✅ Visualization saved to {filename}")
    plt.close()