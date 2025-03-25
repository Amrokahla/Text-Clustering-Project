# Text Clustering & Topic Modeling

This project applies **K-Means clustering** and **LDA topic modeling** to group similar text data, using **Sentence-BERT embeddings** for high-quality feature extraction.

## Features

- **Preprocessing**: Cleans text (removes URLs, punctuation, stopwords, applies stemming)
- **Embeddings**: Uses **SBERT** for sentence representations
- **Clustering**: K-Means for document grouping
- **Topic Modeling**: LDA to extract dominant topics
- **Evaluation**: Silhouette Score for clustering quality
- **Visualization**: PCA-based cluster plotting

## Installation & Setup

It's a better practice to creat a __virtual enviroment__ before using the project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_PROJECT.git
cd YOUR_PROJECT
```

### 2. Install Dependencies

Make sure you have Python 3.10+ installed. Then, run:

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources (One-time setup)

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

## Dynamic Usage

The `main.py` script supports multiple embedding and clustering techniques:

### Data Download
The data being used is [People Wiki Data](https://drive.google.com/file/d/1_oqcPdcdM2Rf0F5VasNgxqmZBP-c50Hd/view?usp=sharing)

### Embedding Techniques
- `sbert`: Sentence-BERT embeddings (default)
- `tfidf`: TF-IDF Vectorization
- `lda`: Latent Dirichlet Allocation features

### Clustering Methods
- `kmeans`: K-Means Clustering
- `lda`: Latent Dirichlet Allocation Topic Modeling

### Example Commands

1. Default (SBERT embeddings with K-Means):
```bash
python src/main.py
```

2. TF-IDF with K-Means and 5 clusters:
```bash
python src/main.py --embedding tfidf --method kmeans --n-clusters 5
```

3. SBERT with different model and LDA:
```bash
python src/main.py --sbert-model all-MiniLM-L6-v2 --method lda --n-topics 7
```

### Customization Options
- `--input-file`: Specify a custom input CSV
- `--embedding`: Choose embedding technique
- `--method`: Select clustering method
- `--n-clusters`: Set number of K-Means clusters
- `--n-topics`: Set number of LDA topics
- `--max-features`: Limit feature dimensions

## Project Structure

```
├── data/                  # Datasets (raw and preprocessed)
├── src/
│   ├── preprocessing.py    # Data cleaning & preprocessing
│   ├── feature_extraction.py  # Vectorization & embedding
│   ├── clustering.py       # Clustering models
│   ├── evaluation.py       # Clustering metrics
│   ├── visualization.py    # Result visualization
│   ├── main.py             # Main script for execution
├── notebooks/              # Jupyter notebooks for EDA
├── results/                # Cluster results & visuals
├── requirements.txt        # Dependencies
├── README.md               # Documentation
```

## Results

After running, you will see:
-  Cluster Assignments
-  Topic Distributions
-  **Visualizations of K-Means & LDA Clusters**

## Notes & Next Steps

- Try different **embedding models** (`all-MiniLM-L6-v2`, `mpnet-base-v2`)
- Experiment with **more clusters** (`n_clusters`)
- Fine-tune **LDA hyperparameters** for better topic extraction

## Contact & Contribution

Feel free to **contribute, suggest improvements, or report issues**. 
