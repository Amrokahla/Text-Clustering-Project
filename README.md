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

## Running the Project

### Run the Pipeline

Execute the `main.py` script:

```bash
python src/main.py
```

This will:
1. download [People wiki dataset](https://drive.google.com/file/d/1_oqcPdcdM2Rf0F5VasNgxqmZBP-c50Hd/view)
2. Load & preprocess text data
3. Extract SBERT embeddings
4. Cluster using K-Means
5. Apply LDA for topic modeling
6. Evaluate clustering performance
7. Visualize the results

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

## 📊 Results

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
