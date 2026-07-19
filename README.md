# 🛍️ Visual Product Recommendation Engine

An end-to-end image-based recommendation system that retrieves visually similar fashion products using Deep Learning. 

Unlike traditional keyword-based search, this engine understands visual similarities such as style, texture, and design. It achieves this by mapping images into dense mathematical vectors (embeddings) using Deep Convolutional Neural Networks, specifically heavily optimized and fine-tuned **ResNet50** and **Siamese Networks**.

---

## ✨ Key Features
- **Visual Search:** Upload any fashion image to instantly find the closest matching products in the catalog.
- **Deep Feature Embeddings:** Leverages a ResNet50 backbone to understand complex visual geometries.
- **Triplet Loss Siamese Network:** The core AI enhancement—a custom Siamese network fine-tuned using contrastive triplet loss (Anchor, Positive, Negative) to drastically improve semantic visual similarity.
- **Ultra-Fast Retrieval:** Uses **FAISS** (Facebook AI Similarity Search) and Cosine Similarity for lightning-fast vector matching.
- **Premium Interactive UI:** A beautifully designed Streamlit web application with modern dark-mode aesthetics, glassmorphism, and micro-animations.

---

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow & Keras
- **Vector Search & Math:** FAISS & NumPy
- **Web Interface:** Streamlit
- **Image Processing:** PIL (Pillow)
- **Data Manipulation:** Pandas

---

## 🚀 Setup Instructions

1. **Install Dependencies**
   Make sure you have Python 3.10+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Dataset**
   The project uses the *Fashion Product Images (Small)* dataset from Kaggle.
   Run the data preparation script to automatically download, filter, and organize a subset of 8 categories (approx 2,000 images):
   ```bash
   python data_preparation.py
   ```

---

## 🧠 Training Pipeline

To train the models and extract the necessary deep learning embeddings, you can run the master training script. This script automatically handles Transfer Learning fine-tuning, Siamese Network triplet loss training, and database embedding extraction:

```bash
python train_all.py
```
*(Note: If the extraction stalls at the very end, simply run `python finish_extraction.py` to finalize the embeddings.)*

---

## 🌐 Running the Web App

Launch the interactive UI to visually test the models:
```bash
streamlit run app.py
```
* **How to use:** Open the provided localhost link in your browser. Select a model (Baseline, Fine-tuned, or Siamese) from the sidebar, upload an image from the `data/subset/val/` folder, and instantly view the visually similar recommendations!

---

## 📊 Evaluation & Benchmarks

To calculate hard metrics (Precision@K, Recall@K, and Inference Time) across all three models and generate visual comparison plots:
```bash
python evaluate.py
```
This script will output a tabular summary in your terminal and generate comparison bar charts and visual grids in the `plots/` directory.

---

## 📁 Project Structure

```text
visual-product-recommendation/
├── .streamlit/               # Streamlit theme configuration (Dark Mode)
├── data/                     # Raw Kaggle images and processed subset categories
├── embeddings/               # Extracted .npy vector databases and metadata
├── models/                   # Saved .keras deep learning models
├── plots/                    # Evaluation charts and visual comparison grids
├── app.py                    # The Streamlit web interface
├── data_preparation.py       # Kaggle dataset downloader and organizer
├── evaluate.py               # Benchmark metrics script (Precision/Recall)
├── feature_extractor.py      # Baseline ResNet50 extraction logic
├── project_config.py         # Global variables, paths, and hyperparameters
├── project_utils.py          # Helper functions for UI and paths
├── siamese_network.py        # Siamese Network triplet training logic
├── similarity_search.py      # FAISS and Cosine Similarity classes
├── train_all.py              # Master orchestration script for training
└── transfer_learning.py      # Fine-tuning classification script
```
