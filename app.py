"""
Streamlit Web Interface for Visual Product Recommendation Engine.

Allows users to upload an image and retrieve visually similar products
using different embedding models (Baseline, Fine-tuned, Siamese).

Usage:
    streamlit run app.py
"""
import os
import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

import config
from feature_extractor import build_baseline_extractor, load_image_resnet
from transfer_learning import build_finetuned_model, build_embedding_extractor
from siamese_network import build_siamese_model
from similarity_search import CosineSimilaritySearch, FAISSSearch
from utils import load_image_for_display

# --- Page Configuration ---
st.set_page_config(
    page_title="Visual Product Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }
    
    /* Highlight accent */
    .highlight {
        color: #bb86fc;
        font-weight: bold;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1e1e1e !important;
    }
    
    /* Cards for product results */
    .product-card {
        background: #1e1e1e;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(187, 134, 252, 0.2);
        border-color: #bb86fc;
    }
    
    /* Score badges */
    .score-badge {
        background: linear-gradient(135deg, #bb86fc, #3700b3);
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }
    
    /* Category tag */
    .category-tag {
        background: #333;
        color: #e0e0e0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        display: inline-block;
    }
    
    /* Divider */
    hr {
        border-color: #333;
    }
</style>
""", unsafe_allow_html=True)


# --- Global State & Caching ---

@st.cache_resource(show_spinner=False)
def load_models():
    """Load deep learning models and return a dictionary."""
    models = {}
    
    # 1. Baseline Model
    try:
        models["baseline"] = build_baseline_extractor()
    except Exception as e:
        st.error(f"Failed to load baseline model: {e}")
        
    # 2. Finetuned Model
    try:
        if os.path.exists(config.FINETUNED_MODEL_PATH):
            full_model = tf.keras.models.load_model(config.FINETUNED_MODEL_PATH)
            models["finetuned"] = build_embedding_extractor(full_model)
        else:
            models["finetuned"] = None
    except Exception as e:
        models["finetuned"] = None
        
    # 3. Siamese Model
    try:
        if os.path.exists(config.SIAMESE_MODEL_PATH):
            models["siamese"] = tf.keras.models.load_model(
                config.SIAMESE_MODEL_PATH,
                compile=False
            )
        else:
            models["siamese"] = None
    except Exception as e:
        models["siamese"] = None
        
    return models


@st.cache_resource(show_spinner=False)
def load_database_searchers():
    """Load precomputed embeddings and initialize search engines."""
    searchers = {}
    db_stats = {}
    
    for prefix in ["baseline", "finetuned", "siamese"]:
        emb_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_embeddings.npy")
        ids_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_ids.npy")
        cats_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_categories.npy")
        paths_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_paths.npy")
        
        if all(os.path.exists(p) for p in [emb_path, ids_path, cats_path, paths_path]):
            try:
                embeddings = np.load(emb_path)
                ids = np.load(ids_path, allow_pickle=True)
                categories = np.load(cats_path, allow_pickle=True)
                paths = np.load(paths_path, allow_pickle=True)
                
                # Initialize FAISS (or Cosine fallback)
                searchers[prefix] = FAISSSearch(embeddings, ids, categories, paths)
                db_stats[prefix] = {
                    "count": len(ids),
                    "dim": embeddings.shape[1],
                    "categories": len(set(categories))
                }
            except Exception as e:
                st.warning(f"Error loading {prefix} embeddings: {e}")
                
    return searchers, db_stats


def process_query_image(uploaded_file, model, model_name):
    """Process uploaded image and extract embeddings."""
    if uploaded_file is None or model is None:
        return None
        
    # Save temp file
    temp_path = os.path.join(config.DATA_DIR, "temp_query.jpg")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Preprocess
    img_array = load_image_resnet(temp_path)
    
    # Extract
    start_time = time.time()
    embedding = model.predict(img_array, verbose=0)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    return embedding.flatten(), inference_time


# --- UI Layout ---

def main():
    # Load assets
    with st.spinner("Loading AI Models & Vector Database..."):
        models = load_models()
        searchers, db_stats = load_database_searchers()

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Model Selection
        st.subheader("Model Architecture")
        model_options = []
        if "baseline" in searchers: model_options.append("Baseline (ResNet50)")
        if "finetuned" in searchers: model_options.append("Fine-tuned (ResNet50)")
        if "siamese" in searchers: model_options.append("Siamese (Triplet Loss)")
        
        if not model_options:
            st.error("No models found! Please run training scripts first.")
            return
            
        selected_model_str = st.selectbox("Select Model", model_options)
        
        # Map string to prefix
        if "Baseline" in selected_model_str: prefix = "baseline"
        elif "Fine-tuned" in selected_model_str: prefix = "finetuned"
        elif "Siamese" in selected_model_str: prefix = "siamese"
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        top_k = st.slider("Number of recommendations (K)", 1, 20, 5)
        
        # Stats
        if prefix in db_stats:
            st.subheader("Database Stats")
            st.metric("Total Items", f"{db_stats[prefix]['count']:,}")
            st.metric("Vector Dimension", db_stats[prefix]['dim'])
            st.metric("Categories", db_stats[prefix]['categories'])

    # Main area
    st.markdown("<h1>🛍️ Visual <span class='highlight'>Product Recommendation</span></h1>", unsafe_allow_html=True)
    st.markdown("Upload an image of a fashion item, and our AI will find visually similar products in the catalog.")
    
    # File upload
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Query Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display query image
            image = Image.open(uploaded_file)
            st.image(image, caption="Query Image", use_container_width=True)
            
    with col2:
        st.subheader("2. Similar Products")
        
        if uploaded_file is not None:
            if models.get(prefix) is None:
                st.error(f"Model '{prefix}' is not loaded properly.")
                return
                
            # Process query
            with st.spinner("Extracting features..."):
                query_emb, inf_time = process_query_image(uploaded_file, models[prefix], prefix)
                
            if query_emb is not None:
                with st.spinner("Searching vector database..."):
                    search_start = time.time()
                    results = searchers[prefix].search(query_emb, top_k=top_k, exclude_self=False)
                    search_time = (time.time() - search_start) * 1000
                    
                # Display metrics
                st.markdown(f"**Metrics:** Feature Extraction: `{inf_time:.1f}ms` | Vector Search: `{search_time:.1f}ms`")
                st.markdown("---")
                
                # Display results in grid
                cols = st.columns(3)
                for i in range(len(results["ids"])):
                    col_idx = i % 3
                    
                    r_id = results["ids"][i]
                    r_cat = results["categories"][i]
                    r_path = results["paths"][i]
                    r_score = results["scores"][i]
                    
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div class="product-card">
                            <div class="category-tag">{r_cat}</div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            # Use fixed height for consistency
                            img = Image.open(r_path)
                            st.image(img, use_container_width=True)
                        except:
                            st.error("Image missing")
                            
                        st.markdown(f"""
                            <div style="margin-top: 10px;">
                                <span style="font-size: 0.9em; color: #aaa;">ID: {r_id}</span><br>
                                <div class="score-badge">Similarity: {r_score:.3f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("👈 Upload an image to see recommendations!")
            
            # Show empty state
            st.markdown("""
            <div style="text-align: center; padding: 50px; background: #1e1e1e; border-radius: 12px; border: 1px dashed #444; color: #888;">
                <h3 style="color: #666;">Waiting for input</h3>
                <p>The recommendation engine uses Deep Learning (ResNet50 & Siamese Networks) to match visual features rather than text keywords.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
