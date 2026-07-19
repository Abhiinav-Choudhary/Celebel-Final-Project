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
import base64
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

import project_config as config
from feature_extractor import build_baseline_extractor, load_image_resnet
from transfer_learning import build_finetuned_model, build_embedding_extractor
from siamese_network import build_embedding_network
from similarity_search import CosineSimilaritySearch, FAISSSearch
from project_utils import load_image_for_display

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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@400;500;700&display=swap');

    /* Main background */
    .stApp {
        background-color: #0f1115;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        letter-spacing: -0.5px;
    }
    
    /* Highlight accent */
    .highlight {
        background: linear-gradient(135deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cards for product results */
    .product-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .product-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 30px rgba(168, 85, 247, 0.15);
        border-color: rgba(168, 85, 247, 0.4);
    }
    
    /* Image inside card */
    .product-card img {
        width: 100%;
        border-radius: 12px;
        margin: 12px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    .product-card:hover img {
        transform: scale(1.05);
    }
    
    /* Score badges */
    .score-badge {
        background: linear-gradient(135deg, #a855f7, #3b82f6);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
        box-shadow: 0 2px 8px rgba(168, 85, 247, 0.4);
    }
    
    /* Category tag */
    .category-tag {
        background: rgba(15, 23, 42, 0.8);
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        align-self: flex-start;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Query container */
    .query-container {
        padding: 20px;
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .query-container img {
        border-radius: 12px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def get_image_base64(img_path):
    """Encode image to base64 for direct HTML embedding."""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


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
            models["siamese"] = build_embedding_network()
            models["siamese"].load_weights(config.SIAMESE_MODEL_PATH)
        else:
            models["siamese"] = None
    except Exception as e:
        print(f"Error loading Siamese model: {e}")
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
            # Display query image with premium container
            temp_query_path = os.path.join(config.DATA_DIR, "temp_query.jpg")
            img_b64 = get_image_base64(temp_query_path)
            st.markdown(f"""
            <div class="query-container">
                <h4 style="margin-top:0; color:#cbd5e1; font-family:'Outfit';">Target Item</h4>
                <img src="data:image/jpeg;base64,{img_b64}">
            </div>
            """, unsafe_allow_html=True)
            
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
                        try:
                            img_b64 = get_image_base64(r_path)
                            img_html = f'<img src="data:image/jpeg;base64,{img_b64}">'
                        except:
                            img_html = '<div style="color:#ef4444; padding:20px; text-align:center;">Image Missing</div>'
                            
                        st.markdown(f"""
                        <div class="product-card">
                            <div class="category-tag">{r_cat}</div>
                            {img_html}
                            <div style="width: 100%; text-align: center;">
                                <span style="font-size: 0.85em; color: #94a3b8; font-family: monospace;">ID: {r_id}</span><br>
                                <div class="score-badge">Match: {r_score:.3f}</div>
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
