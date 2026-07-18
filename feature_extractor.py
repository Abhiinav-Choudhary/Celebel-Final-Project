"""
Extracts baseline features using a pre-trained ResNet50 model.
Generates fixed-size embeddings (2048-dim) from input images.

Usage:
    python feature_extractor.py
"""
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from PIL import Image
import project_config as config
import project_utils as utils


def build_baseline_extractor():
    """
    Sets up the ResNet50 feature extraction model.
    Removes the classification head and adds a GlobalAveragePooling2D layer.

    Returns:
        Keras Model that outputs 2048-dim embeddings.
    """
    print("Loading pretrained ResNet50...")
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=config.IMG_SHAPE
    )
    # Add global average pooling to get fixed-size embeddings
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    print(f"Feature extractor built. Output embedding dim: {model.output_shape[-1]}")
    return model


def load_image_resnet(image_path, target_size=config.IMG_SIZE):
    """
    Load and preprocess a single image for ResNet50.
    Uses Keras' built-in ResNet50 preprocessing (caffe-style).

    Args:
        image_path: Path to image file.
        target_size: (height, width) tuple.

    Returns:
        Preprocessed numpy array of shape (1, H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ResNet50-specific preprocessing
    return img_array


def extract_embeddings(model, image_dir, batch_size=config.BATCH_SIZE):
    """
    Iterates over image directories and extracts embeddings in batches.
    Maintains parallel arrays for IDs, categories, and paths.

    Args:
        model: Feature extraction model.
        image_dir: Root directory containing category subdirectories.
        batch_size: Batch size for prediction.

    Returns:
        embeddings: numpy array of shape (N, embedding_dim).
        image_ids: list of product IDs.
        categories: list of category labels.
        image_paths: list of full image paths.
    """
    print(f"\nExtracting embeddings from {image_dir}...")

    all_paths = []
    all_ids = []
    all_categories = []

    # Walk through category subdirectories
    for category in sorted(os.listdir(image_dir)):
        cat_dir = os.path.join(image_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for img_file in sorted(os.listdir(cat_dir)):
            if img_file.endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(cat_dir, img_file))
                all_ids.append(os.path.splitext(img_file)[0])
                all_categories.append(category)

    print(f"Found {len(all_paths)} images across {len(set(all_categories))} categories")

    # Extract in batches
    all_embeddings = []
    start_time = time.time()

    for i in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[i:i + batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                img = load_image_resnet(path)
                batch_images.append(img[0])
            except Exception as e:
                print(f"  Error loading {path}: {e}")
                # Use zero array as placeholder
                batch_images.append(np.zeros(config.IMG_SHAPE))

        batch_array = np.array(batch_images)
        batch_embeddings = model.predict(batch_array, verbose=0)
        all_embeddings.append(batch_embeddings)

        if (i // batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {i + len(batch_paths)}/{len(all_paths)} "
                  f"images ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    embeddings = np.vstack(all_embeddings)

    print(f"\nExtraction complete!")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Avg time per image: {elapsed / len(all_paths) * 1000:.1f}ms")

    return embeddings, all_ids, all_categories, all_paths


def extract_single_embedding(model, image_path):
    """
    Extract embedding for a single image.

    Args:
        model: Feature extraction model.
        image_path: Path to the image.

    Returns:
        1D numpy array of shape (embedding_dim,).
    """
    img = load_image_resnet(image_path)
    embedding = model.predict(img, verbose=0)
    return embedding.flatten()


def save_embeddings(embeddings, ids, categories, paths, prefix="baseline"):
    """
    Save embeddings and metadata to disk.

    Args:
        embeddings: numpy array of embeddings.
        ids: list of product IDs.
        categories: list of category labels.
        paths: list of image paths.
        prefix: Filename prefix (baseline/finetuned/siamese).
    """
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)

    emb_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_embeddings.npy")
    ids_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_ids.npy")
    cats_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_categories.npy")
    paths_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_paths.npy")

    np.save(emb_path, embeddings)
    np.save(ids_path, np.array(ids))
    np.save(cats_path, np.array(categories))
    np.save(paths_path, np.array(paths))

    print(f"Saved {prefix} embeddings to {config.EMBEDDINGS_DIR}/")
    print(f"  Embeddings: {emb_path} ({embeddings.shape})")
    print(f"  IDs: {ids_path} ({len(ids)} items)")


def load_embeddings(prefix="baseline"):
    """
    Load saved embeddings and metadata from disk.

    Args:
        prefix: Filename prefix.

    Returns:
        Tuple of (embeddings, ids, categories, paths).
    """
    emb_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_embeddings.npy")
    ids_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_ids.npy")
    cats_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_categories.npy")
    paths_path = os.path.join(config.EMBEDDINGS_DIR, f"{prefix}_paths.npy")

    embeddings = np.load(emb_path)
    ids = np.load(ids_path, allow_pickle=True)
    categories = np.load(cats_path, allow_pickle=True)
    paths = np.load(paths_path, allow_pickle=True)

    print(f"Loaded {prefix} embeddings: {embeddings.shape}")
    return embeddings, ids, categories, paths


def main():
    """Main baseline feature extraction pipeline."""
    print("=" * 60)
    print("BASELINE FEATURE EXTRACTION (ResNet50)")
    print("=" * 60)

    # Build model
    model = build_baseline_extractor()

    # Check that subset exists
    if not os.path.exists(config.TRAIN_DIR):
        print(f"ERROR: Training data not found at {config.TRAIN_DIR}")
        print("Please run data_preparation.py first.")
        return

    # Extract embeddings from BOTH train and val sets
    # (we want embeddings for all images in our database)
    all_embeddings = []
    all_ids = []
    all_categories = []
    all_paths = []

    for split_name, split_dir in [("train", config.TRAIN_DIR), ("val", config.VAL_DIR)]:
        if os.path.exists(split_dir):
            embeddings, ids, categories, paths = extract_embeddings(
                model, split_dir, config.BATCH_SIZE
            )
            all_embeddings.append(embeddings)
            all_ids.extend(ids)
            all_categories.extend(categories)
            all_paths.extend(paths)

    # Combine
    combined_embeddings = np.vstack(all_embeddings)
    print(f"\nCombined embeddings shape: {combined_embeddings.shape}")

    # Save
    save_embeddings(combined_embeddings, all_ids, all_categories, all_paths, "baseline")
    print("\n✓ Baseline feature extraction complete!")


if __name__ == "__main__":
    main()
