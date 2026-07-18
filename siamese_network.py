"""
Siamese Network Training script.
Trains an embedding model using contrastive learning on image triplets
(anchor, positive, negative) to optimize for visual similarity.

Architecture: Shared ResNet50 backbone → Dense embedding (128-dim)
Loss: Triplet loss (anchor, positive, negative)

Usage:
    python siamese_network.py
"""
import os
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Lambda, GlobalAveragePooling2D,
    BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from PIL import Image
import config
from feature_extractor import extract_embeddings, save_embeddings


# ─── Triplet Generation ─────────────────────────────────────────────────────

def load_dataset_by_category(data_dir):
    """
    Load all images organized by category.

    Args:
        data_dir: Root directory with category subdirectories.

    Returns:
        Dict mapping category_name → list of image paths.
    """
    category_images = {}
    for category in sorted(os.listdir(data_dir)):
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        images = [
            os.path.join(cat_dir, f)
            for f in sorted(os.listdir(cat_dir))
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        if images:
            category_images[category] = images
    return category_images


def generate_triplets(category_images, n_triplets_per_anchor=config.TRIPLETS_PER_ANCHOR):
    """
    Generates training triplets (Anchor, Positive, Negative).
    Positive images are sampled from the same category as the anchor.
    Negative images are sampled from a different category.
    """
    categories = list(category_images.keys())
    triplets = []

    for cat in categories:
        cat_images = category_images[cat]
        other_categories = [c for c in categories if c != cat]

        for anchor_path in cat_images:
            for _ in range(n_triplets_per_anchor):
                # Positive: same category, different image
                positive_candidates = [p for p in cat_images if p != anchor_path]
                if not positive_candidates:
                    continue
                positive_path = random.choice(positive_candidates)

                # Negative: different category
                neg_cat = random.choice(other_categories)
                negative_path = random.choice(category_images[neg_cat])

                triplets.append((anchor_path, positive_path, negative_path))

    random.shuffle(triplets)
    return triplets


def load_and_preprocess(image_path, target_size=config.IMG_SIZE):
    """Load and preprocess image for ResNet50."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return img_array


class TripletDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence generator for triplet batches.
    Yields [anchor_batch, positive_batch, negative_batch], dummy_labels.
    """

    def __init__(self, triplets, batch_size=config.BATCH_SIZE,
                 target_size=config.IMG_SIZE):
        self.triplets = triplets
        self.batch_size = batch_size
        self.target_size = target_size
        self.indices = np.arange(len(triplets))

    def __len__(self):
        return len(self.triplets) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_triplets = [self.triplets[i] for i in batch_indices]

        anchors = []
        positives = []
        negatives = []

        for anchor_path, pos_path, neg_path in batch_triplets:
            try:
                anchors.append(load_and_preprocess(anchor_path, self.target_size))
                positives.append(load_and_preprocess(pos_path, self.target_size))
                negatives.append(load_and_preprocess(neg_path, self.target_size))
            except Exception as e:
                # Use zero arrays as fallback
                anchors.append(np.zeros((*self.target_size, 3)))
                positives.append(np.zeros((*self.target_size, 3)))
                negatives.append(np.zeros((*self.target_size, 3)))

        return (
            [np.array(anchors), np.array(positives), np.array(negatives)],
            np.zeros(len(anchors))  # Dummy labels
        )

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# ─── Model Architecture ─────────────────────────────────────────────────────

def build_embedding_network():
    """
    Build the shared embedding network (backbone).

    ResNet50 (pretrained) → GlobalAvgPool → Dense(512) → Dense(128)

    Returns:
        Keras Model that maps images to normalized embeddings.
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=config.IMG_SHAPE
    )

    # Freeze most layers, unfreeze last few
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-15:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(config.EMBEDDING_DIM, activation=None, name="embedding")(x)

    # L2 normalize the embeddings
    x = Lambda(lambda t: K.l2_normalize(t, axis=1), name="l2_norm")(x)

    model = Model(inputs=base_model.input, outputs=x, name="embedding_network")
    return model


def triplet_loss(y_true, y_pred, margin=config.SIAMESE_MARGIN):
    """
    Triplet loss function.

    L = max(0, d(a,p) - d(a,n) + margin)

    The y_pred contains stacked [anchor, positive, negative] embeddings.
    """
    embedding_dim = config.EMBEDDING_DIM

    anchor = y_pred[:, 0:embedding_dim]
    positive = y_pred[:, embedding_dim:2 * embedding_dim]
    negative = y_pred[:, 2 * embedding_dim:3 * embedding_dim]

    # Squared Euclidean distances
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    loss = K.maximum(pos_dist - neg_dist + margin, 0.0)
    return K.mean(loss)


def build_siamese_model(base_model=None):
    """
    Constructs the Siamese network architecture.
    Wraps the shared embedding network to process triplets simultaneously
    and compute the triplet loss.
    """
    print("Building Siamese network...")

    embedding_network = build_embedding_network()

    # Three inputs
    anchor_input = Input(shape=config.IMG_SHAPE, name="anchor_input")
    positive_input = Input(shape=config.IMG_SHAPE, name="positive_input")
    negative_input = Input(shape=config.IMG_SHAPE, name="negative_input")

    # Shared embeddings
    anchor_embedding = embedding_network(anchor_input)
    positive_embedding = embedding_network(positive_input)
    negative_embedding = embedding_network(negative_input)

    # Concatenate for loss computation
    merged = tf.keras.layers.Concatenate()(
        [anchor_embedding, positive_embedding, negative_embedding]
    )

    siamese_model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=merged,
        name="siamese_model"
    )

    siamese_model.compile(
        optimizer=Adam(learning_rate=config.SIAMESE_LEARNING_RATE),
        loss=triplet_loss
    )

    trainable_params = sum(
        K.count_params(w) for w in siamese_model.trainable_weights
    )
    print(f"  Embedding dim: {config.EMBEDDING_DIM}")
    print(f"  Trainable parameters: {int(trainable_params):,}")

    return siamese_model, embedding_network


# ─── Training ────────────────────────────────────────────────────────────────

def train_siamese(siamese_model, train_gen, val_gen=None):
    """
    Train the Siamese network.

    Args:
        siamese_model: Compiled Siamese model.
        train_gen: TripletDataGenerator for training.
        val_gen: Optional TripletDataGenerator for validation.

    Returns:
        Training history.
    """
    print(f"\nTraining Siamese network for {config.SIAMESE_EPOCHS} epochs...")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    start_time = time.time()
    history = siamese_model.fit(
        train_gen,
        epochs=config.SIAMESE_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time

    print(f"\nSiamese training complete in {elapsed:.1f}s")
    print(f"  Final loss: {history.history['loss'][-1]:.4f}")

    return history


def extract_siamese_embeddings(embedding_network):
    """
    Extract embeddings using the trained Siamese embedding network.

    Args:
        embedding_network: Trained embedding model.
    """
    print("\nExtracting Siamese embeddings...")

    all_embeddings = []
    all_ids = []
    all_categories = []
    all_paths = []

    for split_name, split_dir in [("train", config.TRAIN_DIR),
                                   ("val", config.VAL_DIR)]:
        if os.path.exists(split_dir):
            embeddings, ids, categories, paths = extract_embeddings(
                embedding_network, split_dir, config.BATCH_SIZE
            )
            all_embeddings.append(embeddings)
            all_ids.extend(ids)
            all_categories.extend(categories)
            all_paths.extend(paths)

    combined = np.vstack(all_embeddings)
    save_embeddings(combined, all_ids, all_categories, all_paths, "siamese")


def plot_siamese_loss(history, save_path=None):
    """Plot Siamese training loss curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Training Loss",
            linewidth=2, color="#e74c3c")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="Validation Loss",
                linewidth=2, color="#3498db")
    ax.set_title("Siamese Network — Triplet Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Triplet Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Siamese loss plot saved to {save_path}")
    plt.show()


def main():
    """Main Siamese network training pipeline."""
    print("=" * 60)
    print("SIAMESE NETWORK TRAINING")
    print("=" * 60)

    # Check that subset exists
    if not os.path.exists(config.TRAIN_DIR):
        print(f"ERROR: Training data not found at {config.TRAIN_DIR}")
        print("Please run data_preparation.py first.")
        return

    # Step 1: Load dataset by category
    print("\nLoading dataset by category...")
    train_images = load_dataset_by_category(config.TRAIN_DIR)
    val_images = load_dataset_by_category(config.VAL_DIR)

    for cat, imgs in train_images.items():
        print(f"  {cat}: {len(imgs)} training images")

    # Step 2: Generate triplets
    print(f"\nGenerating triplets (per anchor: {config.TRIPLETS_PER_ANCHOR})...")
    train_triplets = generate_triplets(
        train_images, config.TRIPLETS_PER_ANCHOR
    )
    val_triplets = generate_triplets(val_images, 2)  # Fewer val triplets
    print(f"  Training triplets: {len(train_triplets)}")
    print(f"  Validation triplets: {len(val_triplets)}")

    # Step 3: Create data generators
    train_gen = TripletDataGenerator(train_triplets, config.BATCH_SIZE)
    val_gen = TripletDataGenerator(val_triplets, config.BATCH_SIZE)

    # Step 4: Build model
    siamese_model, embedding_network = build_siamese_model()

    # Step 5: Train
    history = train_siamese(siamese_model, train_gen, val_gen)

    # Step 6: Save models
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    embedding_network.save(config.SIAMESE_MODEL_PATH)
    print(f"Embedding network saved to {config.SIAMESE_MODEL_PATH}")

    # Step 7: Plot loss
    plot_path = os.path.join(config.PLOTS_DIR, "siamese_loss.png")
    plot_siamese_loss(history, save_path=plot_path)

    # Step 8: Extract Siamese embeddings
    extract_siamese_embeddings(embedding_network)

    print("\n[OK] Siamese network training complete!")


if __name__ == "__main__":
    main()
