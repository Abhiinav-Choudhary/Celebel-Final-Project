"""
Transfer Learning script.
Fine-tunes a pre-trained ResNet50 model by appending a custom
classification head to adapt to the specific dataset domains.

Usage:
    python transfer_learning.py
"""
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import project_config as config
from feature_extractor import extract_embeddings, save_embeddings


def build_finetuned_model(num_classes):
    """
    Appends a custom classification head to a pre-trained ResNet50 base.
    Freezes the majority of the base model layers while unfreezing the final
    layers for domain-specific fine-tuning.
    """
    print(f"Building fine-tuned ResNet50 (classes={num_classes})...")

    # Load pretrained ResNet50 without classification head
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=config.IMG_SHAPE
    )

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last N layers for fine-tuning
    for layer in base_model.layers[-config.TL_UNFREEZE_LAYERS:]:
        layer.trainable = True

    trainable = sum(1 for l in base_model.layers if l.trainable)
    frozen = sum(1 for l in base_model.layers if not l.trainable)
    print(f"  Trainable layers: {trainable}, Frozen layers: {frozen}")

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu", name="embedding_layer")(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation="softmax", name="classification")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=config.TL_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"  Total parameters: {model.count_params():,}")
    return model


def create_data_generators():
    """
    Initializes data generators with data augmentation.
    Yields batches of augmented images to manage memory footprint efficiently.
    """
    print("Creating data generators with augmentation...")

    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode="nearest"
    )

    # Validation generator (no augmentation, just preprocessing)
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_SEED
    )

    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())

    print(f"  Classes: {class_names}")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")

    return train_generator, val_generator, num_classes, class_names


def train_model(model, train_gen, val_gen):
    """
    Train the fine-tuned model.

    Args:
        model: Compiled Keras model.
        train_gen: Training data generator.
        val_gen: Validation data generator.

    Returns:
        Training history.
    """
    print(f"\nTraining for up to {config.TL_EPOCHS} epochs...")

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            config.FINETUNED_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    start_time = time.time()
    history = model.fit(
        train_gen,
        epochs=config.TL_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  Best val loss: {min(history.history['val_loss']):.4f}")

    return history


def build_embedding_extractor(model):
    """
    Build an embedding extractor from the trained model.
    Extracts features from the 'embedding_layer' (256-dim).

    Args:
        model: Trained classification model.

    Returns:
        Model that outputs embeddings.
    """
    embedding_layer = model.get_layer("embedding_layer")
    extractor = Model(inputs=model.input, outputs=embedding_layer.output)
    print(f"Embedding extractor built. Output dim: {extractor.output_shape[-1]}")
    return extractor


def extract_finetuned_embeddings(extractor):
    """
    Extract embeddings using the fine-tuned model for all images.

    Args:
        extractor: Fine-tuned embedding extractor model.
    """
    print("\nExtracting fine-tuned embeddings...")

    all_embeddings = []
    all_ids = []
    all_categories = []
    all_paths = []

    for split_name, split_dir in [("train", config.TRAIN_DIR),
                                   ("val", config.VAL_DIR)]:
        if os.path.exists(split_dir):
            embeddings, ids, categories, paths = extract_embeddings(
                extractor, split_dir, config.BATCH_SIZE
            )
            all_embeddings.append(embeddings)
            all_ids.extend(ids)
            all_categories.extend(categories)
            all_paths.extend(paths)

    combined = np.vstack(all_embeddings)
    save_embeddings(combined, all_ids, all_categories, all_paths, "finetuned")


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras training history.
        save_path: Optional path to save the plot.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training plot saved to {save_path}")
    # plt.show() - Removed to prevent blocking headless execution


def main():
    """Main transfer learning pipeline."""
    print("=" * 60)
    print("TRANSFER LEARNING — FINE-TUNING ResNet50")
    print("=" * 60)

    # Check that subset exists
    if not os.path.exists(config.TRAIN_DIR):
        print(f"ERROR: Training data not found at {config.TRAIN_DIR}")
        print("Please run data_preparation.py first.")
        return

    # Create data generators
    train_gen, val_gen, num_classes, class_names = create_data_generators()

    # Build model
    model = build_finetuned_model(num_classes)

    # Train
    history = train_model(model, train_gen, val_gen)

    # Plot training curves
    plot_path = os.path.join(config.PLOTS_DIR, "training_history.png")
    plot_training_history(history, save_path=plot_path)

    # Build embedding extractor and extract embeddings
    extractor = build_embedding_extractor(model)
    extract_finetuned_embeddings(extractor)

    print("\n[OK] Transfer learning complete!")


if __name__ == "__main__":
    main()
