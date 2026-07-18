"""
Helper functions for our app. 
This handles all the messy stuff like loading images, resizing them, and drawing pretty charts.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import project_config as config


def load_and_preprocess_image(image_path, target_size=config.IMG_SIZE):
    """
    Grabs an image from the hard drive, resizes it, and gets it ready for the model.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    # Normalize to [0, 1] then apply ImageNet normalization
    img_array = img_array / 255.0
    img_array = (img_array - config.IMAGENET_MEAN) / config.IMAGENET_STD
    return np.expand_dims(img_array, axis=0)


def load_image_for_display(image_path, target_size=config.IMG_SIZE):
    """
    Just load the image so we can show it on the screen (no weird normalization needed).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return img


def preprocess_batch(image_paths, target_size=config.IMG_SIZE):
    """
    Processes a whole batch of images at once so we can feed them to the model faster.
    """
    batch = []
    for path in image_paths:
        try:
            img = load_and_preprocess_image(path, target_size)
            batch.append(img[0])
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    return np.array(batch)


def get_image_path_from_id(product_id, images_dir=config.IMAGES_DIR):
    """
    Quick helper to figure out the full file path if we just have the product ID.
    """
    return os.path.join(images_dir, f"{product_id}.jpg")


def plot_query_results(query_image_path, result_image_paths, result_scores,
                       result_categories=None, title="Similar Products",
                       save_path=None):
    """
    Draws a nice row of images showing our original query on the left, 
    and the top recommended products next to it.
    """
    n_results = len(result_image_paths)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 4))

    # Plot query image
    query_img = load_image_for_display(query_image_path)
    axes[0].imshow(query_img)
    axes[0].set_title("Query", fontsize=12, fontweight="bold", color="#e74c3c")
    axes[0].axis("off")

    # Plot results
    for i, (img_path, score) in enumerate(zip(result_image_paths, result_scores)):
        try:
            img = load_image_for_display(img_path)
            axes[i + 1].imshow(img)
            label = f"Score: {score:.3f}"
            if result_categories and i < len(result_categories):
                label = f"{result_categories[i]}\n{label}"
            axes[i + 1].set_title(label, fontsize=10)
        except Exception:
            axes[i + 1].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[i + 1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_comparison(metrics_dict, k_values=config.TOP_K_VALUES,
                    metric_name="Precision@K", save_path=None):
    """
    Draws a bar chart so we can easily compare how our different models are doing.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(k_values))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for i, (model_name, values) in enumerate(metrics_dict.items()):
        bars = ax.bar(x + i * width, values, width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def ensure_dirs():
    """Make sure all our folders exist before we try to save things into them."""
    dirs = [
        config.DATA_DIR, config.RAW_DATA_DIR, config.SUBSET_DIR,
        config.TRAIN_DIR, config.VAL_DIR, config.MODELS_DIR,
        config.EMBEDDINGS_DIR, config.PLOTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("All directories created successfully.")
