"""
Configuration and settings for the product recommendation engine.
Contains all paths and hyperparameters used across modules.
"""
import os

# --- Directory Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
SUBSET_DIR = os.path.join(DATA_DIR, "subset")
TRAIN_DIR = os.path.join(SUBSET_DIR, "train")
VAL_DIR = os.path.join(SUBSET_DIR, "val")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Dataset file paths
STYLES_CSV = os.path.join(RAW_DATA_DIR, "styles.csv")
IMAGES_DIR = os.path.join(RAW_DATA_DIR, "images")

# --- Dataset setup ---
# Selected categories for the training subset
SELECTED_CATEGORIES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Watches",
    "Tops",
    "Handbags",
    "Sandals",
]

SAMPLES_PER_CATEGORY = 250        # Number of items per category for subsetting
VALIDATION_SPLIT = 0.2            # Train/validation split ratio
RANDOM_SEED = 42                  # For reproducibility

# --- Image Processing ---
IMG_SIZE = (224, 224)              # Target input size for ResNet50
IMG_SHAPE = (224, 224, 3)

# Mean and std dev for ImageNet (used to normalize our images before feeding to ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Model Hyperparameters ---
BATCH_SIZE = 32
EMBEDDING_DIM = 128               # Siamese embedding dimension

# Fine-tuning settings for the Transfer Learning model
TL_EPOCHS = 10
TL_LEARNING_RATE = 1e-4
TL_UNFREEZE_LAYERS = 20           # Just unfreeze the last 20 layers, freeze the rest

# Siamese Network settings
SIAMESE_EPOCHS = 15
SIAMESE_LEARNING_RATE = 1e-4
SIAMESE_MARGIN = 1.0              # Triplet loss margin
TRIPLETS_PER_ANCHOR = 5           # Number of triplets generated per anchor image

# --- App Settings ---
TOP_K_VALUES = [1, 3, 5, 10]      # We'll evaluate accuracy at these K values
DEFAULT_TOP_K = 5                 # Show 5 recommendations in the UI by default

# --- Save Locations ---
BASELINE_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "baseline_embeddings.npy")
BASELINE_IDS_PATH = os.path.join(EMBEDDINGS_DIR, "baseline_ids.npy")

FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, "finetuned_resnet50.keras")
FINETUNED_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "finetuned_embeddings.npy")
FINETUNED_IDS_PATH = os.path.join(EMBEDDINGS_DIR, "finetuned_ids.npy")

SIAMESE_MODEL_PATH = os.path.join(MODELS_DIR, "siamese_model.keras")
SIAMESE_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "siamese_embeddings.npy")
SIAMESE_IDS_PATH = os.path.join(EMBEDDINGS_DIR, "siamese_ids.npy")

METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.csv")
