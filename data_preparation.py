"""
Dataset Preparation for Visual Product Recommendation Engine.

Loads the Fashion Product Images dataset, creates a filtered subset
with selected categories, and splits into train/val directories.

Usage:
    python data_preparation.py
"""
import os
import shutil
import random
import pandas as pd
import numpy as np
from PIL import Image
import config
import utils


def download_dataset():
    """
    Download the Fashion Product Images (Small) dataset from Kaggle.
    Requires kaggle CLI to be installed and configured.
    Falls back to manual download instructions if not available.
    """
    target_dir = config.RAW_DATA_DIR
    os.makedirs(target_dir, exist_ok=True)

    # Check if dataset already exists
    styles_path = config.STYLES_CSV
    images_path = config.IMAGES_DIR
    if os.path.exists(styles_path) and os.path.exists(images_path):
        print("Dataset already exists. Skipping download.")
        return True

    try:
        import subprocess
        print("Downloading Fashion Product Images (Small) from Kaggle...")
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "paramaggarwal/fashion-product-images-small",
                "-p", target_dir, "--unzip"
            ],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("Dataset downloaded and extracted successfully!")
            # The small dataset extracts into a subdirectory, fix paths
            _fix_extracted_paths(target_dir)
            return True
        else:
            print(f"Kaggle download failed: {result.stderr}")
    except FileNotFoundError:
        print("Kaggle CLI not found.")
    except Exception as e:
        print(f"Download error: {e}")

    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("Please download the dataset from:")
    print("  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small")
    print(f"\nExtract to: {target_dir}")
    print("Expected structure:")
    print(f"  {target_dir}/")
    print(f"    styles.csv")
    print(f"    images/")
    print(f"      1163.jpg")
    print(f"      1164.jpg")
    print(f"      ...")
    print("=" * 60)
    return False


def _fix_extracted_paths(target_dir):
    # It's a bit messy with nested folders, so we move everything up to the raw folder.
    # Check for common nested patterns
    possible_subdirs = [
        os.path.join(target_dir, "fashion-product-images-small"),
        os.path.join(target_dir, "myntradataset"),
    ]

    for subdir in possible_subdirs:
        if os.path.exists(subdir):
            # Move contents up one level
            for item in os.listdir(subdir):
                src = os.path.join(subdir, item)
                dst = os.path.join(target_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            # Remove empty subdirectory
            if not os.listdir(subdir):
                os.rmdir(subdir)
            print(f"Fixed extracted paths from {subdir}")
            break


def load_metadata():
    """
    Reads the big styles.csv file and filters it down to just the categories we care about.
    """
    print("Loading metadata from styles.csv...")
    # Kaggle CSV has a bad line sometimes, skip it to avoid crashing.
    df = pd.read_csv(config.STYLES_CSV, on_bad_lines="skip")

    print(f"Total products in dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Display available categories
    if "subCategory" in df.columns:
        print(f"\nAvailable subCategories:")
        category_counts = df["subCategory"].value_counts()
        for cat, count in category_counts.head(20).items():
            print(f"  {cat}: {count}")

    if "articleType" in df.columns:
        print(f"\nAvailable articleTypes:")
        type_counts = df["articleType"].value_counts()
        for atype, count in type_counts.head(20).items():
            print(f"  {atype}: {count}")

    return df


def create_subset(df):
    """
    Filter the dataset to selected categories and sample a subset.

    Args:
        df: Full metadata DataFrame.

    Returns:
        Filtered DataFrame with the subset.
    """
    print(f"\nCreating subset with categories: {config.SELECTED_CATEGORIES}")
    print(f"Samples per category: {config.SAMPLES_PER_CATEGORY}")

    # Filter by articleType (more specific than masterCategory)
    subset_frames = []
    for category in config.SELECTED_CATEGORIES:
        cat_df = df[df["articleType"] == category]
        if len(cat_df) == 0:
            # Try subCategory as fallback
            cat_df = df[df["subCategory"] == category]
        if len(cat_df) == 0:
            print(f"  WARNING: Category '{category}' not found. Skipping.")
            continue

        # Filter to only products with existing images
        cat_df = cat_df[cat_df["id"].apply(
            lambda x: os.path.exists(utils.get_image_path_from_id(x))
        )]

        # Sample
        n_samples = min(config.SAMPLES_PER_CATEGORY, len(cat_df))
        sampled = cat_df.sample(n=n_samples, random_state=config.RANDOM_SEED)
        subset_frames.append(sampled)
        print(f"  {category}: {n_samples} images (from {len(cat_df)} available)")

    subset_df = pd.concat(subset_frames, ignore_index=True)
    print(f"\nTotal subset size: {len(subset_df)}")
    return subset_df


def split_and_organize_images(subset_df):
    """
    Copies image files into organized train and validation directories.
    """
    print("\nOrganizing into train/val splits...")

    # Clean any existing subset directory
    if os.path.exists(config.SUBSET_DIR):
        shutil.rmtree(config.SUBSET_DIR)

    # Group by articleType
    for category in subset_df["articleType"].unique():
        cat_df = subset_df[subset_df["articleType"] == category]
        # Safe directory name (replace spaces, special chars)
        safe_cat = category.replace(" ", "_").replace("/", "_")

        # Create train and val directories
        train_cat_dir = os.path.join(config.TRAIN_DIR, safe_cat)
        val_cat_dir = os.path.join(config.VAL_DIR, safe_cat)
        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(val_cat_dir, exist_ok=True)

        # Shuffle and split
        indices = list(cat_df.index)
        random.seed(config.RANDOM_SEED)
        random.shuffle(indices)
        split_idx = int(len(indices) * (1 - config.VALIDATION_SPLIT))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Copy images
        for idx in train_indices:
            row = subset_df.loc[idx]
            src = utils.get_image_path_from_id(row["id"])
            dst = os.path.join(train_cat_dir, f"{row['id']}.jpg")
            if os.path.exists(src):
                shutil.copy2(src, dst)

        for idx in val_indices:
            row = subset_df.loc[idx]
            src = utils.get_image_path_from_id(row["id"])
            dst = os.path.join(val_cat_dir, f"{row['id']}.jpg")
            if os.path.exists(src):
                shutil.copy2(src, dst)

        print(f"  {category}: {len(train_indices)} train, {len(val_indices)} val")

    # Save subset metadata
    metadata_path = config.METADATA_PATH
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    subset_df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path}")


def validate_subset():
    """
    Validates the integrity of the data split and tests image loading.
    """
    print("\n" + "=" * 40)
    print("SUBSET VALIDATION")
    print("=" * 40)

    total = 0
    for split in ["train", "val"]:
        split_dir = os.path.join(config.SUBSET_DIR, split)
        if not os.path.exists(split_dir):
            print(f"  {split} directory not found!")
            continue
        for category in sorted(os.listdir(split_dir)):
            cat_dir = os.path.join(split_dir, category)
            if os.path.isdir(cat_dir):
                n_images = len([f for f in os.listdir(cat_dir) if f.endswith(".jpg")])
                total += n_images
                print(f"  {split}/{category}: {n_images} images")

    print(f"\nTotal images in subset: {total}")

    # Validate a few images can be loaded
    sample_dir = config.TRAIN_DIR
    if os.path.exists(sample_dir):
        categories = os.listdir(sample_dir)
        if categories:
            sample_cat = os.path.join(sample_dir, categories[0])
            images = [f for f in os.listdir(sample_cat) if f.endswith(".jpg")]
            if images:
                test_img_path = os.path.join(sample_cat, images[0])
                img = Image.open(test_img_path)
                print(f"\nSample image size: {img.size}")
                print(f"Sample image mode: {img.mode}")
                print("Image loading validation: PASSED ✓")


def main():
    """Main data preparation pipeline."""
    print("=" * 60)
    print("VISUAL PRODUCT RECOMMENDATION — DATA PREPARATION")
    print("=" * 60)

    # Ensure directories exist
    utils.ensure_dirs()

    # Step 1: Download dataset
    download_dataset()

    # Step 2: Check if dataset exists
    if not os.path.exists(config.STYLES_CSV):
        print(f"\nERROR: styles.csv not found at {config.STYLES_CSV}")
        print("Please download the dataset first.")
        return

    # Step 3: Load metadata
    df = load_metadata()

    # Step 4: Create subset
    subset_df = create_subset(df)

    # Step 5: Split and organize
    split_and_organize(subset_df)

    # Step 6: Validate
    validate_subset()

    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()
