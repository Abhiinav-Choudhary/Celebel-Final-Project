"""
Master Training Script — Visual Product Recommendation Engine.

Runs all training pipelines in sequence:
  1. Extract finetuned embeddings (from already-trained finetuned_resnet50.keras)
  2. Train Siamese network (triplet loss)
  3. Extract Siamese embeddings

Usage:
    python train_all.py
"""
import os
import sys
import time

import project_config as config


def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─── Step 1: Extract finetuned embeddings ────────────────────────────────────

def run_finetuned_embeddings():
    banner("STEP 1 — Extracting Fine-tuned Embeddings")

    finetuned_emb = os.path.join(config.EMBEDDINGS_DIR, "finetuned_embeddings.npy")
    if os.path.exists(finetuned_emb):
        print(f"[SKIP] Finetuned embeddings already exist at {finetuned_emb}")
        return

    if not os.path.exists(config.FINETUNED_MODEL_PATH):
        print(f"[ERROR] Finetuned model not found at {config.FINETUNED_MODEL_PATH}")
        print("        Running transfer_learning.py to train it first...")
        import transfer_learning
        transfer_learning.main()
        return

    import numpy as np
    import tensorflow as tf
    from feature_extractor import extract_embeddings, save_embeddings
    from transfer_learning import build_embedding_extractor

    print(f"Loading finetuned model from {config.FINETUNED_MODEL_PATH}...")
    model = tf.keras.models.load_model(config.FINETUNED_MODEL_PATH)
    extractor = build_embedding_extractor(model)

    all_embeddings = []
    all_ids = []
    all_categories = []
    all_paths = []

    for split_name, split_dir in [("train", config.TRAIN_DIR), ("val", config.VAL_DIR)]:
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
    print("\n[OK] Finetuned embedding extraction complete!")


# ─── Step 2: Train Siamese network ───────────────────────────────────────────

def run_siamese_training():
    banner("STEP 2 — Training Siamese Network")

    siamese_emb = os.path.join(config.EMBEDDINGS_DIR, "siamese_embeddings.npy")
    if os.path.exists(siamese_emb) and os.path.exists(config.SIAMESE_MODEL_PATH):
        print(f"[SKIP] Siamese model and embeddings already exist.")
        return

    import siamese_network
    siamese_network.main()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    overall_start = time.time()
    banner("VISUAL PRODUCT RECOMMENDATION — FULL TRAINING PIPELINE")
    print(f"  Project root : {config.PROJECT_ROOT}")
    print(f"  Train dir    : {config.TRAIN_DIR}")
    print(f"  Models dir   : {config.MODELS_DIR}")
    print(f"  Embeddings   : {config.EMBEDDINGS_DIR}")

    # --- Verify data exists ---
    if not os.path.exists(config.TRAIN_DIR):
        print(f"\n[ERROR] Training data not found at {config.TRAIN_DIR}")
        print("        Please run data_preparation.py first.")
        sys.exit(1)

    run_finetuned_embeddings()
    run_siamese_training()

    elapsed = time.time() - overall_start
    banner("ALL TRAINING COMPLETE")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print("\nArtifacts produced:")
    for name in [
        "finetuned_embeddings.npy",
        "siamese_embeddings.npy",
        "siamese_model.keras",
    ]:
        path = os.path.join(
            config.EMBEDDINGS_DIR if name.endswith(".npy") else config.MODELS_DIR,
            name,
        )
        status = "✓" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {path}")

    print("\n[DONE] You can now run app.py or evaluate.py.")


if __name__ == "__main__":
    main()
