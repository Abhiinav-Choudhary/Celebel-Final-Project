"""
Evaluation Module for Visual Product Recommendation Engine.

Computes Precision@K, Recall@K, and inference time metrics.
Generates comparison charts between Baseline, Fine-tuned, and Siamese models.

Usage:
    python evaluate.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from collections import defaultdict
import config
from feature_extractor import load_embeddings
from similarity_search import CosineSimilaritySearch


def compute_precision_at_k(search_engine, embeddings, categories, k):
    """
    Compute Precision@K: proportion of relevant (same category) items
    among the top-K retrieved results, averaged over all queries.

    Args:
        search_engine: Search engine instance.
        embeddings: All embeddings.
        categories: Category labels for each embedding.
        k: Number of results to consider.

    Returns:
        Average Precision@K across all queries.
    """
    precisions = []

    for i in range(len(embeddings)):
        query_embedding = embeddings[i]
        query_category = categories[i]
        query_id = str(i)  # Use index as ID for exclusion

        results = search_engine.search(
            query_embedding, top_k=k,
            exclude_self=True, query_id=None
        )

        # Since we can't reliably exclude self by ID, skip first result
        # if its score is exactly 1.0 (perfect match = self)
        result_categories = results["categories"]
        if results["scores"] and results["scores"][0] > 0.9999:
            result_categories = result_categories[1:k + 1]
        else:
            result_categories = result_categories[:k]

        # Count relevant (same category) items
        relevant = sum(1 for cat in result_categories if cat == query_category)
        precisions.append(relevant / k if k > 0 else 0)

    return np.mean(precisions)


def compute_recall_at_k(search_engine, embeddings, categories, k):
    """
    Compute Recall@K: proportion of relevant items retrieved from
    all relevant items available, averaged over all queries.

    Args:
        search_engine: Search engine instance.
        embeddings: All embeddings.
        categories: Category labels for each embedding.
        k: Number of results to consider.

    Returns:
        Average Recall@K across all queries.
    """
    # Count total relevant items per category
    category_counts = defaultdict(int)
    for cat in categories:
        category_counts[cat] += 1

    recalls = []

    for i in range(len(embeddings)):
        query_embedding = embeddings[i]
        query_category = categories[i]

        results = search_engine.search(
            query_embedding, top_k=k,
            exclude_self=True, query_id=None
        )

        result_categories = results["categories"]
        if results["scores"] and results["scores"][0] > 0.9999:
            result_categories = result_categories[1:k + 1]
        else:
            result_categories = result_categories[:k]

        # Count relevant items retrieved
        relevant_retrieved = sum(
            1 for cat in result_categories if cat == query_category
        )
        # Total relevant (minus self)
        total_relevant = category_counts[query_category] - 1
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        recalls.append(recall)

    return np.mean(recalls)


def benchmark_inference_time(search_engine, embeddings, n_queries=100, k=5):
    """
    Measure average inference time per query.

    Args:
        search_engine: Search engine instance.
        embeddings: Database embeddings.
        n_queries: Number of queries to benchmark.
        k: Top-K results.

    Returns:
        Average time per query in milliseconds.
    """
    indices = np.random.choice(len(embeddings), min(n_queries, len(embeddings)),
                               replace=False)
    times = []

    for idx in indices:
        start = time.time()
        search_engine.search(embeddings[idx], top_k=k)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    return np.mean(times)


def evaluate_model(prefix, k_values=config.TOP_K_VALUES):
    """
    Run full evaluation for a model's embeddings.

    Args:
        prefix: Embedding prefix (baseline/finetuned/siamese).
        k_values: List of K values to evaluate.

    Returns:
        Dict with precision, recall, and timing metrics.
    """
    print(f"\n{'─' * 40}")
    print(f"Evaluating: {prefix.upper()}")
    print(f"{'─' * 40}")

    try:
        embeddings, ids, categories, paths = load_embeddings(prefix)
    except FileNotFoundError:
        print(f"  Embeddings for '{prefix}' not found. Skipping.")
        return None

    # Build search engine
    search = CosineSimilaritySearch(embeddings, ids, categories, paths)

    # Compute metrics
    precisions = []
    recalls = []
    print(f"\n  Computing metrics for K = {k_values}...")

    for k in k_values:
        p = compute_precision_at_k(search, embeddings, categories, k)
        r = compute_recall_at_k(search, embeddings, categories, k)
        precisions.append(p)
        recalls.append(r)
        print(f"    K={k:2d}: Precision={p:.4f}, Recall={r:.4f}")

    # Benchmark inference time
    avg_query_time = benchmark_inference_time(search, embeddings)
    print(f"\n  Avg query time: {avg_query_time:.2f}ms")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Database size: {len(embeddings)} images")

    return {
        "prefix": prefix,
        "precisions": precisions,
        "recalls": recalls,
        "avg_query_time_ms": avg_query_time,
        "embedding_dim": embeddings.shape[1],
        "n_images": len(embeddings),
    }


def plot_comparison_charts(all_results, k_values=config.TOP_K_VALUES):
    """
    Generate comparison bar charts for all models.

    Args:
        all_results: List of result dicts from evaluate_model().
        k_values: K values used.
    """
    if not all_results:
        print("No results to plot.")
        return

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    # ─── Precision@K Comparison ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(k_values))
    width = 0.8 / len(all_results)

    for i, result in enumerate(all_results):
        offset = (i - len(all_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, result["precisions"], width,
                      label=result["prefix"].capitalize(),
                      color=colors[i % len(colors)], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, result["precisions"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Precision@K", fontsize=12)
    ax.set_title("Precision@K Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "precision_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # ─── Recall@K Comparison ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, result in enumerate(all_results):
        offset = (i - len(all_results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, result["recalls"], width,
                      label=result["prefix"].capitalize(),
                      color=colors[i % len(colors)], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, result["recalls"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_title("Recall@K Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "recall_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    # ─── Inference Time Comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["prefix"].capitalize() for r in all_results]
    times = [r["avg_query_time_ms"] for r in all_results]
    dims = [r["embedding_dim"] for r in all_results]

    bars = ax.bar(names, times, color=colors[:len(all_results)],
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, t, d in zip(bars, times, dims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{t:.2f}ms\n(dim={d})", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Average Query Time (ms)", fontsize=12)
    ax.set_title("Inference Time Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, "inference_time.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def generate_visual_comparison(n_queries=5, top_k=5):
    """
    Generate visual comparison of retrieval results across models.

    Args:
        n_queries: Number of random query images to compare.
        top_k: Number of results per query.
    """
    print("\nGenerating visual comparison...")
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    models = {}
    for prefix in ["baseline", "finetuned", "siamese"]:
        try:
            embeddings, ids, categories, paths = load_embeddings(prefix)
            search = CosineSimilaritySearch(embeddings, ids, categories, paths)
            models[prefix] = (embeddings, ids, categories, paths, search)
        except FileNotFoundError:
            continue

    if not models:
        print("No model embeddings found.")
        return

    # Use baseline paths for query selection
    first_model = list(models.values())[0]
    all_paths = first_model[3]
    all_categories = first_model[2]

    # Pick random queries
    np.random.seed(config.RANDOM_SEED)
    query_indices = np.random.choice(len(all_paths), n_queries, replace=False)

    for qi, query_idx in enumerate(query_indices):
        fig, axes = plt.subplots(
            len(models), top_k + 1,
            figsize=(3 * (top_k + 1), 3 * len(models))
        )
        if len(models) == 1:
            axes = axes.reshape(1, -1)

        query_path = all_paths[query_idx]
        query_cat = all_categories[query_idx]

        for mi, (model_name, (embs, ids, cats, paths, search)) in enumerate(models.items()):
            # Query
            query_emb = embs[query_idx]
            results = search.search(query_emb, top_k=top_k + 1)

            # Plot query image
            try:
                img = Image.open(query_path).convert("RGB").resize(config.IMG_SIZE)
                axes[mi, 0].imshow(img)
            except Exception:
                pass
            axes[mi, 0].set_title(f"Query\n({query_cat})", fontsize=9,
                                   fontweight="bold", color="#e74c3c")
            axes[mi, 0].axis("off")
            axes[mi, 0].set_ylabel(model_name.capitalize(), fontsize=11,
                                    fontweight="bold")

            # Plot results
            result_idx = 0
            for j in range(min(top_k, len(results["paths"]))):
                rpath = results["paths"][j]
                rcat = results["categories"][j]
                rscore = results["scores"][j]

                # Skip self
                if rscore > 0.9999:
                    continue

                col = result_idx + 1
                if col >= top_k + 1:
                    break

                try:
                    img = Image.open(rpath).convert("RGB").resize(config.IMG_SIZE)
                    axes[mi, col].imshow(img)
                except Exception:
                    pass

                color = "#2ecc71" if rcat == query_cat else "#e74c3c"
                axes[mi, col].set_title(f"{rcat}\n{rscore:.3f}", fontsize=8,
                                         color=color)
                axes[mi, col].axis("off")
                result_idx += 1

            # Clear unused axes
            for j in range(result_idx + 1, top_k + 1):
                axes[mi, j].axis("off")

        fig.suptitle(f"Query #{qi + 1}: {query_cat}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_path = os.path.join(config.PLOTS_DIR, f"visual_comparison_{qi + 1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close()


def print_summary_table(all_results, k_values=config.TOP_K_VALUES):
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Header
    header = f"{'Model':<12}"
    for k in k_values:
        header += f"{'P@' + str(k):<10}"
    for k in k_values:
        header += f"{'R@' + str(k):<10}"
    header += f"{'Time(ms)':<10}{'Dim':<6}"
    print(header)
    print("─" * 70)

    for result in all_results:
        row = f"{result['prefix']:<12}"
        for p in result["precisions"]:
            row += f"{p:<10.4f}"
        for r in result["recalls"]:
            row += f"{r:<10.4f}"
        row += f"{result['avg_query_time_ms']:<10.2f}"
        row += f"{result['embedding_dim']:<6}"
        print(row)

    print("=" * 70)


def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("EVALUATION — MODEL COMPARISON")
    print("=" * 60)

    all_results = []

    # Evaluate all available models
    for prefix in ["baseline", "finetuned", "siamese"]:
        result = evaluate_model(prefix)
        if result:
            all_results.append(result)

    if not all_results:
        print("No embeddings found. Run the extraction pipelines first.")
        return

    # Print summary
    print_summary_table(all_results)

    # Generate comparison plots
    plot_comparison_charts(all_results)

    # Generate visual comparisons
    from PIL import Image
    generate_visual_comparison(n_queries=5, top_k=5)

    print("\n[OK] Evaluation complete! Check plots/ directory for visualizations.")


if __name__ == "__main__":
    main()
