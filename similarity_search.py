"""
Similarity Search module.

Implements cosine similarity and FAISS-based approximate nearest
neighbor search for retrieving top-K similar products.

Usage:
    python similarity_search.py
"""
import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config
from feature_extractor import load_embeddings


class CosineSimilaritySearch:
    """Exact cosine similarity search using scikit-learn."""

    def __init__(self, embeddings, ids, categories, paths):
        """
        Initialize with database embeddings.

        Args:
            embeddings: numpy array (N, D).
            ids: array of product IDs.
            categories: array of category labels.
            paths: array of image paths.
        """
        self.embeddings = embeddings
        self.ids = ids
        self.categories = categories
        self.paths = paths
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.normalized = embeddings / (norms + 1e-8)
        print(f"CosineSimilaritySearch initialized with {len(ids)} items")

    def search(self, query_embedding, top_k=config.DEFAULT_TOP_K,
               exclude_self=True, query_id=None):
        """
        Find top-K similar items to query.

        Args:
            query_embedding: 1D array (D,) or 2D array (1, D).
            top_k: Number of results to return.
            exclude_self: Whether to exclude the query from results.
            query_id: ID of query image (for self-exclusion).

        Returns:
            Dict with keys: ids, categories, paths, scores, indices.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Compute similarities
        similarities = cosine_similarity(query_norm, self.normalized)[0]

        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Optionally exclude self
        if exclude_self and query_id is not None:
            sorted_indices = [i for i in sorted_indices
                              if str(self.ids[i]) != str(query_id)]

        # Take top K
        top_indices = sorted_indices[:top_k]

        return {
            "ids": [self.ids[i] for i in top_indices],
            "categories": [self.categories[i] for i in top_indices],
            "paths": [self.paths[i] for i in top_indices],
            "scores": [float(similarities[i]) for i in top_indices],
            "indices": top_indices,
        }


class FAISSSearch:
    """
    Approximate nearest neighbor search using FAISS.
    Falls back to cosine similarity if FAISS is not available.
    """

    def __init__(self, embeddings, ids, categories, paths):
        """
        Initialize FAISS index with database embeddings.

        Args:
            embeddings: numpy array (N, D).
            ids: array of product IDs.
            categories: array of category labels.
            paths: array of image paths.
        """
        self.ids = ids
        self.categories = categories
        self.paths = paths
        self.use_faiss = False

        try:
            import faiss

            # Normalize for cosine similarity via inner product
            normalized = embeddings.copy().astype(np.float32)
            norms = np.linalg.norm(normalized, axis=1, keepdims=True)
            normalized = normalized / (norms + 1e-8)

            # Build FAISS index (Inner Product = cosine sim for normalized vectors)
            dim = normalized.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(normalized)
            self.use_faiss = True
            print(f"FAISS index built with {self.index.ntotal} vectors (dim={dim})")

        except ImportError:
            print("FAISS not available. Falling back to cosine similarity.")
            self.fallback = CosineSimilaritySearch(
                embeddings, ids, categories, paths
            )

    def search(self, query_embedding, top_k=config.DEFAULT_TOP_K,
               exclude_self=True, query_id=None):
        """
        Find top-K similar items using FAISS.

        Args:
            query_embedding: 1D array (D,) or 2D array (1, D).
            top_k: Number of results to return.
            exclude_self: Whether to exclude the query from results.
            query_id: ID of query image (for self-exclusion).

        Returns:
            Dict with keys: ids, categories, paths, scores, indices.
        """
        if not self.use_faiss:
            return self.fallback.search(
                query_embedding, top_k, exclude_self, query_id
            )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query
        query = query_embedding.astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        # Search (get extra results in case we need to exclude self)
        search_k = top_k + 5
        scores, indices = self.index.search(query, search_k)
        scores = scores[0]
        indices = indices[0]

        # Filter
        results_ids = []
        results_cats = []
        results_paths = []
        results_scores = []
        results_indices = []

        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            if exclude_self and query_id and str(self.ids[idx]) == str(query_id):
                continue
            results_ids.append(self.ids[idx])
            results_cats.append(self.categories[idx])
            results_paths.append(self.paths[idx])
            results_scores.append(float(score))
            results_indices.append(idx)
            if len(results_ids) >= top_k:
                break

        return {
            "ids": results_ids,
            "categories": results_cats,
            "paths": results_paths,
            "scores": results_scores,
            "indices": results_indices,
        }


def benchmark_search(search_engine, embeddings, n_queries=50, top_k=5):
    """
    Benchmark search speed.

    Args:
        search_engine: Search engine instance.
        embeddings: Database embeddings.
        n_queries: Number of random queries.
        top_k: Number of results per query.

    Returns:
        Average query time in milliseconds.
    """
    indices = np.random.choice(len(embeddings), n_queries, replace=False)
    times = []

    for idx in indices:
        query = embeddings[idx]
        start = time.time()
        search_engine.search(query, top_k=top_k)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)

    avg_time = np.mean(times)
    print(f"Average query time ({n_queries} queries, K={top_k}): {avg_time:.2f}ms")
    return avg_time


def main():
    """Demo similarity search with baseline embeddings."""
    print("=" * 60)
    print("SIMILARITY SEARCH DEMO")
    print("=" * 60)

    # Load baseline embeddings
    try:
        embeddings, ids, categories, paths = load_embeddings("baseline")
    except FileNotFoundError:
        print("Baseline embeddings not found. Run feature_extractor.py first.")
        return

    # Initialize search engines
    print("\n--- Cosine Similarity Search ---")
    cosine_search = CosineSimilaritySearch(embeddings, ids, categories, paths)

    print("\n--- FAISS Search ---")
    faiss_search = FAISSSearch(embeddings, ids, categories, paths)

    # Demo query
    query_idx = 0
    query_embedding = embeddings[query_idx]
    query_id = ids[query_idx]
    query_category = categories[query_idx]

    print(f"\nQuery image: ID={query_id}, Category={query_category}")

    # Cosine similarity results
    print("\n--- Cosine Similarity Results ---")
    results = cosine_search.search(query_embedding, top_k=5, query_id=query_id)
    for i, (rid, rcat, rscore) in enumerate(
            zip(results["ids"], results["categories"], results["scores"])):
        print(f"  {i + 1}. ID={rid}, Category={rcat}, Score={rscore:.4f}")

    # FAISS results
    print("\n--- FAISS Results ---")
    results = faiss_search.search(query_embedding, top_k=5, query_id=query_id)
    for i, (rid, rcat, rscore) in enumerate(
            zip(results["ids"], results["categories"], results["scores"])):
        print(f"  {i + 1}. ID={rid}, Category={rcat}, Score={rscore:.4f}")

    # Benchmark
    print("\n--- Benchmarks ---")
    benchmark_search(cosine_search, embeddings, n_queries=50)
    benchmark_search(faiss_search, embeddings, n_queries=50)

    print("\n✓ Similarity search demo complete!")


if __name__ == "__main__":
    main()
