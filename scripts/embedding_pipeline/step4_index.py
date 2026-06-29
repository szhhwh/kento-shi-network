#!/usr/bin/env python3
"""Step 4: Build search index from encoded vectors (numpy-based).

For 34K vectors, pure numpy brute-force search is faster than FAISS
(no overhead, all in memory). Saves vectors as memory-mapped file.
"""

import time

import numpy as np

from embedding_pipeline.config import VECTORS_NPY, FAISS_INDEX


def main():
    print("=" * 60)
    print("Step 4: Build Search Index (numpy)")
    print("=" * 60)

    # Load vectors
    print(f"\n  Loading vectors from: {VECTORS_NPY}")
    vectors = np.load(VECTORS_NPY).astype(np.float32)
    print(f"  Vector shape: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1)
    print(f"  Norm range: [{norms.min():.6f}, {norms.max():.6f}]")

    # Save as a binary file (columns = 768, rows = N, float32, C order)
    print(f"\n  Saving index to: {FAISS_INDEX}")
    vectors.tofile(str(FAISS_INDEX))

    # Store metadata alongside
    dim = vectors.shape[1]
    n_total = vectors.shape[0]
    import json

    meta = {"dim": dim, "n_total": n_total, "dtype": "float32"}
    meta_path = str(FAISS_INDEX) + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # Verify with self-search
    print("\n  Verifying (self-search top-3 via numpy)...")
    t0 = time.time()
    query = vectors[:1]
    scores = np.dot(query, vectors.T)[0]  # cosine similarity (normalized)
    top_k = np.argsort(-scores)[:3]
    for rank, idx in enumerate(top_k):
        print(f"    Rank {rank + 1}: chunk {idx}, score={scores[idx]:.6f}")
    elapsed = time.time() - t0
    print(f"  Search time for 1 query: {elapsed * 1000:.1f}ms")

    print(f"\n  Index saved: {FAISS_INDEX} ({vectors.nbytes / 1024 / 1024:.1f}MB)")
    print("Step 4 complete.")


if __name__ == "__main__":
    main()
