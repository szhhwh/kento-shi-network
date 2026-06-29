#!/usr/bin/env python3
"""Step 3: Encode text chunks into 768-dim vectors using SikuRoBERTa.

GPU-optimized: pre-tokenizes all texts, then batched model inference on GPU.
"""

import json
import time

import numpy as np
import torch

from embedding_pipeline.config import (
    CHUNKS_JSONL,
    VECTORS_NPY,
    MODEL_NAME,
    MODEL_FALLBACK,
    MAX_SEQ_LENGTH,
)

# ── Tunables ────────────────────────────────────────────────────────────────
ENCODE_BATCH_SIZE = 256  # GPU batch size
TOKENIZE_BATCH = 4096  # tokenizer sub-batch


def load_model_and_tokenizer():
    """Load SikuRoBERTa and move to GPU."""
    from transformers import AutoTokenizer, AutoModel

    for model_id in [MODEL_NAME, MODEL_FALLBACK]:
        try:
            print(f"  Trying model: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                print(f"  Loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("  Loaded on CPU")
            print(f"  Model: {model_id}")
            return tokenizer, model
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    raise RuntimeError("Could not load model.")


def pre_tokenize_all(texts: list[str], tokenizer) -> dict[str, np.ndarray]:
    """Tokenize all texts into fixed-length padded arrays."""
    all_ids = []
    all_masks = []

    print(f"  Pre-tokenizing {len(texts)} texts (sub-batch={TOKENIZE_BATCH})...")
    t0 = time.time()

    for i in range(0, len(texts), TOKENIZE_BATCH):
        batch_texts = texts[i : i + TOKENIZE_BATCH]
        encoded = tokenizer(
            batch_texts,
            return_tensors="np",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        all_ids.append(encoded["input_ids"].astype(np.int32))
        all_masks.append(encoded["attention_mask"].astype(np.int32))

        pct = min(i + TOKENIZE_BATCH, len(texts)) / len(texts) * 100
        elapsed = time.time() - t0
        print(f"\r  Tokenized: {pct:.0f}%  ({elapsed:.0f}s)", end="", flush=True)

    print()
    return {
        "input_ids": np.concatenate(all_ids, axis=0),
        "attention_mask": np.concatenate(all_masks, axis=0),
    }


def encode_all(tokenized: dict, model) -> np.ndarray:
    """Encode pre-tokenized texts through SikuRoBERTa on GPU."""
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    total = len(input_ids)
    dim = model.config.hidden_size
    use_cuda = torch.cuda.is_available()

    print(
        f"  Encoding {total} texts (batch={ENCODE_BATCH_SIZE}, dim={dim}, "
        f"device={'cuda' if use_cuda else 'cpu'})..."
    )
    t0 = time.time()

    all_vectors = np.empty((total, dim), dtype=np.float32)

    with torch.inference_mode():
        for i in range(0, total, ENCODE_BATCH_SIZE):
            end = min(i + ENCODE_BATCH_SIZE, total)
            batch_ids = torch.from_numpy(input_ids[i:end])
            batch_mask = torch.from_numpy(attention_mask[i:end])

            if use_cuda:
                batch_ids = batch_ids.cuda()
                batch_mask = batch_mask.cuda()

            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
            )

            hidden = outputs.last_hidden_state
            mask = batch_mask.unsqueeze(-1).expand(hidden.size()).float()

            # Mean pooling on GPU then move to CPU
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            vectors = pooled.cpu().numpy().astype(np.float32)

            # L2 normalize
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            vectors = vectors / norms

            all_vectors[i:end] = vectors

            elapsed = time.time() - t0
            rate = end / elapsed if elapsed > 0 else 0
            eta = (total - end) / rate if rate > 0 else 0
            print(
                f"\r  [{end}/{total}] {end / total * 100:.0f}%  "
                f"rate={rate:.0f}/s  ETA={eta:.0f}s",
                end="",
                flush=True,
            )

    print()
    return all_vectors


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f}GB"
        )

    print("=" * 60)
    print(f"Step 3: Text Encoding (SikuRoBERTa, {device.upper()})")
    print("=" * 60)

    # Load chunks
    print(f"\n  Loading chunks from: {CHUNKS_JSONL}")
    chunks = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    total = len(chunks)
    print(f"  Total chunks: {total}")

    # Load model
    print("\n  Loading model...")
    tokenizer, model = load_model_and_tokenizer()

    # Phase 1: Pre-tokenize
    print("\n[Phase 1] Pre-tokenization")
    texts = [c["text"] for c in chunks]
    tokenized = pre_tokenize_all(texts, tokenizer)
    print(f"  input_ids shape: {tokenized['input_ids'].shape}")

    # Phase 2: Encode
    print("\n[Phase 2] Model encoding")
    start_time = time.time()
    vectors = encode_all(tokenized, model)

    # Save
    print(f"\n  Vector shape: {vectors.shape}")
    print(f"  Saving to: {VECTORS_NPY}")
    np.save(VECTORS_NPY, vectors)

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print(
        f"\n  Done in {elapsed:.1f}s ({elapsed / 60:.1f}min), {rate:.0f} chunks/s total"
    )
    print("Step 3 complete.")


if __name__ == "__main__":
    main()
