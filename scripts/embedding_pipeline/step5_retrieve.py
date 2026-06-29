#!/usr/bin/env python3
"""Step 5: Semantic retrieval — search vector index with query groups.

Designs 5 query groups covering different aspects of kentoshi missions,
encodes queries with SikuRoBERTa, searches with numpy dot-product,
deduplicates results.
"""

import json
import time
from pathlib import Path

import numpy as np

from embedding_pipeline.config import (
    CHUNKS_JSONL,
    FAISS_INDEX,
    RETRIEVED_JSONL,
    TOP_K_PER_QUERY,
    SIMILARITY_THRESHOLD,
)

# ── Query Groups ────────────────────────────────────────────────────────────
# Each group has a theme and multiple query strings designed to capture
# different phrasings in classical Chinese/Japanese historical texts.
# Queries are based on ACTUAL vocabulary found in the source texts.
#
# Key findings from text analysis:
# - "写得持来" (not "将来"/"请来") is the primary phrase for bringing back texts
# - "入唐留学", "被宛入唐留学", "留学唐" are the actual learning patterns
# - "受具足戒", "灌頂", "求法" are actual ceremony/learning terms
# - "聘唐大使", "遣唐大使", "遣唐使准判官" are mission title terms
# - "漂着", "漂泊", "遭風" are shipwreck terms
# - In China text: "日本國遣使", "日本國朝貢", "倭國遣使入朝"
# - "齎" sometimes used for "bring" but mostly in non-kentoshi contexts
# - "帰朝", "帰来", "帰国" for returning to Japan
QUERY_GROUPS = {
    "派遣與出使": [
        # Direct mission terms from texts
        "遣唐使 遣唐大使 遣唐副使 遣唐使准判官 入唐大使 聘唐大使",
        # Ships and departure
        "遣唐使第一船 遣唐使第二船 遣唐使船 發從 出帆 解纜",
        # Mission types and roles
        "遣唐使判官 遣唐録事 執節使 送唐客使 迎入唐大使",
        # Appointment and ceremony
        "拜遣唐使 為遣唐大使 遣唐使拜朝 奉還節刀 賜遣唐使",
        # General mission language
        "遣使入唐 入唐 渡唐 赴唐 使唐 聘唐",
        # China text patterns
        "日本國遣使 遣使朝貢 日本國朝貢 倭國遣使入朝",
        # Mission frequency / context
        "遣唐使 朝貢 貢獻 貢方物 進貢",
    ],
    "文化傳入": [
        # Primary phrase from texts: "写得持来" (wrote/copied and brought back)
        "写得持来 寫得持來 持来 將來經 將來聖教 持歸",
        # Actual bring-back vocabulary
        "寫得 写得 將來 将來 持來 齎来 齎 請来 請來",
        # Cultural items actually mentioned
        "聖教要文 經論 經疏 佛像 曼荼羅 曼陀羅 曼荼罗",
        # Religious transmission
        "密教 天台宗 真言宗 法相宗 兩部大法 金剛界 胎藏界",
        # Text types brought back
        "經卷 經軌 經疏論記 目錄 目録 聖教",
        # Broader cultural transmission concepts
        "傳入 伝来 佛法東漸 傳來 将来目录 請来目录",
    ],
    "師從學習": [
        # Primary patterns from texts: "入唐留学", "受具足戒", "求法"
        "入唐留学 被宛入唐留学 留学唐 入唐學問 入唐求法",
        # Ceremony/ordination terms
        "受具足戒 受戒 灌頂 受法 受菩薩戒 傳法 付法",
        # Learning relationships
        "師從 師事 稟受 傳授 従學 從某學法",
        # Student/monk types from texts
        "留学僧 留學僧 学問僧 請益僧 入唐僧 入唐留学僧",
        # Specific learning content
        "學唯識 学法花 学天台 传真言 受兩部大法 求法",
        # General learning in Tang
        "在唐學習 於唐學法 從師學 入唐求法僧",
    ],
    "制度與技術傳入": [
        # Administrative systems actually mentioned
        "律令 官制 位階 班田 租庸調 條坊 格式",
        # Calendar systems
        "大衍暦 大衍曆 宣明暦 宣明曆 麟德暦 暦法 曆法",
        # Tang institutions adopted
        "唐制 唐禮 唐令 唐式 漢制 仿唐 模倣唐制",
        # Reform context
        "大化改新 改新 新制 引入唐制 用唐制",
        # Knowledge transfer
        "漢籍 書籍 經書 論語 孝經 文選 漢詩",
        # Medical/technical
        "醫學 医學 針灸 本草 測量 算術 陰陽道 天文",
    ],
    "路線與地點": [
        # Major Tang cities from texts
        "長安 洛陽 揚州 明州 台州 福州 蘇州 登州 楚州 越州",
        # Japan-side ports
        "博多 難波 大宰府 肥前国松浦郡 對馬嶋 薩摩",
        # Temples and mountains
        "天台山 五臺山 五台山 青龍寺 青竜寺 青龍寺 大興善寺 西明寺",
        # Shipwreck/disaster terms (from texts)
        "漂着 漂著 漂泊 遭風 覆没 遇難 船舶損壊 不得渡海",
        # Travel and route
        "入唐経路 南路 北路 渡海 發從 到泊 歸著 歸国 帰朝",
        # Arrival/departure patterns
        "到達長安 來朝長安 到着 到岸 著岸 入京 解纜 出帆",
    ],
}


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load all chunks from JSONL file."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_vectors(index_path: Path) -> np.ndarray:
    """Load vectors from binary file with metadata."""
    meta_path = str(index_path) + ".meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    vectors = np.fromfile(str(index_path), dtype=np.float32)
    return vectors.reshape(meta["n_total"], meta["dim"])


def encode_queries(queries: list[str], tokenizer, model) -> np.ndarray:
    """Encode query strings into L2-normalized vectors."""
    import torch

    inputs = tokenizer(
        queries,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs["attention_mask"]
    hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1)

    vectors = pooled.numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vectors = vectors / norms
    return vectors


def search_numpy(
    query_vectors: np.ndarray, corpus: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Search corpus vectors using dot product (cosine similarity).

    Args:
        query_vectors: shape (n_queries, dim)
        corpus: shape (n_docs, dim)
        top_k: number of top results to return

    Returns:
        scores: shape (n_queries, top_k) — cosine similarities
        indices: shape (n_queries, top_k) — corpus indices
    """
    scores = np.dot(query_vectors, corpus.T)  # (n_queries, n_docs)
    # Get top-k indices (argsort descending)
    indices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
    # Sort within top-k
    batch_idx = np.arange(len(query_vectors))[:, None]
    top_scores = scores[batch_idx, indices]
    sort_order = np.argsort(-top_scores, axis=1)
    indices = indices[batch_idx, sort_order]
    top_scores = top_scores[batch_idx, sort_order]
    return top_scores, indices


def main():
    print("=" * 60)
    print("Step 5: Semantic Retrieval (numpy)")
    print("=" * 60)

    # Load chunks
    print(f"\n  Loading chunks from: {CHUNKS_JSONL}")
    chunks = load_chunks(CHUNKS_JSONL)
    print(f"  Total chunks: {len(chunks)}")

    # Load vectors
    print(f"\n  Loading vectors from: {FAISS_INDEX}")
    vectors = load_vectors(FAISS_INDEX)
    print(f"  Vectors: {vectors.shape}")

    # Load model for encoding queries
    print("\n  Loading SikuRoBERTa for query encoding...")
    from transformers import AutoTokenizer, AutoModel

    from embedding_pipeline.config import MODEL_NAME, MODEL_FALLBACK

    for model_id in [MODEL_NAME, MODEL_FALLBACK]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            model.eval()
            print(f"  Loaded: {model_id}")
            break
        except Exception as e:
            print(f"  Failed {model_id}: {e}")
            continue
    else:
        raise RuntimeError("Could not load model for query encoding")

    # Search with each query group
    all_results: dict[int, dict] = {}
    total_queries = sum(len(qs) for qs in QUERY_GROUPS.values())

    print(
        f"\n  Searching with {len(QUERY_GROUPS)} query groups "
        f"({total_queries} total queries)..."
    )
    t0 = time.time()

    for group_name, queries in QUERY_GROUPS.items():
        print(f"\n  [{group_name}] Encoding {len(queries)} queries...")
        q_vectors = encode_queries(queries, tokenizer, model)

        # Search with numpy dot product
        scores, indices = search_numpy(q_vectors, vectors, TOP_K_PER_QUERY)

        for qi, query in enumerate(queries):
            for rank in range(TOP_K_PER_QUERY):
                score = float(scores[qi][rank])
                chunk_id = int(indices[qi][rank])

                if score < SIMILARITY_THRESHOLD:
                    continue

                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        "chunk_id": chunk_id,
                        "score": score,
                        "query_group": group_name,
                        "query": query,
                        "hit_groups": [group_name],
                    }
                else:
                    existing = all_results[chunk_id]
                    if group_name not in existing.setdefault(
                        "hit_groups", [existing["query_group"]]
                    ):
                        existing["hit_groups"].append(group_name)
                    if score > existing["score"]:
                        existing["score"] = score
                        existing["query_group"] = group_name
                        existing["query"] = query

    # Build output records
    print("\n  Building output records...")
    retrieved = []
    for chunk_id, info in sorted(all_results.items()):
        chunk = chunks[chunk_id]
        record = {
            "chunk_id": chunk_id,
            "score": info["score"],
            "query_group": info["query_group"],
            "hit_groups": info.get("hit_groups", [info["query_group"]]),
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "section": chunk["section"],
            "source_offset": chunk["source_offset"],
        }
        retrieved.append(record)

    # ── Tang keyword post-filter ──────────────────────────────────────────
    from embedding_pipeline.config import TANG_KEYWORDS

    before = len(retrieved)
    retrieved = [r for r in retrieved if any(kw in r["text"] for kw in TANG_KEYWORDS)]
    print(
        f"  Tang filter: {before} → {len(retrieved)} "
        f"({len(retrieved) / before * 100:.1f}%)"
    )

    # Sort by score descending
    retrieved.sort(key=lambda r: r["score"], reverse=True)

    # Write output
    print(f"  Writing {len(retrieved)} results to: {RETRIEVED_JSONL}")
    with open(RETRIEVED_JSONL, "w", encoding="utf-8") as f:
        for rec in retrieved:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    elapsed = time.time() - t0
    print(f"\n  {'=' * 40}")
    print(f"  Total retrieved chunks: {len(retrieved)}")
    if retrieved:
        print(
            f"  Score range: [{retrieved[-1]['score']:.4f}, "
            f"{retrieved[0]['score']:.4f}]"
        )

    # Group statistics
    group_counts = {}
    for rec in retrieved:
        for g in rec["hit_groups"]:
            group_counts[g] = group_counts.get(g, 0) + 1
    print("\n  Chunks per query group:")
    for g, c in sorted(group_counts.items(), key=lambda x: -x[1]):
        print(f"    {g}: {c}")

    # Multi-hit statistics
    if retrieved:
        multi_hit = sum(1 for r in retrieved if len(r["hit_groups"]) > 1)
        print(
            f"\n  Chunks hit by multiple groups: {multi_hit} "
            f"({multi_hit / len(retrieved) * 100:.1f}%)"
        )

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("Step 5 complete.")


if __name__ == "__main__":
    main()
