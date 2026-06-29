#!/usr/bin/env python3
"""
Consensus Person Ranking: Combine centrality metrics, check bias, cross-validate.
Pure stdlib implementation - no pandas/numpy/scipy needed.
"""

import csv
import math
from collections import OrderedDict


# ============================================================
# Utility functions
# ============================================================
def read_csv(filename):
    """Read CSV and return list of dicts."""
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def stddev(vals):
    m = mean(vals)
    return (
        math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
    )


def pearson_corr(xs, ys):
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    sx = stddev(xs)
    sy = stddev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n * sx * sy)


def spearman_corr(xs, ys):
    """Spearman rank correlation."""
    n = len(xs)
    if n < 3:
        return 0.0
    # Rank xs
    x_ranked = sorted(enumerate(xs), key=lambda t: t[1])
    x_ranks = [0] * n
    for rank, (idx, _) in enumerate(x_ranked, 1):
        x_ranks[idx] = rank
    # Rank ys
    y_ranked = sorted(enumerate(ys), key=lambda t: t[1])
    y_ranks = [0] * n
    for rank, (idx, _) in enumerate(y_ranked, 1):
        y_ranks[idx] = rank
    return pearson_corr(x_ranks, y_ranks)


def min_max_norm(vals):
    """Min-max normalize to [0,1]."""
    mn = min(vals)
    mx = max(vals)
    if mx == mn:
        return [0.0] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]


def quantile(vals, q):
    """Return the q-th quantile (0..1)."""
    sorted_vals = sorted(vals)
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ============================================================
# 1. LOAD ALL-CENTRALITY DATA
# ============================================================
rows = read_csv("analysis/centrality_all.csv")
print("=== centrality_all.csv ===")
print(f"Rows: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")

# Parse numeric fields
for r in rows:
    r["weighted_degree"] = float(r["weighted_degree"])
    r["betweenness"] = float(r["betweenness"])
    r["eigenvector"] = float(r["eigenvector"])
    r["pagerank"] = float(r["pagerank"])
    r["source_count"] = int(r["source_count"])

# Check duplicates
person_ids = [r["person_id"] for r in rows]
dup_ids = set()
seen_ids = set()
for pid in person_ids:
    if pid in seen_ids:
        dup_ids.add(pid)
    seen_ids.add(pid)
if dup_ids:
    print(f"\nDuplicate person_ids: {dup_ids}")
    for pid in dup_ids:
        matches = [r for r in rows if r["person_id"] == pid]
        for m in matches:
            print(
                f"  {m['person_id']} {m['person_name']} deg={m['weighted_degree']} src={m['source_count']}"
            )

person_names = [r["person_name"] for r in rows]
dup_names_set = set()
seen_names = set()
for nm in person_names:
    if nm in seen_names:
        dup_names_set.add(nm)
    seen_names.add(nm)
if dup_names_set:
    print(f"\nDuplicate person_names: {dup_names_set}")

# ============================================================
# 2. NORMALIZE METRICS (Min-Max scaling)
# ============================================================
metrics = ["weighted_degree", "betweenness", "eigenvector", "pagerank"]
for col in metrics:
    vals = [r[col] for r in rows]
    norm_vals = min_max_norm(vals)
    for i, r in enumerate(rows):
        r[f"{col}_norm"] = norm_vals[i]

# Composite score = mean of normalized metrics
for r in rows:
    r["composite_raw"] = mean([r[f"{m}_norm"] for m in metrics])

# Normalize composite to 0-100 for readability
comp_raws = [r["composite_raw"] for r in rows]
comp_norm = min_max_norm(comp_raws)
for i, r in enumerate(rows):
    r["composite_score"] = comp_norm[i] * 100

# ============================================================
# 3. RANK ALL PERSONS
# ============================================================
rows_sorted = sorted(rows, key=lambda r: r["composite_score"], reverse=True)
for i, r in enumerate(rows_sorted):
    r["rank"] = i + 1

print("\n\n=== TOP 20 PERSONS BY CONSENSUS COMPOSITE SCORE ===")
print(
    f"{'Rank':<5} {'ID':<8} {'Name':<32} {'Composite':>8} {'Degree':>10} {'Between':>10} {'Eigenvec':>10} {'PageRank':>10} {'Src':>4}"
)
print("-" * 105)
for r in rows_sorted[:20]:
    print(
        f"#{r['rank']:<4d} {r['person_id']:<8} {r['person_name']:<32} {r['composite_score']:>7.1f} {r['weighted_degree']:>9.3f} {r['betweenness']:>9.4f} {r['eigenvector']:>9.6f} {r['pagerank']:>9.4f} {r['source_count']:>4d}"
    )

# ============================================================
# 4. CHECK HISTORICAL FIGURES
# ============================================================
print("\n\n=== HISTORICAL FIGURE CHECK ===")

# Search for 吉備真備 and 阿倍仲麻呂
found_kibi = None
found_abe = None
for r in rows:
    nm = r["person_name"]
    if "吉備" in nm:
        print(f"  Found 吉備-related: {r['person_id']} {nm}")
        if found_kibi is None:
            found_kibi = r["person_id"]
    if "仲麻呂" in nm:
        print(f"  Found 仲麻呂-related: {r['person_id']} {nm}")
        if found_abe is None:
            found_abe = r["person_id"]

targets = OrderedDict(
    [
        ("空海 (Kukai)", "P143"),
        ("最澄 (Saicho)", "P118"),
        ("円仁 (Ennin)", "P034"),
        ("吉備真備 (Kibi no Makibi)", found_kibi),
        ("阿倍仲麻呂 (Abe no Nakamaro)", found_abe),
    ]
)

id_to_row = {r["person_id"]: r for r in rows_sorted}
for fig_name, pid in targets.items():
    if pid is None:
        print(f"  {fig_name}: NOT FOUND in centrality_all.csv")
    elif pid in id_to_row:
        r = id_to_row[pid]
        print(
            f"  {fig_name} ({r['person_id']}): rank=#{r['rank']}, composite={r['composite_score']:.1f}, deg={r['weighted_degree']:.3f}, btw={r['betweenness']:.4f}, eig={r['eigenvector']:.6f}, pr={r['pagerank']:.4f}, src={r['source_count']}"
        )
    else:
        print(f"  {fig_name} ({pid}): NOT FOUND in centrality_all.csv")

# ============================================================
# 5. CORRELATION WITH source_count
# ============================================================
print("\n\n=== CORRELATION: composite_score vs source_count ===")
composites = [r["composite_score"] for r in rows]
src_counts = [r["source_count"] for r in rows]

r_pearson = pearson_corr(composites, src_counts)
r_spearman = spearman_corr(composites, src_counts)
print(f"  Pearson r = {r_pearson:.4f}")
print(f"  Spearman \u03c1 = {r_spearman:.4f}")

# Per-metric correlations
print("\n  Per-metric Pearson correlation with source_count:")
for col in metrics:
    vals = [r[col] for r in rows]
    r_p = pearson_corr(vals, src_counts)
    print(f"    {col}: r={r_p:.4f}")

# ============================================================
# 6. BIAS CHECK
# ============================================================
print("\n\n=== BIAS ANALYSIS ===")
q75 = quantile(src_counts, 0.75)
q25 = quantile(src_counts, 0.25)
high_src = [r for r in rows if r["source_count"] >= q75]
low_src = [r for r in rows if r["source_count"] <= q25]
print(
    f"  Top quartile by source_count (>= {q75}, n={len(high_src)}): mean composite = {mean([r['composite_score'] for r in high_src]):.2f}"
)
print(
    f"  Bottom quartile by source_count (<= {q25}, n={len(low_src)}): mean composite = {mean([r['composite_score'] for r in low_src]):.2f}"
)

# Persons with high composite but low source_count
print("\n  Persons with composite >= 50 but source_count <= 5:")
for r in rows_sorted:
    if r["composite_score"] >= 50 and r["source_count"] <= 5:
        print(
            f"    {r['person_name']:30s} composite={r['composite_score']:.1f} source_count={r['source_count']}"
        )

# Persons with low composite but high source_count
print("\n  Persons with composite < 30 but source_count >= 10:")
for r in rows_sorted:
    if r["composite_score"] < 30 and r["source_count"] >= 10:
        print(
            f"    {r['person_name']:30s} composite={r['composite_score']:.1f} source_count={r['source_count']}"
        )

# ============================================================
# 7. CROSS-VALIDATE WITH PHASE4-V3 DATA
# ============================================================
print("\n\n=== CROSS-VALIDATION WITH PHASE4-V3 ===")
p4_rows = read_csv("analysis/centrality_Phase4_v3.csv")
print(f"Phase4-v3 rows: {len(p4_rows)}")
print(f"Phase4-v3 columns: {list(p4_rows[0].keys())}")

# Parse numeric fields
for r in p4_rows:
    r["weighted_degree"] = float(r["weighted_degree"])
    r["betweenness"] = float(r["betweenness"])
    r["pagerank"] = float(r["pagerank"])

# Compute composite for Phase4 (no eigenvector in Phase4)
p4_metrics = ["weighted_degree", "betweenness", "pagerank"]
for col in p4_metrics:
    vals = [r[col] for r in p4_rows]
    norm_vals = min_max_norm(vals)
    for i, r in enumerate(p4_rows):
        r[f"{col}_norm"] = norm_vals[i]

for r in p4_rows:
    r["composite_raw"] = mean([r[f"{m}_norm"] for m in p4_metrics])

comp_raws_p4 = [r["composite_raw"] for r in p4_rows]
comp_norm_p4 = min_max_norm(comp_raws_p4)
for i, r in enumerate(p4_rows):
    r["composite_score"] = comp_norm_p4[i] * 100

p4_sorted = sorted(p4_rows, key=lambda r: r["composite_score"], reverse=True)
for i, r in enumerate(p4_sorted):
    r["rank"] = i + 1

print("\n  Top-20 in Phase4-v3 composite ranking:")
print(
    f"  {'Rank':<5} {'ID':<8} {'Name':<32} {'Composite':>8} {'Degree':>10} {'Between':>10} {'PageRank':>10}"
)
print("  " + "-" * 85)
for r in p4_sorted[:20]:
    print(
        f"  #{r['rank']:<4d} {r['person_id']:<8} {r['person_name']:<32} {r['composite_score']:>7.1f} {r['weighted_degree']:>9.3f} {r['betweenness']:>9.4f} {r['pagerank']:>9.4f}"
    )

# Historical figures in Phase4
print("\n  Historical figures in Phase4-v3:")
p4_id_to_row = {r["person_id"]: r for r in p4_sorted}
for fig_name, pid in targets.items():
    if pid and pid in p4_id_to_row:
        r = p4_id_to_row[pid]
        print(
            f"    {fig_name} ({r['person_id']}): rank=#{r['rank']}, composite={r['composite_score']:.1f}, deg={r['weighted_degree']:.3f}, btw={r['betweenness']:.4f}, pr={r['pagerank']:.4f}"
        )
    elif pid:
        print(f"    {fig_name} ({pid}): NOT FOUND in Phase4-v3")
    else:
        print(f"    {fig_name}: NOT IN EITHER DATASET")

# ============================================================
# 8. COMPARE RANKINGS: all vs Phase4
# ============================================================
print("\n\n=== RANK COMPARISON: ALL vs PHASE4 ===")
common_ids = set(id_to_row.keys()) & set(p4_id_to_row.keys())
print(f"  Common persons: {len(common_ids)}")

rank_comparison = []
for pid in common_ids:
    r_all = id_to_row[pid]["rank"]
    r_p4 = p4_id_to_row[pid]["rank"]
    nm = id_to_row[pid]["person_name"]
    rank_comparison.append(
        {
            "person_id": pid,
            "person_name": nm,
            "rank_all": r_all,
            "rank_p4": r_p4,
            "rank_diff": r_p4 - r_all,
            "composite_all": id_to_row[pid]["composite_score"],
            "composite_p4": p4_id_to_row[pid]["composite_score"],
        }
    )

r_s = spearman_corr(
    [rc["rank_all"] for rc in rank_comparison],
    [rc["rank_p4"] for rc in rank_comparison],
)
print(f"  Spearman rank correlation (all vs Phase4): \u03c1 = {r_s:.4f}")

# Big movers
print("\n  Big movers (|diff| >= 10):")
rank_comparison.sort(key=lambda x: abs(x["rank_diff"]), reverse=True)
for rc in rank_comparison:
    if abs(rc["rank_diff"]) >= 10:
        direction = "UP in P4" if rc["rank_diff"] < 0 else "DOWN in P4"
        print(
            f"    {rc['person_name']:30s}  all=#{rc['rank_all']:2d}  p4=#{rc['rank_p4']:2d}  diff={rc['rank_diff']:+3d}  ({direction})"
        )

# ============================================================
# 9. ADDITIONAL ANALYSIS: ENTITY TYPE CHECK
# ============================================================
print("\n\n=== ENTITY TYPE CHECK ===")
# Load nodes.csv to check types
nodes = read_csv("output/nodes.csv")
node_types = {}
for n in nodes:
    node_types[n["id"]] = n.get("type", "Unknown")

# Check if any of the top-ranked are not actually persons
print("  Top-20 entity types (from nodes.csv):")
for r in rows_sorted[:20]:
    ntype = node_types.get(r["person_id"], "NOT_FOUND")
    print(f"    {r['person_id']} {r['person_name']:<30s} type={ntype}")

# ============================================================
# 10. SAVE RESULTS
# ============================================================
print("\n\n=== SAVING RESULTS ===")

# Save consensus ranking
with open("analysis/consensus_ranking_all.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rank",
            "person_id",
            "person_name",
            "weighted_degree",
            "betweenness",
            "eigenvector",
            "pagerank",
            "composite_score",
            "source_count",
        ],
    )
    writer.writeheader()
    for r in rows_sorted:
        writer.writerow({k: r[k] for k in writer.fieldnames})

# Save Phase4 consensus ranking
with open(
    "analysis/consensus_ranking_phase4.csv", "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rank",
            "person_id",
            "person_name",
            "weighted_degree",
            "betweenness",
            "pagerank",
            "composite_score",
        ],
    )
    writer.writeheader()
    for r in p4_sorted:
        writer.writerow({k: r[k] for k in writer.fieldnames})

# Save comparison
with open(
    "analysis/ranking_comparison_all_vs_phase4.csv", "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "person_id",
            "person_name",
            "rank_all",
            "rank_p4",
            "rank_diff",
            "composite_all",
            "composite_p4",
        ],
    )
    writer.writeheader()
    for rc in sorted(rank_comparison, key=lambda x: abs(x["rank_diff"]), reverse=True):
        writer.writerow(rc)

print("Saved:")
print("  - analysis/consensus_ranking_all.csv")
print("  - analysis/consensus_ranking_phase4.csv")
print("  - analysis/ranking_comparison_all_vs_phase4.csv")
print("\nDone.")
