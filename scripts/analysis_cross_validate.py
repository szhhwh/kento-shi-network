#!/usr/bin/env python3
"""Redo cross-validation by NAME (not ID), compute proper rank comparison."""

import csv
import math


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def min_max_norm(vals):
    mn, mx = min(vals), max(vals)
    return [(v - mn) / (mx - mn) if mx > mn else 0.0 for v in vals]


def spearman_corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0

    def rank(vals):
        ranked = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0] * n
        for r, (idx, _) in enumerate(ranked, 1):
            ranks[idx] = r
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    sx = math.sqrt(sum((x - mx) ** 2 for x in rx) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ry) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / (n * sx * sy)


# Read all data
def read_csv(fn):
    with open(fn, "r") as f:
        return [r for r in csv.DictReader(f)]


all_rows = read_csv("analysis/centrality_all.csv")
p4_rows = read_csv("analysis/centrality_Phase4_v3.csv")

# Index by name
all_by_name = {}
for r in all_rows:
    all_by_name[r["person_name"]] = r
p4_by_name = {}
for r in p4_rows:
    p4_by_name[r["person_name"]] = r

# Find common names
common_names = set(all_by_name.keys()) & set(p4_by_name.keys())
print(f"Common names between all and Phase4: {len(common_names)}")
print(f"Names only in all: {len(set(all_by_name.keys()) - set(p4_by_name.keys()))}")
print(f"Names only in Phase4: {len(set(p4_by_name.keys()) - set(all_by_name.keys()))}")

# Recompute rankings within common set
# For all data
all_common = []
for nm in common_names:
    r = all_by_name[nm]
    all_common.append(
        {
            "name": nm,
            "deg": float(r["weighted_degree"]),
            "btw": float(r["betweenness"]),
            "eig": float(r["eigenvector"]),
            "pr": float(r["pagerank"]),
            "src": int(r["source_count"]),
            "id_all": r["person_id"],
        }
    )

# For Phase4 data
p4_common = []
for nm in common_names:
    r = p4_by_name[nm]
    p4_common.append(
        {
            "name": nm,
            "deg": float(r["weighted_degree"]),
            "btw": float(r["betweenness"]),
            "pr": float(r["pagerank"]),
            "id_p4": r["person_id"],
        }
    )

# Normalize and compute composites within common subset
# All: deg, btw, eig, pr
all_metrics = ["deg", "btw", "eig", "pr"]
for m in all_metrics:
    vals = [r[m] for r in all_common]
    norms = min_max_norm(vals)
    for i, r in enumerate(all_common):
        r[f"{m}_norm"] = norms[i]

for r in all_common:
    r["composite"] = mean([r[f"{m}_norm"] for m in all_metrics])

# Rank all common
all_common.sort(key=lambda r: r["composite"], reverse=True)
for i, r in enumerate(all_common):
    r["rank"] = i + 1

# Phase4: deg, btw, pr
p4_metrics = ["deg", "btw", "pr"]
for m in p4_metrics:
    vals = [r[m] for r in p4_common]
    norms = min_max_norm(vals)
    for i, r in enumerate(p4_common):
        r[f"{m}_norm"] = norms[i]

for r in p4_common:
    r["composite"] = mean([r[f"{m}_norm"] for m in p4_metrics])

p4_common.sort(key=lambda r: r["composite"], reverse=True)
for i, r in enumerate(p4_common):
    r["rank"] = i + 1

# Compare
name_to_rank_all = {r["name"]: r["rank"] for r in all_common}
name_to_rank_p4 = {r["name"]: r["rank"] for r in p4_common}

print("\n=== RANK COMPARISON BY NAME ===")
print(f"{'Name':<30s} {'All#':>5} {'P4#':>5} {'Diff':>6} {'AllComp':>8} {'P4Comp':>8}")
print("-" * 70)
comparisons = []
for nm in common_names:
    ra = name_to_rank_all.get(nm, 999)
    rp = name_to_rank_p4.get(nm, 999)
    ca = next(r["composite"] for r in all_common if r["name"] == nm)
    cp = next(r["composite"] for r in p4_common if r["name"] == nm)
    diff = rp - ra
    comparisons.append((nm, ra, rp, diff, ca, cp))
    if abs(diff) >= 5:
        print(f"{nm:<30s} {ra:>5d} {rp:>5d} {diff:>+6d} {ca:>8.3f} {cp:>8.3f}")

# Spearman correlation
ranks_all = [name_to_rank_all[nm] for nm in common_names]
ranks_p4 = [name_to_rank_p4[nm] for nm in common_names]
rho = spearman_corr(ranks_all, ranks_p4)
print(f"\nSpearman rank correlation (all vs Phase4, by name): \u03c1 = {rho:.4f}")

# Top 10 in each
print("\n=== Top-10 by NAME in ALL data (common subset) ===")
for r in all_common[:10]:
    print(
        f"  #{r['rank']} {r['name']:<30s} comp={r['composite']:.3f} deg={r['deg']:.2f} btw={r['btw']:.4f} eig={r['eig']:.6f} pr={r['pr']:.4f}"
    )

print("\n=== Top-10 by NAME in Phase4 data (common subset) ===")
for r in p4_common[:10]:
    print(
        f"  #{r['rank']} {r['name']:<30s} comp={r['composite']:.3f} deg={r['deg']:.2f} btw={r['btw']:.4f} pr={r['pr']:.4f}"
    )

# Historical figures in this comparison
print("\n=== Historical figures in common subset ===")
for nm in ["空海", "最澄", "円仁"]:
    ra = name_to_rank_all.get(nm, "N/A")
    rp = name_to_rank_p4.get(nm, "N/A")
    print(f"  {nm}: all=#{ra}, p4=#{rp}")

# Print names only in Phase4 but NOT in all
p4_only = set(p4_by_name.keys()) - set(all_by_name.keys())
print(f"\n=== Names only in Phase4 (not in all centrality): {len(p4_only)} ===")
# Show ones with non-zero metrics
p4_only_significant = []
for nm in p4_only:
    r = p4_by_name[nm]
    deg = float(r["weighted_degree"])
    btw = float(r["betweenness"])
    pr = float(r["pagerank"])
    if deg > 0 or btw > 0:
        p4_only_significant.append((nm, deg, btw, pr))
p4_only_significant.sort(key=lambda x: x[1], reverse=True)
for nm, deg, btw, pr in p4_only_significant[:20]:
    print(f"  {nm:<30s} deg={deg:.3f} btw={btw:.4f} pr={pr:.4f}")

# Names only in all but NOT in Phase4
all_only = set(all_by_name.keys()) - set(p4_by_name.keys())
print(f"\n=== Names only in ALL centrality (not in Phase4): {len(all_only)} ===")
all_only_sig = []
for nm in all_only:
    r = all_by_name[nm]
    deg = float(r["weighted_degree"])
    btw = float(r["betweenness"])
    eig = float(r["eigenvector"])
    pr = float(r["pagerank"])
    if deg > 0 or btw > 0:
        all_only_sig.append((nm, deg, btw, eig, pr))
all_only_sig.sort(key=lambda x: x[1], reverse=True)
for nm, deg, btw, eig, pr in all_only_sig[:20]:
    print(f"  {nm:<30s} deg={deg:.3f} btw={btw:.4f} eig={eig:.6f} pr={pr:.4f}")

print("\nDone.")
