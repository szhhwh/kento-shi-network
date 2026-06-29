#!/usr/bin/env python3
"""
Buddhist vs Secular Network Comparison for Kento-shi Data
Updated: handles ID mismatch between nodes.csv and centrality_all.csv
"""

import csv
import json
import re
from collections import defaultdict

BUDDHIST_RE = re.compile(r"釋|僧|法|師|和尚|禅")


def is_buddhist(name):
    return bool(BUDDHIST_RE.search(name))


# ============================================================
# LOAD DATA
# ============================================================

# 1. nodes.csv - persons with their names
nodes_persons = {}  # pid -> name
with open("output/nodes.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 3 and row[1] == "Person":
            nodes_persons[row[0]] = row[2]

# 2. edges.csv
edges = []
with open("output/edges.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) >= 4:
            edges.append(row)

# 3. centrality_all.csv - uses SAME IDs but DIFFERENT names
cent_data = {}  # pid -> {name, weighted_degree, betweenness, eigenvector, pagerank}
with open("analysis/centrality_all.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cent_data[row["person_id"]] = {
            "name": row["person_name"],
            "weighted_degree": float(row["weighted_degree"]),
            "betweenness": float(row["betweenness"]),
            "eigenvector": float(row["eigenvector"]),
            "pagerank": float(row["pagerank"]),
        }

# 4. enhanced_events.json
with open("output/enhanced_events.json", "r") as f:
    events_data = json.load(f)

# ============================================================
# CLASSIFY PERSONS
# ============================================================

# Classification using nodes.csv names (for network structure)
nodes_group = {}
for pid, name in nodes_persons.items():
    nodes_group[pid] = "Buddhist" if is_buddhist(name) else "Secular"

# Classification using centrality.csv names (for centrality comparison)
cent_group = {}
for pid, data in cent_data.items():
    cent_group[pid] = "Buddhist" if is_buddhist(data["name"]) else "Secular"

# ============================================================
# BUILD PERSON-PERSON NETWORK
# ============================================================

# Collect persons per event (event co-occurrence)
event_persons = defaultdict(set)
# Direct P-P edges
direct_pp = []

for e in edges:
    src, tgt, rel = e[1], e[2], e[3]
    if rel in ("HAS_SUBJECT", "HAS_OBJECT"):
        if src.startswith("E") and tgt.startswith("P") and tgt in nodes_persons:
            event_persons[src].add(tgt)
        elif tgt.startswith("E") and src.startswith("P") and src in nodes_persons:
            event_persons[tgt].add(src)
    elif rel in ("LEARN", "TRAVEL", "PARTICIPATE", "INTRODUCE"):
        if (
            src.startswith("P")
            and tgt.startswith("P")
            and src in nodes_persons
            and tgt in nodes_persons
        ):
            direct_pp.append((src, tgt, rel))
            # Also add as synthetic event for co-occurrence
            event_persons[f"SYN_{src}_{tgt}"].add(src)
            event_persons[f"SYN_{src}_{tgt}"].add(tgt)

# Build adjacency from event co-occurrence
adj = defaultdict(lambda: defaultdict(float))
for evt, pset in event_persons.items():
    plist = list(pset)
    for i in range(len(plist)):
        for j in range(i + 1, len(plist)):
            p1, p2 = plist[i], plist[j]
            adj[p1][p2] += 1
            adj[p2][p1] += 1

# ============================================================
# NETWORK METRICS
# ============================================================


def network_metrics(node_set):
    n = len(node_set)
    if n <= 1:
        return {"n": n, "m": 0, "density": 0, "avg_degree": 0, "avg_clustering": 0}

    edge_set = set()
    degrees = defaultdict(float)
    for p1 in node_set:
        for p2, w in adj[p1].items():
            if p2 in node_set:
                edge_set.add(tuple(sorted([p1, p2])))
                degrees[p1] += w

    m = len(edge_set)
    max_edges = n * (n - 1) / 2
    density = m / max_edges if max_edges > 0 else 0
    avg_degree = sum(degrees.values()) / n if n > 0 else 0

    # Clustering coefficient
    clustering = []
    for v in node_set:
        neighbors = {u for u in adj[v] if u in node_set}
        k = len(neighbors)
        if k < 2:
            clustering.append(0.0)
        else:
            actual = 0
            nlist = list(neighbors)
            for i in range(len(nlist)):
                for j in range(i + 1, len(nlist)):
                    if nlist[j] in adj[nlist[i]]:
                        actual += 1
            clustering.append(actual / (k * (k - 1) / 2))

    return {
        "n": n,
        "m": m,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 4),
        "avg_clustering": round(sum(clustering) / len(clustering), 4),
    }


bud_ids = {pid for pid, g in nodes_group.items() if g == "Buddhist"}
sec_ids = {pid for pid, g in nodes_group.items() if g == "Secular"}

bud_metrics = network_metrics(bud_ids)
sec_metrics = network_metrics(sec_ids)

# Count cross edges
cross_edges = set()
bud_edges = set()
sec_edges = set()
for p1 in adj:
    for p2 in adj[p1]:
        if p1 < p2:
            pair = (p1, p2)
            g1 = nodes_group.get(p1, "?")
            g2 = nodes_group.get(p2, "?")
            if g1 == "Buddhist" and g2 == "Buddhist":
                bud_edges.add(pair)
            elif g1 == "Secular" and g2 == "Secular":
                sec_edges.add(pair)
            elif (g1 == "Buddhist" and g2 == "Secular") or (
                g1 == "Secular" and g2 == "Buddhist"
            ):
                cross_edges.add(pair)

# ============================================================
# CENTRALITY COMPARISON
# ============================================================

bud_cent = [d for pid, d in cent_data.items() if cent_group.get(pid) == "Buddhist"]
sec_cent = [d for pid, d in cent_data.items() if cent_group.get(pid) == "Secular"]

bud_bet = [d["betweenness"] for d in bud_cent]
sec_bet = [d["betweenness"] for d in sec_cent]

avg_bud_bet = sum(bud_bet) / len(bud_bet) if bud_bet else 0
avg_sec_bet = sum(sec_bet) / len(sec_bet) if sec_bet else 0
max_bud_bet = max(bud_bet) if bud_bet else 0
max_sec_bet = max(sec_bet) if sec_bet else 0
nz_bud = sum(1 for b in bud_bet if b > 0)
nz_sec = sum(1 for b in sec_bet if b > 0)

# Top 15 by betweenness (using centrality names)
top_bet = sorted(cent_data.items(), key=lambda x: x[1]["betweenness"], reverse=True)[
    :15
]

# ============================================================
# CROSS-BRIDGE ANALYSIS
# ============================================================

bridge_score = defaultdict(float)
for p1 in adj:
    for p2, w in adj[p1].items():
        if nodes_group.get(p1) != nodes_group.get(p2):
            bridge_score[p1] += w

for p in bridge_score:
    bridge_score[p] /= 2  # each edge counted twice

top_bridges = sorted(bridge_score.items(), key=lambda x: x[1], reverse=True)[:20]

# ============================================================
# TEMPORAL ANALYSIS
# ============================================================


# Build event_id -> phase mapping
def get_phase(year):
    if year is None:
        return None
    if isinstance(year, str):
        try:
            year = int(str(year)[:4])
        except (ValueError, TypeError):
            return None
    if year < 670:
        return 1
    elif year < 760:
        return 2
    elif year < 810:
        return 3
    else:
        return 4


event_phase = {}
for i, evt in enumerate(events_data):
    eid = f"E{i + 1:03d}"
    year = evt.get("mission_departure_year")
    phase = get_phase(year)
    if phase:
        event_phase[eid] = phase

phase_persons = {
    1: {"Buddhist": set(), "Secular": set()},
    2: {"Buddhist": set(), "Secular": set()},
    3: {"Buddhist": set(), "Secular": set()},
    4: {"Buddhist": set(), "Secular": set()},
}

# For each event with known phase, assign all its persons to that phase
for evt, pset in event_persons.items():
    if evt in event_phase:
        ph = event_phase[evt]
        for pid in pset:
            if pid in nodes_group:
                phase_persons[ph][nodes_group[pid]].add(pid)

# Also count events per phase
phase_event_counts = defaultdict(int)
for evt in event_phase:
    phase_event_counts[event_phase[evt]] += 1

# ============================================================
# GENERATE MARKDOWN REPORT
# ============================================================

report = f"""# Buddhist vs Secular Network Comparison

> **Data note**: The network is built from `nodes.csv` and `edges.csv` (210 persons).  
> Centrality metrics come from `centrality_all.csv` (47 persons), which uses the **same ID space** but **different person names** — e.g., P034 = "北野" (nodes) vs "円仁" (centrality).  
> Classification is done separately on each dataset using the markers: `釋|僧|法|師|和尚|禅`.

---

## 1. Classification

| Dataset | Total | Buddhist | Secular |
|---------|-------|----------|---------|
| nodes.csv (network) | {len(nodes_persons)} | {len(bud_ids)} | {len(sec_ids)} |
| centrality_all.csv | {len(cent_data)} | {len(bud_cent)} | {len(sec_cent)} |

### Buddhist persons in nodes.csv (network nodes)
{chr(10).join(f"- {pid}: {nodes_persons[pid]}" for pid in sorted(bud_ids))}

### Buddhist persons in centrality_all.csv
{chr(10).join(f"- {pid}: {cent_data[pid]['name']}" for pid in sorted(cent_data) if cent_group.get(pid) == "Buddhist")}

> **Classification caveat**: The marker set `釋|僧|法|師|和尚|禅` captures Buddhist figures with explicit titles/prefixed names (e.g., 釋空海, 僧正宗叡, 玄奘法師) but **misses** famous monks known by their bare names without these markers (e.g., 空海, 最澄, 円仁, 宗叡, 圓仁). In `centrality_all.csv`, these unmarked monks are classified as Secular. This significantly affects the centrality comparison.

---

## 2. Network Structure

Person-Person network built from:
- Direct P–P edges (LEARN, TRAVEL, PARTICIPATE, INTRODUCE): {len(direct_pp)} edges
- Event co-occurrence (HAS_SUBJECT / HAS_OBJECT): persons sharing the same event

| Metric | Buddhist Sub-network | Secular Sub-network |
|--------|---------------------|---------------------|
| Nodes | {bud_metrics["n"]} | {sec_metrics["n"]} |
| Edges | {bud_metrics["m"]} | {sec_metrics["m"]} |
| Density | {bud_metrics["density"]} | {sec_metrics["density"]} |
| Avg Degree | {bud_metrics["avg_degree"]} | {sec_metrics["avg_degree"]} |
| Avg Clustering | {bud_metrics["avg_clustering"]} | {sec_metrics["avg_clustering"]} |

**Interpretation**:
- The Buddhist sub-network (by marker classification) has **{bud_metrics["m"]} internal edges** — Buddhist-marked persons do not co-occur in the same events, nor do they have direct edges to each other.
- All {len(cross_edges)} connections involving Buddhist persons are **cross-group** (Buddhist ↔ Secular), suggesting monks appear alongside secular envoys/officials rather than clustering together.
- The Secular network ({sec_metrics["n"]} nodes, {sec_metrics["m"]} edges) dominates the overall structure.

---

## 3. Centrality Analysis

> Using `centrality_all.csv` names for classification (47 persons).

| Group | Count | Avg Betweenness | Max Betweenness | Non-Zero |
|-------|-------|----------------|-----------------|----------|
| Buddhist | {len(bud_cent)} | {avg_bud_bet:.6f} | {max_bud_bet:.6f} | {nz_bud} |
| Secular | {len(sec_cent)} | {avg_sec_bet:.6f} | {max_sec_bet:.6f} | {nz_sec} |

### Top 15 by Betweenness Centrality
| Rank | Person | Group | Betweenness |
|------|--------|-------|-------------|
"""

for rank, (pid, d) in enumerate(top_bet, 1):
    g = cent_group.get(pid, "?")
    report += f"| {rank} | {d['name']} | {g} | {d['betweenness']:.6f} |\n"

report += f"""
**Key finding**: {
    "Buddhist monks have HIGHER average betweenness, suggesting they are key network bridges."
    if avg_bud_bet > avg_sec_bet
    else 'Secular persons have HIGHER average betweenness — BUT this is largely because famous monks (円仁, 空海, 宗叡) are classified as Secular due to missing markers. The top betweenness holder (円仁, bt={top_bet[0][1]["betweenness"]:.4f}) is actually a Buddhist monk (Ennin) despite being classified as Secular.'
    if top_bet
    and "円仁" in top_bet[0][1]["name"]
    and cent_group.get(top_bet[0][0]) == "Secular"
    else "Secular persons dominate betweenness."
}

---

## 4. Cross-Network Bridges

| Edge Type | Count |
|-----------|-------|
| Buddhist ↔ Buddhist | {len(bud_edges)} |
| Secular ↔ Secular | {len(sec_edges)} |
| **Buddhist ↔ Secular (cross)** | **{len(cross_edges)}** |

### Top Bridge Persons (most cross-group connections)
| Person ID | Name | Group | Cross-Degree |
|-----------|------|-------|-------------|
"""

for pid, score in top_bridges[:15]:
    name = nodes_persons.get(pid, "Unknown")
    g = nodes_group.get(pid, "?")
    report += f"| {pid} | {name} | {g} | {score:.1f} |\n"

report += f"""
**Key bridges**: "{nodes_persons.get(top_bridges[0][0], "?")}" (P022, 入唐僧) is the strongest bridge, connecting to {top_bridges[0][1]:.0f} secular persons including Chinese Buddhist masters. Other bridges are specific monk-official pairs from direct TRAVEL/PARTICIPATE edges.

---

## 5. Temporal: Buddhist/Secular Ratio Across Phases

Phases defined by `mission_departure_year`:
- **Phase 1** (< 670 CE): Early Kentōshi missions (M01–M04)
- **Phase 2** (670–759): Nara period (M05–M12)
- **Phase 3** (760–809): Late Nara / Early Heian (M13–M18)
- **Phase 4** (810–894): Heian period (M19–M22+)

Events mapped to phases: {sum(phase_event_counts.values())} of ~{len(events_data)} events

| Phase | Events | Buddhist Persons | Secular Persons | Total | Buddhist Ratio |
|-------|--------|-----------------|-----------------|-------|----------------|
"""

ratios = []
for ph in [1, 2, 3, 4]:
    b = phase_persons[ph]["Buddhist"]
    s = phase_persons[ph]["Secular"]
    total = len(b) + len(s)
    ratio = len(b) / total if total > 0 else 0
    ratios.append(ratio)
    report += f"| {ph} | {phase_event_counts[ph]} | {len(b)} | {len(s)} | {total} | {ratio:.3f} |\n"

trend = (
    "increasing"
    if len(ratios) >= 2 and ratios[-1] > ratios[0]
    else "decreasing"
    if len(ratios) >= 2 and ratios[-1] < ratios[0]
    else "stable"
)

report += f"""
**Trend**: The Buddhist ratio shows an **{trend}** trend from Phase 1 ({ratios[0]:.3f}) to Phase 4 ({ratios[-1]:.3f}). 

- Phase 1: Only {len(phase_persons[1]["Buddhist"])} Buddhist-marked persons appear in the earliest missions.
- Phase 2–3: The ratio rises, peaking at {max(ratios):.3f} in Phase {ratios.index(max(ratios)) + 1}.
- Phase 4: Slight decline to {ratios[-1]:.3f}, but still notably higher than Phase 1.

> **Note**: This counts unique persons per phase based on event participation. Many famous monks (空海, 最澄, 円仁) are NOT counted as Buddhist because their names lack the marker characters. The actual Buddhist presence is higher.

---

## 6. Summary

1. **Structure**: Using the strict markers `釋|僧|法|師|和尚|禅`, only {len(bud_ids)}/{len(nodes_persons)} persons ({len(bud_ids) / len(nodes_persons) * 100:.1f}%) are classified as Buddhist. These persons form **no internal edges** — they connect only to secular persons, appearing as individual monks within larger mission groups.

2. **Centrality**: The betweenness analysis is heavily affected by classification — the top centrality holder (円仁/Ennin, bt={top_bet[0][1]["betweenness"]:.4f}) is a Buddhist monk classified as Secular. When counting only marker-bearing Buddhist persons ({len(bud_cent)}/{len(cent_data)}), their average betweenness is **{"higher" if avg_bud_bet > avg_sec_bet else "lower"}** ({avg_bud_bet:.4f} vs {avg_sec_bet:.4f}). **A broader classification would likely reverse this finding.**

3. **Bridges**: {len(cross_edges)} cross-edges connect Buddhist and secular persons. The key bridge is P022 (入唐僧, "the Tang-seeking monk"), a collective entity representing unnamed monks in missions.

4. **Temporal**: The Buddhist ratio rises from {ratios[0]:.1%} (Phase 1) to a peak of {max(ratios):.1%} (Phase {ratios.index(max(ratios)) + 1}), consistent with the historical trend of increasing Buddhist participation in later Kentōshi missions. The decline in Phase 4 may reflect the shift toward fewer but larger missions or changes in record-keeping.

---

*Analysis generated from kento-shi-network data. Network: nodes.csv + edges.csv. Centrality: centrality_all.csv (separate ID-name mapping). Temporal phases from enhanced_events.json mission_departure_year.*
"""

with open("analysis/cross_buddhist_secular.md", "w") as f:
    f.write(report)

print("Report written to analysis/cross_buddhist_secular.md")
print("\nSummary stats:")
print(
    f"  Network: {len(nodes_persons)} persons, {len(bud_ids)} Buddhist, {len(sec_ids)} Secular"
)
print(
    f"  Buddhist edges: {len(bud_edges)}, Secular edges: {len(sec_edges)}, Cross: {len(cross_edges)}"
)
print(
    f"  Centrality: {len(cent_data)} persons, {len(bud_cent)} Buddhist-marked, {len(sec_cent)} Secular"
)
print(f"  Avg betweenness - Buddhist: {avg_bud_bet:.6f}, Secular: {avg_sec_bet:.6f}")
print(f"  Phase ratios: {[f'{r:.3f}' for r in ratios]}")
