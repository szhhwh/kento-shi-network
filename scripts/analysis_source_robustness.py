#!/usr/bin/env python3
"""Source-stratified robustness analysis for kento-shi network.

ANALYSIS 1: Source Stratification
- Group events by source_file pattern (ennin, japan_chronicles, china_histories, other)
- Build Person-Person Jaccard-weighted network per stratum
- Compare nodes, edges, density, centrality top-10, rank correlation

ANALYSIS 2: Robustness to Ennin's Diary
- Build full network vs network WITHOUT ennind_diary events
- Compare: nodes/edges lost, centrality ranking changes
- Spearman rank correlation of Betweenness before/after removal

ANALYSIS 3: Random Drop Robustness
- Randomly drop 10%, 20%, 30%, 50% of edges, 10 reps per level
- Measure mean Spearman correlation of centrality rankings vs original
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from scipy.stats import spearmanr
import json
import os
import warnings

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUT_DIR), "output")
REPORT_PATH = os.path.join(OUT_DIR, "source_bias_report.md")

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
nodes = pd.read_csv(
    os.path.join(DATA_DIR, "nodes.csv"), dtype=str, keep_default_na=False
)
edges = pd.read_csv(
    os.path.join(DATA_DIR, "edges.csv"), dtype=str, keep_default_na=False
)

with open(os.path.join(DATA_DIR, "enhanced_events.json")) as f:
    events = json.load(f)

print(f"Events: {len(events)}, Nodes: {len(nodes)}, Edges: {len(edges)}")

# Filter valid data
nodes = nodes[nodes["type"].isin(["Person", "Event", "Place", "Culture"])].copy()
valid_rels = [
    "HAS_SUBJECT",
    "HAS_OBJECT",
    "AT_LOCATION",
    "TRAVEL",
    "PARTICIPATE",
    "LEARN",
    "INTRODUCE",
]
edges = edges[edges["relation"].isin(valid_rels)].copy()

node_type = dict(zip(nodes["id"], nodes["type"]))
node_name = dict(zip(nodes["id"], nodes["name"]))

META_RELS = {"HAS_SUBJECT", "HAS_OBJECT", "AT_LOCATION"}
SOCIAL_RELS = {"TRAVEL", "PARTICIPATE", "LEARN", "INTRODUCE"}

social_edges = edges[edges["relation"].isin(SOCIAL_RELS)].copy()
print(f"Social edges: {len(social_edges)}")


# ── Helper: Jaccard similarity ──────────────────────────────────────────────
def jaccard(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union > 0 else 0.0


# ── Helper: Build Person-Person Jaccard graph from a set of edge indices ────
def build_pp_jaccard(edge_df):
    """Build Person-Person Jaccard-weighted network from social edges."""
    # TRAVEL: Person↔Place/Target
    travel_edges = edge_df[edge_df["relation"] == "TRAVEL"]
    person_to_places = defaultdict(set)
    for _, row in travel_edges.iterrows():
        src, tgt = row["source"], row["target"]
        if node_type.get(src) == "Person":
            person_to_places[src].add(tgt)
        elif node_type.get(tgt) == "Person":
            person_to_places[tgt].add(src)
        elif node_type.get(src) != "Person" and node_type.get(tgt) != "Person":
            pass

    # LEARN/INTRODUCE: Person↔Culture/Target
    culture_edges = edge_df[edge_df["relation"].isin(["LEARN", "INTRODUCE"])]
    person_to_cultures = defaultdict(set)
    for _, row in culture_edges.iterrows():
        src, tgt = row["source"], row["target"]
        if node_type.get(src) == "Person":
            person_to_cultures[src].add(tgt)
        elif node_type.get(tgt) == "Person":
            person_to_cultures[tgt].add(src)

    # Combine all persons
    all_persons = sorted(
        set(list(person_to_places.keys()) + list(person_to_cultures.keys()))
    )

    G = nx.Graph()
    for p in all_persons:
        G.add_node(p, type="Person")

    # Jaccard projection: Person-Person via shared targets (TRAVEL)
    for i in range(len(all_persons)):
        for j in range(i + 1, len(all_persons)):
            pi, pj = all_persons[i], all_persons[j]
            jw_place = jaccard(
                person_to_places.get(pi, set()), person_to_places.get(pj, set())
            )
            jw_culture = jaccard(
                person_to_cultures.get(pi, set()), person_to_cultures.get(pj, set())
            )
            jw = max(jw_place, jw_culture)
            if jw > 0:
                G.add_edge(pi, pj, weight=jw)

    return G


# ── Helper: Compute centrality ──────────────────────────────────────────────
def compute_centrality(G):
    """Compute centrality measures for a graph. Returns DataFrame."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    # Weighted degree
    wdeg = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}

    # Betweenness (using weight as similarity → shorter path for higher weight)
    betweenness = {}
    if G.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    else:
        betweenness = {n: 0.0 for n in G.nodes()}

    df = pd.DataFrame(
        {
            "person_id": list(G.nodes()),
            "person_name": [node_name.get(n, n) for n in G.nodes()],
            "weighted_degree": [wdeg[n] for n in G.nodes()],
            "betweenness": [betweenness[n] for n in G.nodes()],
        }
    )
    return df.sort_values("betweenness", ascending=False)


# ── Helper: Spearman rank correlation between two centrality DataFrames ─────
def spearman_corr(df1, df2, col="betweenness"):
    """Compute Spearman correlation on shared persons."""
    shared = set(df1["person_id"]) & set(df2["person_id"])
    if len(shared) < 3:
        return np.nan, len(shared)
    d1 = df1.set_index("person_id").loc[list(shared)]
    d2 = df2.set_index("person_id").loc[list(shared)]
    r, p = spearmanr(d1[col], d2[col])
    return r, len(shared)


# ── Source stratification ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 1: SOURCE STRATIFICATION")
print("=" * 70)

# Map events to their source stratum via source_text
# Build dict: event_id -> source stratum
event_stratum = {}
for e in events:
    sf = e.get("source_file", "")
    if "ennin" in sf.lower():
        event_stratum[e["event_type"]] = (
            "ennin_diary"  # use event_type as key? No, need event_id
        )
    # Actually, we need to map event_id to stratum
    # But events don't have IDs directly. Let me find them by matching source_text with nodes

# Better approach: map source_text -> stratum
# Since edges reference events by ID from nodes.csv
source_text_to_stratum = {}
for e in events:
    sf = e.get("source_file", "")
    st = e.get("source_text", "")
    if "ennin" in sf.lower():
        source_text_to_stratum[st] = "ennin_diary"
    elif "japan_clean" in sf:
        source_text_to_stratum[st] = "japan_chronicles"
    elif "china_clean" in sf:
        source_text_to_stratum[st] = "china_histories"
    elif sf:
        source_text_to_stratum[st] = "other_zenrin"
    else:
        source_text_to_stratum[st] = "unlabeled"

print(f"Source text entries mapped: {len(source_text_to_stratum)}")
stratum_counts = Counter(source_text_to_stratum.values())
for k, v in stratum_counts.most_common():
    print(f"  {k}: {v}")

# Map edges to stratum via source_text
# Each edge has source_text field
edge_stratum = {}
for idx, row in social_edges.iterrows():
    st = row.get("source_text", "")
    stratum = source_text_to_stratum.get(st, "unlabeled")
    edge_stratum[idx] = stratum

# Also try matching edges via event node in metadata edges
# An event (E###) connects to persons through HAS_SUBJECT/HAS_OBJECT
# We need to know which event nodes belong to which stratum
event_to_stratum = {}
for e in events:
    sf = e.get("source_file", "")
    st = e.get("source_text", "")
    if "ennin" in sf.lower():
        stratum = "ennin_diary"
    elif "japan_clean" in sf:
        stratum = "japan_chronicles"
    elif "china_clean" in sf:
        stratum = "china_histories"
    elif sf:
        stratum = "other_zenrin"
    else:
        stratum = "unlabeled"
    # Find event node by matching source_text in nodes
    event_to_stratum[st] = stratum

# Now map edges: an edge belongs to a stratum if its source_text matches
# or if its source/target is an event node in that stratum
# Better: find event nodes in meta_edges, then propagate to social edges
meta_edges = edges[edges["relation"].isin(META_RELS)].copy()

# Build event_meta_map: event_id -> set of person/place IDs it touches
event_nodes = nodes[nodes["type"] == "Event"]
event_text_to_id = {}
for _, row in event_nodes.iterrows():
    sources = row.get("sources", "")
    if sources:
        # The 'sources' field has the source_text
        event_text_to_id[sources] = row["id"]

# Map source_text to event_id
print(f"\nEvent nodes: {len(event_nodes)}")
print(f"event_text_to_id entries: {len(event_text_to_id)}")

# For each edge, determine stratum:
# 1. By matching source_text on the edge
# 2. By matching event nodes connected to the persons in meta_edges
# Build: person_id -> set of strata (from events they're connected to)
person_strata = defaultdict(set)
for _, row in meta_edges.iterrows():
    src, tgt, st = row["source"], row["target"], row.get("source_text", "")
    stratum = event_to_stratum.get(st, "unlabeled")
    # Both src and tgt could be person or event
    if node_type.get(src) == "Person":
        person_strata[src].add(stratum)
    if node_type.get(tgt) == "Person":
        person_strata[tgt].add(stratum)

# For social edges, assign stratum from source_text if available
social_edge_strata = {}
for idx, row in social_edges.iterrows():
    st = row.get("source_text", "")
    stratum = event_to_stratum.get(st, None)
    if stratum is None:
        # Try to infer from persons involved
        src, tgt = row["source"], row["target"]
        src_strata = person_strata.get(src, set())
        tgt_strata = person_strata.get(tgt, set())
        combined = src_strata | tgt_strata
        if len(combined) == 1:
            stratum = list(combined)[0]
        elif len(combined) > 1:
            stratum = "multi_source"
        else:
            stratum = "unlabeled"
    social_edge_strata[idx] = stratum

# Count social edges by stratum
social_stratum_counts = Counter(social_edge_strata.values())
print("\nSocial edges by stratum:")
for k, v in social_stratum_counts.most_common():
    print(f"  {k}: {v}")

# Build networks by stratum
stratum_nets = {}
stratum_cent = {}
for stratum in [
    "ennin_diary",
    "japan_chronicles",
    "china_histories",
    "other_zenrin",
    "unlabeled",
]:
    indices = [idx for idx, s in social_edge_strata.items() if s == stratum]
    if not indices:
        print(f"\n{stratum}: No edges")
        stratum_nets[stratum] = nx.Graph()
        stratum_cent[stratum] = pd.DataFrame()
        continue

    sub_edges = social_edges.loc[indices].copy()
    G = build_pp_jaccard(sub_edges)
    stratum_nets[stratum] = G
    df = compute_centrality(G)
    stratum_cent[stratum] = df

    density = nx.density(G) if G.number_of_nodes() > 1 else 0
    print(f"\n{stratum}:")
    print(
        f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Density: {density:.6f}"
    )
    if len(df) > 0:
        print("  Top-5 Betweenness:")
        for _, row in df.head(5).iterrows():
            print(f"    {row['person_name']}: {row['betweenness']:.6f}")
        print("  Top-5 Weighted Degree:")
        wd_top = df.nlargest(5, "weighted_degree")
        for _, row in wd_top.iterrows():
            print(f"    {row['person_name']}: {row['weighted_degree']:.6f}")

# Cross-stratum Spearman correlations
print("\n--- Cross-stratum Betweenness Spearman correlations ---")
strata_list = ["ennin_diary", "japan_chronicles", "china_histories", "other_zenrin"]
for i in range(len(strata_list)):
    for j in range(i + 1, len(strata_list)):
        s1, s2 = strata_list[i], strata_list[j]
        if len(stratum_cent[s1]) > 0 and len(stratum_cent[s2]) > 0:
            r, n = spearman_corr(stratum_cent[s1], stratum_cent[s2])
            print(f"  {s1} vs {s2}: r={r:.4f} (n={n})")

# ── Full network ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 2: ROBUSTNESS TO ENNIN'S DIARY")
print("=" * 70)

G_full = build_pp_jaccard(social_edges)
cent_full = compute_centrality(G_full)
print(
    f"Full network: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges"
)
print("Top-5 Betweenness (full):")
for _, row in cent_full.head(5).iterrows():
    print(f"  {row['person_name']}: {row['betweenness']:.6f}")

# Build network WITHOUT Ennin diary events
ennin_edge_indices = [
    idx for idx, s in social_edge_strata.items() if s == "ennin_diary"
]
non_ennin_indices = [
    idx for idx in social_edges.index if idx not in set(ennin_edge_indices)
]
non_ennin_edges = social_edges.loc[non_ennin_indices].copy()

G_no_ennin = build_pp_jaccard(non_ennin_edges)
cent_no_ennin = compute_centrality(G_no_ennin)
print(
    f"\nNetwork WITHOUT Ennin diary: {G_no_ennin.number_of_nodes()} nodes, {G_no_ennin.number_of_edges()} edges"
)

# Compare
nodes_lost = set(G_full.nodes()) - set(G_no_ennin.nodes())
edges_lost = G_full.number_of_edges() - G_no_ennin.number_of_edges()
print(f"Nodes lost: {len(nodes_lost)}")
if nodes_lost:
    lost_names = [node_name.get(n, n) for n in list(nodes_lost)[:10]]
    print(f"  Lost persons (first 10): {', '.join(lost_names)}")
print(f"Edges lost: {edges_lost}")

# Centrality rank correlation
r_betweenness, n_shared = spearman_corr(cent_full, cent_no_ennin, "betweenness")
r_wdeg, _ = spearman_corr(cent_full, cent_no_ennin, "weighted_degree")
print("\nSpearman correlation (full vs no-ennin):")
print(f"  Betweenness: r={r_betweenness:.4f} (n={n_shared} shared persons)")
print(f"  Weighted Degree: r={r_wdeg:.4f}")

print("\nTop-5 Betweenness (no Ennin):")
for _, row in cent_no_ennin.head(5).iterrows():
    print(f"  {row['person_name']}: {row['betweenness']:.6f}")

# ── Random Drop Robustness ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 3: RANDOM DROP ROBUSTNESS")
print("=" * 70)

np.random.seed(42)
drop_levels = [0.10, 0.20, 0.30, 0.50]
n_reps = 10
full_edges_list = list(G_full.edges())

# Pre-compute full centrality for comparison
full_cent_dict = cent_full.set_index("person_id")[
    ["betweenness", "weighted_degree"]
].to_dict("index")

results = {}
for level in drop_levels:
    level_corrs_bw = []
    level_corrs_wd = []
    for rep in range(n_reps):
        n_drop = int(len(full_edges_list) * level)
        keep_edges = full_edges_list.copy()
        drop_indices = np.random.choice(len(keep_edges), size=n_drop, replace=False)
        keep_edges = [e for i, e in enumerate(keep_edges) if i not in set(drop_indices)]

        # Build graph from kept edges
        G_drop = nx.Graph()
        for p in G_full.nodes():
            G_drop.add_node(p)
        for u, v in keep_edges:
            w = G_full[u][v]["weight"]
            G_drop.add_edge(u, v, weight=w)

        cent_drop = compute_centrality(G_drop)

        # Spearman correlation
        shared = set(cent_drop["person_id"]) & set(full_cent_dict.keys())
        if len(shared) >= 3:
            d1 = cent_drop.set_index("person_id").loc[list(shared)]
            bw_full = [full_cent_dict[pid]["betweenness"] for pid in shared]
            bw_drop = [d1.loc[pid, "betweenness"] for pid in shared]
            r_bw, _ = spearmanr(bw_full, bw_drop)

            wd_full = [full_cent_dict[pid]["weighted_degree"] for pid in shared]
            wd_drop = [d1.loc[pid, "weighted_degree"] for pid in shared]
            r_wd, _ = spearmanr(wd_full, wd_drop)
        else:
            r_bw, r_wd = np.nan, np.nan

        level_corrs_bw.append(r_bw)
        level_corrs_wd.append(r_wd)

    mean_bw = np.nanmean(level_corrs_bw)
    std_bw = np.nanstd(level_corrs_bw)
    mean_wd = np.nanmean(level_corrs_wd)
    std_wd = np.nanstd(level_corrs_wd)
    results[level] = {
        "betweenness_mean": mean_bw,
        "betweenness_std": std_bw,
        "wdegree_mean": mean_wd,
        "wdegree_std": std_wd,
    }
    print(
        f"  {level * 100:.0f}% drop: BW r={mean_bw:.4f}±{std_bw:.4f}, WD r={mean_wd:.4f}±{std_wd:.4f}"
    )

# ── Write report ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WRITING REPORT")
print("=" * 70)

report_lines = []
report_lines.append("# Source Bias & Robustness Analysis Report\n")
report_lines.append("## Overview\n")
report_lines.append(f"- Total events: {len(events)}")
report_lines.append(f"- Total social edges: {len(social_edges)}")
report_lines.append(
    f"- Full network: {G_full.number_of_nodes()} persons, {G_full.number_of_edges()} edges\n"
)

# Section 1
report_lines.append("## 1. Source Stratification\n")
report_lines.append("### Source Distribution\n")
report_lines.append(
    "| Source Stratum | Events | Social Edges | Persons | Edges | Density |"
)
report_lines.append("|---|---|---|---|---|---|")
for stratum in [
    "ennin_diary",
    "japan_chronicles",
    "china_histories",
    "other_zenrin",
    "unlabeled",
]:
    evt_count = stratum_counts.get(stratum, 0)
    edge_count = social_stratum_counts.get(stratum, 0)
    G = stratum_nets.get(stratum, nx.Graph())
    density = nx.density(G) if G.number_of_nodes() > 1 else 0
    report_lines.append(
        f"| {stratum} | {evt_count} | {edge_count} | {G.number_of_nodes()} | {G.number_of_edges()} | {density:.6f} |"
    )

report_lines.append("")

# Top-10 tables per stratum
for stratum in ["ennin_diary", "japan_chronicles", "china_histories", "other_zenrin"]:
    df = stratum_cent.get(stratum, pd.DataFrame())
    if len(df) == 0:
        continue
    report_lines.append(f"### {stratum} — Top-10 Betweenness Centrality\n")
    report_lines.append("| Rank | Person | Betweenness | Weighted Degree |")
    report_lines.append("|---|---|---|---|")
    for rank, (_, row) in enumerate(df.head(10).iterrows(), 1):
        report_lines.append(
            f"| {rank} | {row['person_name']} | {row['betweenness']:.6f} | {row['weighted_degree']:.6f} |"
        )
    report_lines.append("")

# Cross-stratum correlations
report_lines.append("### Cross-Stratum Betweenness Rank Correlations\n")
report_lines.append("| Stratum A | Stratum B | Spearman r | Shared Persons |")
report_lines.append("|---|---|---|---|")
for i in range(len(strata_list)):
    for j in range(i + 1, len(strata_list)):
        s1, s2 = strata_list[i], strata_list[j]
        if len(stratum_cent[s1]) > 0 and len(stratum_cent[s2]) > 0:
            r, n = spearman_corr(stratum_cent[s1], stratum_cent[s2])
            report_lines.append(f"| {s1} | {s2} | {r:.4f} | {n} |")

report_lines.append("")

# Section 2
report_lines.append("## 2. Robustness to Ennin's Diary Removal\n")
report_lines.append(
    f"- Full network: {G_full.number_of_nodes()} persons, {G_full.number_of_edges()} edges"
)
report_lines.append(
    f"- Without Ennin diary: {G_no_ennin.number_of_nodes()} persons, {G_no_ennin.number_of_edges()} edges"
)
report_lines.append(f"- Persons lost: {len(nodes_lost)}")
report_lines.append(f"- Edges lost: {edges_lost}")
report_lines.append(f"- Betweenness Spearman r: {r_betweenness:.4f} (n={n_shared})")
report_lines.append(f"- Weighted Degree Spearman r: {r_wdeg:.4f}")
report_lines.append("")

report_lines.append("### Top-10 Betweenness Shift (Full vs No-Ennin)\n")
report_lines.append("| Rank | Full Network | Full BW | No-Ennin | No-Ennin BW |")
report_lines.append("|---|---|---|---|---|")
for rank in range(min(10, len(cent_full))):
    f_name = cent_full.iloc[rank]["person_name"]
    f_bw = cent_full.iloc[rank]["betweenness"]
    if rank < len(cent_no_ennin):
        ne_name = cent_no_ennin.iloc[rank]["person_name"]
        ne_bw = cent_no_ennin.iloc[rank]["betweenness"]
    else:
        ne_name, ne_bw = "-", "-"
    report_lines.append(
        f"| {rank + 1} | {f_name} | {f_bw:.6f} | {ne_name} | {ne_bw:.6f} |"
    )

report_lines.append("")

# Section 3
report_lines.append("## 3. Random Edge-Drop Stability\n")
report_lines.append(
    "| Drop Rate | Betweenness r (mean±std) | Weighted Degree r (mean±std) |"
)
report_lines.append("|---|---|---|")
for level in drop_levels:
    r = results[level]
    report_lines.append(
        f"| {level * 100:.0f}% | {r['betweenness_mean']:.4f}±{r['betweenness_std']:.4f} | {r['wdegree_mean']:.4f}±{r['wdegree_std']:.4f} |"
    )

report_lines.append("")
report_lines.append("### Interpretation\n")
report_lines.append(
    "- **Source stratification**: Networks built from different text sources vary substantially in size and structure. "
)
report_lines.append(
    "  High rank correlations between strata suggest the core set of important figures is consistent across sources."
)
report_lines.append(
    "- **Ennin diary dependency**: The degree of centrality rank preservation after removing Ennin's diary "
)
report_lines.append(
    "  indicates whether the network structure is dominated by this single source."
)
report_lines.append(
    "- **Random drop stability**: High correlation under random edge removal indicates the network's "
)
report_lines.append(
    "  centrality rankings are robust to partial data loss. Declining correlations at higher drop rates "
)
report_lines.append("  reveal the fragility threshold.\n")
report_lines.append(
    f"*Report generated from {len(events)} events, {G_full.number_of_nodes()} persons.*\n"
)

with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"Report written to {REPORT_PATH}")
print("Done.")
