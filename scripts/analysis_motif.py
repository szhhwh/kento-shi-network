#!/usr/bin/env python3
"""
Triad Motif Census for Kento-shi Person-Person Jaccard-weighted Network.

Tasks:
1. Count all 3-node triad types (networkx triad census). Compare to random baseline.
2. Identify most common 3-person interaction patterns.
3. For top 5 motifs, show 3 concrete examples with actual person names.
4. Per-community breakdown (Louvain communities).
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import os
import itertools
import random
import warnings

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUT_DIR), "output")


# ── Load data ─────────────────────────────────────────────────────────────
def safe_read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


nodes = safe_read_csv(os.path.join(DATA_DIR, "nodes.csv"))
edges = safe_read_csv(os.path.join(DATA_DIR, "edges.csv"))

nodes = nodes[nodes["type"].isin(["Person", "Event", "Place", "Culture"])].copy()
edges = edges[
    edges["relation"].isin(
        [
            "HAS_SUBJECT",
            "HAS_OBJECT",
            "AT_LOCATION",
            "TRAVEL",
            "PARTICIPATE",
            "LEARN",
            "INTRODUCE",
        ]
    )
].copy()

SOCIAL_RELS = {"TRAVEL", "PARTICIPATE", "LEARN", "INTRODUCE"}
social_edges = edges[edges["relation"].isin(SOCIAL_RELS)].copy()

node_type = dict(zip(nodes["id"], nodes["type"]))
node_name = dict(zip(nodes["id"], nodes["name"]))


# ── Build bipartite projections (same as analysis_jaccard.py) ─────────────
def jaccard(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union > 0 else 0.0


# TRAVEL bipartite
travel_edges = social_edges[social_edges["relation"] == "TRAVEL"].copy()
person_to_places = defaultdict(set)
for _, row in travel_edges.iterrows():
    src, tgt = row["source"], row["target"]
    src_type = node_type.get(src, "")
    tgt_type = node_type.get(tgt, "")
    if src_type == "Person":
        person_to_places[src].add(tgt)
    elif tgt_type == "Person":
        person_to_places[tgt].add(src)

persons_with_places = sorted(person_to_places.keys())

# LEARN + INTRODUCE bipartite
culture_edges = social_edges[
    social_edges["relation"].isin(["LEARN", "INTRODUCE"])
].copy()
person_to_cultures = defaultdict(set)
for _, row in culture_edges.iterrows():
    src, tgt = row["source"], row["target"]
    src_type = node_type.get(src, "")
    tgt_type = node_type.get(tgt, "")
    if src_type == "Person":
        person_to_cultures[src].add(tgt)
    elif tgt_type == "Person":
        person_to_cultures[tgt].add(src)

persons_with_culture = sorted(person_to_cultures.keys())

# Combined Person-Person Jaccard
G_pp = nx.Graph()
all_persons = set(persons_with_places) | set(persons_with_culture)
for p in all_persons:
    G_pp.add_node(p, name=node_name.get(p, p))

# TRAVEL projection
for i in range(len(persons_with_places)):
    for j in range(i + 1, len(persons_with_places)):
        pi, pj = persons_with_places[i], persons_with_places[j]
        jw = jaccard(person_to_places[pi], person_to_places[pj])
        if jw > 0:
            G_pp.add_edge(pi, pj, weight=jw)

# LEARN/INTRODUCE projection
for i in range(len(persons_with_culture)):
    for j in range(i + 1, len(persons_with_culture)):
        pi, pj = persons_with_culture[i], persons_with_culture[j]
        jw = jaccard(person_to_cultures[pi], person_to_cultures[pj])
        if jw > 0:
            if G_pp.has_edge(pi, pj):
                G_pp[pi][pj]["weight"] = max(G_pp[pi][pj]["weight"], jw)
            else:
                G_pp.add_edge(pi, pj, weight=jw)

# Keep largest connected component
components = list(nx.connected_components(G_pp))
largest_cc = max(components, key=len)
G = G_pp.subgraph(largest_cc).copy()

print(
    f"Person-Person Jaccard network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
)

# ── Load community labels ─────────────────────────────────────────────────
# Recompute communities fresh (more robust than loading pre-computed CSV)
try:
    import community as community_louvain

    partition = community_louvain.best_partition(G, weight="weight", resolution=1.0)
    comm_map = partition  # {node_id: community_label}
    print(
        f"Louvain communities: {len(set(partition.values()))} (modularity Q = {community_louvain.modularity(partition, G, weight='weight'):.4f})"
    )
except ImportError:
    # Fallback: try loading from CSV
    comm_df = pd.read_csv(os.path.join(OUT_DIR, "communities.csv"), dtype=str)
    comm_map = dict(zip(comm_df["person_id"], comm_df["community"].astype(int)))
    print(
        f"Loaded communities from CSV: {len(comm_map)} persons, {len(set(comm_map.values()))} communities"
    )

# ── 1. TRIAD CENSUS ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. TRIAD MOTIF CENSUS")
print("=" * 70)

# Convert to unweighted (presence/absence) for triad census
G_uw = nx.Graph()
G_uw.add_nodes_from(G.nodes())
G_uw.add_edges_from(G.edges())

# Full triad census (requires directed graph)
G_dir = nx.DiGraph()
G_dir.add_nodes_from(G_uw.nodes())
for u, v in G_uw.edges():
    G_dir.add_edge(u, v)
    G_dir.add_edge(v, u)

tc = nx.triadic_census(G_dir)

print("\nTriad Census (undirected treated as symmetric directed):")
print(f"  Total triads: {sum(tc.values())}")

# Map triad codes to readable names
TRIAD_NAMES = {
    "003": "Empty (no edges)",
    "012": "Single edge",
    "102": "Open pair (2 edges, V-shape)",
    "021D": "Star: 1 source, 2 targets (out-out)",
    "021U": "Star: 2 sources, 1 target (in-in)",
    "021C": "Star: directed path of 2",
    "111D": "3-cycle missing 1 reciprocal",
    "111U": "3-cycle missing 1 reciprocal (alt)",
    "030T": "Transitive triad (3 edges, all one-way)",
    "030C": "Cycle (3-cycle)",
    "201": "3 edges: 2 mutual + 1 asymmetric",
    "120D": "4 edges: 2 mutuals sharing source",
    "120U": "4 edges: 2 mutuals sharing target",
    "120C": "4 edges: mutual + path",
    "210": "5 edges",
    "300": "Complete (6 edges, all mutual)",
}

print(f"\n{'Code':<8} {'Name':<45} {'Count':>8} {'%':>8}")
print("-" * 72)
total = sum(tc.values())
for code in sorted(tc.keys()):
    name = TRIAD_NAMES.get(code, code)
    cnt = tc[code]
    pct = cnt / total * 100 if total > 0 else 0
    print(f"{code:<8} {name:<45} {cnt:>8} {pct:>7.2f}%")

# ── For undirected graphs, the meaningful codes are ────────────────────────
# 003 = 0 edges (empty, but in symmetric digraph, 003 means no edges in either dir → actually 0 edges)
# Wait, for a symmetric digraph of an undirected graph:
# - 003: no edges at all → truly empty triad
# - 012: exactly 1 directed edge → impossible in symmetric digraph of undirected (always 2 edges)
# Let me think about this more carefully.
#
# In an undirected graph, a triad can have:
# - 0 edges (empty) → in directed symmetric: 0 directed edges → code 003
# - 1 edge → 2 directed edges (u→v, v→u) → code 102 (mutual dyad + isolate)
# - 2 edges (path of 3: u-v-w) → 4 directed edges (u↔v, v↔w) → code 201
# - 3 edges (triangle) → 6 directed edges → code 300
#
# So for undirected networks the relevant codes are: 003, 102, 201, 300.
# The other codes (012, 021D/U/C, 111D/U, 030T/C, 120D/U/C, 210) should be zero.

print("\n--- Undirected-mapped triad interpretation ---")
UNDIRECTED_MAP = {
    "003": ("Empty triad (0 edges)", 0),
    "102": ("Single edge (1 edge, mutual dyad + isolate)", 1),
    "201": ("Path of 2 edges (V-shape / open triplet)", 2),
    "300": ("Triangle / closed triad (3 edges, fully connected)", 3),
}

for code, (label, nedges) in UNDIRECTED_MAP.items():
    cnt = tc.get(code, 0)
    print(f"  {code}: {label} → {cnt} ({cnt / total * 100:.2f}%)")

# ── 2. RANDOM BASELINE ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. RANDOM BASELINE (Configuration Model)")
print("=" * 70)

n_random = 100
random.seed(42)
np.random.seed(42)

# Configuration model: preserve degree sequence
deg_seq = [d for _, d in G_uw.degree()]
# Filter out isolated nodes that might cause issues
deg_seq = [d for d in deg_seq if d > 0]

random_triads = defaultdict(list)
for trial in range(n_random):
    try:
        G_rand = nx.configuration_model(deg_seq, seed=42 + trial)
        G_rand = nx.Graph(G_rand)
        G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
        # Take largest component
        if G_rand.number_of_nodes() > 0:
            comps = list(nx.connected_components(G_rand))
            largest = max(comps, key=len)
            G_rand = G_rand.subgraph(largest).copy()

        # Convert to symmetric directed
        G_rand_dir = nx.DiGraph()
        G_rand_dir.add_nodes_from(G_rand.nodes())
        for u, v in G_rand.edges():
            G_rand_dir.add_edge(u, v)
            G_rand_dir.add_edge(v, u)

        tc_rand = nx.triadic_census(G_rand_dir)
        for code in tc_rand:
            random_triads[code].append(tc_rand[code])
    except Exception:
        pass

print(f"Generated {len(random_triads.get('003', []))} successful random networks")

print(
    f"\n{'Code':<8} {'Actual':>8} {'Random Mean':>12} {'Random Std':>10} {'Z-score':>10} {'Over/Under':>12}"
)
print("-" * 70)
for code in ["003", "102", "201", "300"]:
    actual = tc.get(code, 0)
    rand_vals = random_triads.get(code, [0])
    rand_mean = np.mean(rand_vals)
    rand_std = np.std(rand_vals)
    z = (actual - rand_mean) / rand_std if rand_std > 0 else 0
    direction = "OVER" if z > 2 else ("UNDER" if z < -2 else "≈ expected")
    print(
        f"{code:<8} {actual:>8} {rand_mean:>12.1f} {rand_std:>10.1f} {z:>10.2f} {direction:>12}"
    )

# ── 3. OPEN TRIPLETS (V-shape) vs CLOSED TRIANGLES ────────────────────────
print("\n" + "=" * 70)
print("3. OPEN TRIPLETS vs CLOSED TRIANGLES")
print("=" * 70)

open_triplets = tc.get("201", 0)
closed_triads = tc.get("300", 0)
transitivity = (
    3 * closed_triads / (3 * closed_triads + open_triplets)
    if (closed_triads + open_triplets) > 0
    else 0
)
print(f"Open triplets (201): {open_triplets}")
print(f"Closed triangles (300): {closed_triads}")
print(f"Transitivity (clustering coefficient): {transitivity:.4f}")
print(f"Average clustering coefficient (nx): {nx.average_clustering(G_uw):.4f}")

# Find the most central triangles (high-weight)
print("\n--- Most significant triangles (by sum of edge weights) ---")
triangles_list = []
all_cliques = nx.enumerate_all_cliques(G)
for clique in all_cliques:
    if len(clique) == 3:
        w_sum = 0
        edges_in = []
        for a, b in itertools.combinations(clique, 2):
            if G.has_edge(a, b):
                w = G[a][b]["weight"]
            else:
                w = 0
            w_sum += w
            edges_in.append((a, b, w))
        triangles_list.append((clique, w_sum, edges_in))
    elif len(clique) > 3:
        break  # don't need larger cliques

triangles_list.sort(key=lambda x: -x[1])
print(f"Total unique triangles: {len(triangles_list)}")

for i, (clique, w_sum, edges_in) in enumerate(triangles_list[:10]):
    names = [node_name.get(n, n) for n in clique]
    edges_str = ", ".join(
        f"{node_name.get(a, a)}-{node_name.get(b, b)}:{w:.3f}" for a, b, w in edges_in
    )
    print(f"  #{i + 1}: {' + '.join(names)} (Σw={w_sum:.4f}) [{edges_str}]")

# ── 4. TOP 5 MOTIFS with EXAMPLES ─────────────────────────────────────────
print("\n" + "=" * 70)
print("4. TOP 5 MOTIFS WITH CONCRETE EXAMPLES")
print("=" * 70)

# For undirected networks, the motifs are:
# 1. Triangle (3 edges) - fully connected trio
# 2. Path of 2 (V-shape) - one person connects two others
# 3. Single edge + isolate
# 4. Empty

# Find examples of V-shapes (open triplets) - hub-spoke stars
print("\n--- Motif 1: Closed Triangle (fully connected trio) ---")
for i, (clique, w_sum, edges_in) in enumerate(triangles_list[:3]):
    names = [node_name.get(n, n) for n in clique]
    print(f"  Example {i + 1}: {' ↔ '.join(names)}")
    for a, b, w in edges_in:
        w_label = f" (Jaccard={w:.3f})"
        places_a = person_to_places.get(a, set()) | person_to_cultures.get(a, set())
        places_b = person_to_places.get(b, set()) | person_to_cultures.get(b, set())
        shared = places_a & places_b
        shared_names = [node_name.get(x, x) for x in list(shared)[:3]]
        print(
            f"    {node_name.get(a, a)} — {node_name.get(b, b)}{w_label}: shared → {', '.join(shared_names)}"
        )

# Find open triplets (V-shapes / stars) - sorted by weight sum
print("\n--- Motif 2: Open Triplet / V-shape (hub connects 2 others) ---")
open_triplet_list = []
for n in G.nodes():
    neighbors = list(G.neighbors(n))
    if len(neighbors) >= 2:
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                ni, nj = neighbors[i], neighbors[j]
                if not G.has_edge(ni, nj):
                    w1 = G[n][ni]["weight"]
                    w2 = G[n][nj]["weight"]
                    w_sum = w1 + w2
                    open_triplet_list.append((n, ni, nj, w1, w2, w_sum))

open_triplet_list.sort(key=lambda x: -x[5])

for i, (hub, a, b, w1, w2, w_sum) in enumerate(open_triplet_list[:3]):
    hub_name = node_name.get(hub, hub)
    a_name = node_name.get(a, a)
    b_name = node_name.get(b, b)
    print(
        f"  Example {i + 1}: {a_name} ← {hub_name} → {b_name} (hub: {hub_name}, Σw={w_sum:.4f})"
    )

    # Show what hub shares with each
    hub_set = person_to_places.get(hub, set()) | person_to_cultures.get(hub, set())
    a_set = person_to_places.get(a, set()) | person_to_cultures.get(a, set())
    b_set = person_to_places.get(b, set()) | person_to_cultures.get(b, set())

    shared_a = hub_set & a_set
    shared_b = hub_set & b_set
    shared_ab = a_set & b_set

    shared_a_names = [node_name.get(x, x) for x in list(shared_a)[:3]]
    shared_b_names = [node_name.get(x, x) for x in list(shared_b)[:3]]

    print(f"    {hub_name}↔{a_name} (J={w1:.3f}): shared → {', '.join(shared_a_names)}")
    print(f"    {hub_name}↔{b_name} (J={w2:.3f}): shared → {', '.join(shared_b_names)}")
    if shared_ab:
        print(
            f"    NOTE: {a_name}↔{b_name} also share {len(shared_ab)} but not directly connected (sub-threshold)"
        )
    else:
        print(f"    {a_name} and {b_name} share nothing directly")

# ── 5. HUB-SPOKE (Star) patterns ─────────────────────────────────────────
print("\n--- Motif 3: Star / Hub-Spoke (1 person connected to ≥3 others) ---")
degree_dict = dict(G.degree())
top_hubs = sorted(degree_dict.items(), key=lambda x: -x[1])[:5]
for hub, deg in top_hubs:
    hub_name = node_name.get(hub, hub)
    neighbors = list(G.neighbors(hub))
    neighbor_names = [node_name.get(n, n) for n in neighbors[:8]]
    weights = [G[hub][n]["weight"] for n in neighbors[:5]]
    w_str = ", ".join(f"{w:.3f}" for w in weights)
    print(
        f"  Hub: {hub_name} (degree={deg}): {', '.join(neighbor_names)}... [weights: {w_str}]"
    )

# ── 6. PER-COMMUNITY BREAKDOWN ────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. PER-COMMUNITY MOTIF BREAKDOWN")
print("=" * 70)

# Get communities present in the largest CC
communities = defaultdict(list)
for n in G.nodes():
    c = comm_map.get(n, -1)
    communities[c].append(n)

print(f"\nCommunities in network: {len(communities)}")
for c, members in sorted(communities.items(), key=lambda x: -len(x[1])):
    names = [node_name.get(m, m) for m in members[:5]]
    print(f"  C{c}: {len(members)} persons ({', '.join(names)}...)")

# Per-community triad analysis
print(
    f"\n{'Comm':<6} {'Size':>6} {'Triangles':>10} {'Open Triplets':>10} {'Transitivity':>12} {'Avg Clust':>10}"
)
print("-" * 60)

for c, members in sorted(communities.items(), key=lambda x: -len(x[1])):
    if len(members) < 3:
        print(
            f"  C{c:<4} {len(members):>6} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10}"
        )
        continue

    sub_g = G_uw.subgraph(members).copy()
    sub_dir = nx.DiGraph()
    sub_dir.add_nodes_from(sub_g.nodes())
    for u, v in sub_g.edges():
        sub_dir.add_edge(u, v)
        sub_dir.add_edge(v, u)

    stc = nx.triadic_census(sub_dir)
    tri = stc.get("300", 0)
    opn = stc.get("201", 0)
    trans = 3 * tri / (3 * tri + opn) if (tri + opn) > 0 else 0
    avg_cc = nx.average_clustering(sub_g) if sub_g.number_of_nodes() > 0 else 0

    print(
        f"  C{c:<4} {len(members):>6} {tri:>10} {opn:>10} {trans:>12.4f} {avg_cc:>10.4f}"
    )

# ── 7. INTER-COMMUNITY EDGES ──────────────────────────────────────────────
print("\n--- Inter-community connectivity ---")
inter_edges = defaultdict(int)
for u, v in G.edges():
    cu = comm_map.get(u, -1)
    cv = comm_map.get(v, -1)
    if cu != cv:
        key = tuple(sorted([cu, cv]))
        inter_edges[key] += 1

for (c1, c2), cnt in sorted(inter_edges.items(), key=lambda x: -x[1]):
    names1 = [node_name.get(m, m) for m in communities[c1][:3]]
    names2 = [node_name.get(m, m) for m in communities[c2][:3]]
    print(
        f"  C{c1} ↔ C{c2}: {cnt} edges  ({', '.join(names1)} ... ↔ ... {', '.join(names2)})"
    )

# ── 8. GENERATE REPORT ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. GENERATING REPORT")
print("=" * 70)

report = []
report.append(
    "# Deep Motif Analysis: Triad Motif Census of the Kento-shi Person-Person Jaccard-weighted Network\n"
)
report.append(
    f"*Network: {G.number_of_nodes()} persons, {G.number_of_edges()} weighted edges*\n"
)

report.append("## 1. Triad Census Overview\n")
report.append(
    "The triad census counts all possible 3-node induced subgraphs in the network."
)
report.append("For undirected networks, only 4 triad types are possible:\n")
report.append("| Code | Description | Edges |")
report.append("|------|-------------|-------|")
for code in ["003", "102", "201", "300"]:
    label, nedges = UNDIRECTED_MAP[code]
    cnt = tc.get(code, 0)
    pct = cnt / total * 100 if total > 0 else 0
    report.append(f"| {code} | {label} | {nedges} |")
report.append("")

report.append("### Census Results\n")
report.append("| Code | Description | Count | % of Total |")
report.append("|------|-------------|-------|------------|")
for code in ["003", "102", "201", "300"]:
    label, _ = UNDIRECTED_MAP[code]
    cnt = tc.get(code, 0)
    pct = cnt / total * 100 if total > 0 else 0
    report.append(f"| {code} | {label} | {cnt} | {pct:.2f}% |")
report.append(f"\n**Total triads:** {total}\n")

# Random baseline comparison
report.append("## 2. Comparison to Random Baseline\n")
report.append(
    "Configuration model preserves degree sequence but randomizes edge connections (n=100 randomizations).\n"
)
report.append("| Code | Actual | Random Mean | Random Std | Z-score | Significance |")
report.append("|------|--------|-------------|------------|---------|--------------|")
for code in ["003", "102", "201", "300"]:
    actual = tc.get(code, 0)
    rand_vals = random_triads.get(code, [0])
    rand_mean = np.mean(rand_vals)
    rand_std = np.std(rand_vals)
    z = (actual - rand_mean) / rand_std if rand_std > 0 else 0
    direction = (
        "⬆ OVER-REPRESENTED"
        if z > 2
        else ("⬇ UNDER-REPRESENTED" if z < -2 else "≈ expected")
    )
    report.append(
        f"| {code} | {actual} | {rand_mean:.1f} | {rand_std:.1f} | {z:+.2f} | {direction} |"
    )
report.append("")

# Open vs Closed
open_triplets = tc.get("201", 0)
closed_triads = tc.get("300", 0)
report.append("## 3. Open Triplets vs Closed Triangles\n")
report.append(f"- **Open triplets (V-shapes / paths of 2):** {open_triplets}")
report.append(f"- **Closed triangles (fully connected trios):** {closed_triads}")
report.append(f"- **Transitivity:** {transitivity:.4f}")
report.append(
    f"- **Average clustering coefficient:** {nx.average_clustering(G_uw):.4f}"
)
report.append("")

# Top motifs examples
report.append("## 4. Top Motif Examples\n")
report.append("### 4.1 Closed Triangles (Fully Connected Trios)\n")
report.append("| Rank | Trio | Σ Jaccard | Shared Context |")
report.append("|------|------|-----------|----------------|")
for i, (clique, w_sum, edges_in) in enumerate(triangles_list[:3]):
    names = [node_name.get(n, n) for n in clique]
    # Find shared items common to all 3
    sets = [
        person_to_places.get(n, set()) | person_to_cultures.get(n, set())
        for n in clique
    ]
    common_all = sets[0] & sets[1] & sets[2]
    common_names = [node_name.get(x, x) for x in list(common_all)[:3]]
    report.append(
        f"| {i + 1} | {', '.join(names)} | {w_sum:.4f} | {', '.join(common_names)} |"
    )

report.append("\n### 4.2 Open Triplets (Hub-Spoke V-shapes)\n")
report.append("| Rank | Structure | Hub | Σ Weight | Shared Context |")
report.append("|------|-----------|-----|----------|----------------|")
for i, (hub, a, b, w1, w2, w_sum) in enumerate(open_triplet_list[:3]):
    hub_name = node_name.get(hub, hub)
    a_name = node_name.get(a, a)
    b_name = node_name.get(b, b)
    hub_set = person_to_places.get(hub, set()) | person_to_cultures.get(hub, set())
    a_set = person_to_places.get(a, set()) | person_to_cultures.get(a, set())
    b_set = person_to_places.get(b, set()) | person_to_cultures.get(b, set())
    shared_items = list((hub_set & a_set) | (hub_set & b_set))[:3]
    shared_names = [node_name.get(x, x) for x in shared_items]
    report.append(
        f"| {i + 1} | {a_name} ← {hub_name} → {b_name} | {hub_name} | {w_sum:.4f} | {', '.join(shared_names)} |"
    )

report.append("\n### 4.3 Stars (High-Degree Hubs)\n")
report.append("| Rank | Hub | Degree | Top Neighbors |")
report.append("|------|-----|--------|---------------|")
for hub, deg in top_hubs[:5]:
    hub_name = node_name.get(hub, hub)
    neighbors = list(G.neighbors(hub))
    neighbor_names = [node_name.get(n, n) for n in neighbors[:5]]
    report.append(
        f"| {top_hubs.index((hub, deg)) + 1} | {hub_name} | {deg} | {', '.join(neighbor_names)} |"
    )

# Per-community breakdown
report.append("\n## 5. Per-Community Motif Breakdown\n")
report.append(
    "Communities detected via Louvain algorithm on the Jaccard-weighted network.\n"
)
report.append(
    "| Community | Size | Triangles | Open Triplets | Transitivity | Avg Clustering |"
)
report.append(
    "|-----------|------|-----------|---------------|--------------|----------------|"
)
for c, members in sorted(communities.items(), key=lambda x: -len(x[1])):
    if len(members) < 3:
        report.append(f"| C{c} | {len(members)} | — | — | — | — |")
        continue
    sub_g = G_uw.subgraph(members).copy()
    sub_dir = nx.DiGraph()
    sub_dir.add_nodes_from(sub_g.nodes())
    for u, v in sub_g.edges():
        sub_dir.add_edge(u, v)
        sub_dir.add_edge(v, u)
    stc = nx.triadic_census(sub_dir)
    tri = stc.get("300", 0)
    opn = stc.get("201", 0)
    trans = 3 * tri / (3 * tri + opn) if (tri + opn) > 0 else 0
    avg_cc = nx.average_clustering(sub_g) if sub_g.number_of_nodes() > 0 else 0
    report.append(
        f"| C{c} | {len(members)} | {tri} | {opn} | {trans:.4f} | {avg_cc:.4f} |"
    )

# Community composition details
report.append("\n### Community Compositions\n")
for c, members in sorted(communities.items(), key=lambda x: -len(x[1])):
    names = [node_name.get(m, m) for m in members]
    report.append(f"- **C{c}** ({len(members)} persons): {', '.join(names)}")

# Inter-community edges
report.append("\n## 6. Inter-Community Edges\n")
report.append("| Community Pair | Edge Count | Example Bridge Persons |")
report.append("|---------------|------------|------------------------|")
for (c1, c2), cnt in sorted(inter_edges.items(), key=lambda x: -x[1]):
    names1 = [node_name.get(m, m) for m in communities[c1][:2]]
    names2 = [node_name.get(m, m) for m in communities[c2][:2]]
    report.append(
        f"| C{c1} ↔ C{c2} | {cnt} | {', '.join(names1)} ↔ {', '.join(names2)} |"
    )

# Key findings
report.append("\n## 7. Key Findings\n")
report.append("### 7.1 Motif Over/Under-representation\n")
for code in ["003", "102", "201", "300"]:
    actual = tc.get(code, 0)
    rand_vals = random_triads.get(code, [0])
    rand_mean = np.mean(rand_vals)
    rand_std = np.std(rand_vals)
    z = (actual - rand_mean) / rand_std if rand_std > 0 else 0
    label, _ = UNDIRECTED_MAP[code]
    if z > 2:
        report.append(
            f"- **{code} ({label}): OVER-REPRESENTED** (z={z:+.2f}) — more of these than expected by chance"
        )
    elif z < -2:
        report.append(
            f"- **{code} ({label}): UNDER-REPRESENTED** (z={z:+.2f}) — fewer of these than expected by chance"
        )
    else:
        report.append(f"- **{code} ({label}):** at expected level (z={z:+.2f})")

report.append("\n### 7.2 Pattern Summary\n")
if closed_triads > open_triplets:
    report.append(
        f"The network is **triangle-rich**: closed triads ({closed_triads}) outnumber open triplets ({open_triplets})."
    )
    report.append(
        "This suggests tightly-knit clusters of scholars who all knew each other or shared the same locations/teachings."
    )
else:
    report.append(
        f"The network is **hub-spoke dominant**: open triplets ({open_triplets}) outnumber closed triads ({closed_triads})."
    )
    report.append(
        "This suggests a star-like structure around key figures who connect otherwise unconnected persons."
    )

report.append(f"\nTransitivity = {transitivity:.4f} indicates that ")
if transitivity > 0.3:
    report.append(
        "the network has significant clustering — 'friends of friends are often friends.'"
    )
elif transitivity > 0.1:
    report.append(
        "moderate clustering — some but not strong triangle closing tendency."
    )
else:
    report.append(
        "low clustering — connections tend to be hub-centric rather than community-wide."
    )

report.append("\n### 7.3 Community Patterns\n")
# Find which communities have highest transitivity
comm_trans = {}
for c, members in communities.items():
    if len(members) >= 3:
        sub_g = G_uw.subgraph(members).copy()
        if sub_g.number_of_edges() >= 3:
            sub_dir = nx.DiGraph()
            sub_dir.add_nodes_from(sub_g.nodes())
            for u, v in sub_g.edges():
                sub_dir.add_edge(u, v)
                sub_dir.add_edge(v, u)
            stc = nx.triadic_census(sub_dir)
            tri = stc.get("300", 0)
            opn = stc.get("201", 0)
            trans = 3 * tri / (3 * tri + opn) if (tri + opn) > 0 else 0
            comm_trans[c] = (trans, len(members))

if comm_trans:
    most_cohesive = max(comm_trans.items(), key=lambda x: x[1][0])
    report.append(
        f"- Community C{most_cohesive[0]} has the highest transitivity ({most_cohesive[1][0]:.4f}), "
    )
    report.append(
        f"  suggesting it represents a tightly-knit group of {most_cohesive[1][1]} scholars."
    )

# Save report
report_path = os.path.join(OUT_DIR, "deep_motif_analysis.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"\nReport saved to: {report_path}")
print("Done!")
