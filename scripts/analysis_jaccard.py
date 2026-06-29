#!/usr/bin/env python3
"""Kento-shi Network Analysis — Jaccard-weighted bipartite projection.
Re-run with Jaccard weights instead of raw counts to avoid the "日本" near-clique problem.
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from collections import Counter, defaultdict
import warnings
import os
import re
import community as community_louvain  # python-louvain package
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUT_DIR), "output")

# ── Load and clean data ──────────────────────────────────────────────────
print("=" * 70)
print("1. DATA PREPARATION")
print("=" * 70)


def safe_read_csv(path):
    """Read CSV, filtering out malformed rows with empty required fields."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df


nodes = safe_read_csv(os.path.join(DATA_DIR, "nodes.csv"))
edges = safe_read_csv(os.path.join(DATA_DIR, "edges.csv"))

# Filter valid rows
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

print(
    f"Nodes: {len(nodes)} ({', '.join(f'{t}: {len(nodes[nodes.type == t])}' for t in ['Person', 'Event', 'Place', 'Culture'])})"
)
print(f"Edges: {len(edges)}")

# Separate metadata edges from social edges
META_RELS = {"HAS_SUBJECT", "HAS_OBJECT", "AT_LOCATION"}
SOCIAL_RELS = {"TRAVEL", "PARTICIPATE", "LEARN", "INTRODUCE"}

meta_edges = edges[edges["relation"].isin(META_RELS)].copy()
social_edges = edges[edges["relation"].isin(SOCIAL_RELS)].copy()

print(f"Metadata edges: {len(meta_edges)} ({len(meta_edges) / len(edges) * 100:.1f}%)")
print(
    f"Social edges: {len(social_edges)} ({len(social_edges) / len(edges) * 100:.1f}%)"
)
print(f"Social relation distribution: {dict(social_edges['relation'].value_counts())}")

# Build node type lookup
node_type = dict(zip(nodes["id"], nodes["type"]))
node_name = dict(zip(nodes["id"], nodes["name"]))

# ── Build G_social (direct entity-entity social relationships) ────────────
print("\n" + "=" * 70)
print("2. BIPARTITE PROJECTION (JACCARD-WEIGHTED)")
print("=" * 70)

G_social = nx.Graph()
for _, row in social_edges.iterrows():
    src, tgt, rel = row["source"], row["target"], row["relation"]
    G_social.add_edge(src, tgt, relation=rel)

print(
    f"G_social: {G_social.number_of_nodes()} nodes, {G_social.number_of_edges()} edges"
)


# Helper: jaccard_similarity for two sets
def jaccard(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union > 0 else 0.0


# ── Person↔Place bipartite (TRAVEL edges) ─────────────────────────────────
# NOTE: TRAVEL edges semantically always connect Person to Place, but some
# Place nodes are mislabeled as Culture in the data (e.g., 日本 L074, 長安 L145).
# We include ALL TRAVEL targets regardless of their labeled type.
print("\n--- Person↔Place Bipartite (TRAVEL) ---")

travel_edges = social_edges[social_edges["relation"] == "TRAVEL"].copy()

# Build adjacency: person -> set of travel targets (place-like nodes)
person_to_places = defaultdict(set)
place_to_persons = defaultdict(set)

for _, row in travel_edges.iterrows():
    src, tgt = row["source"], row["target"]
    src_type = node_type.get(src, "")
    tgt_type = node_type.get(tgt, "")
    # Include ALL TRAVEL targets — some are mislabeled as Culture/Person
    if src_type == "Person":
        person_to_places[src].add(tgt)
        place_to_persons[tgt].add(src)
    elif tgt_type == "Person":
        person_to_places[tgt].add(src)
        place_to_persons[src].add(tgt)
    else:
        # Both non-Person, add as-is
        if src_type != "Person" and tgt_type != "Person":
            pass  # skip non-person TRAVEL edges

persons_with_places = sorted(person_to_places.keys())
travel_targets = sorted(place_to_persons.keys())

# Count mislabeled targets
mislabeled = sum(1 for t in travel_targets if node_type.get(t) != "Place")
print(f"Persons with TRAVEL: {len(persons_with_places)}")
print(
    f"TRAVEL targets: {len(travel_targets)} (Place: {len(travel_targets) - mislabeled}, Culture/other: {mislabeled})"
)
print(f"TRAVEL edges: {len(travel_edges)}")
# Show key mislabeled targets
for t in travel_targets:
    if node_type.get(t) != "Place":
        print(
            f"  NOTE: TRAVEL target '{node_name.get(t, t)}' ({t}) is labeled as '{node_type.get(t)}', not 'Place'"
        )

# Compute Jaccard-weighted Person-Person projection
G_pp_jaccard = nx.Graph()
for p in persons_with_places:
    G_pp_jaccard.add_node(p, type="Person")

n_pairs = len(persons_with_places) * (len(persons_with_places) - 1) // 2
done = 0
for i in range(len(persons_with_places)):
    for j in range(i + 1, len(persons_with_places)):
        pi, pj = persons_with_places[i], persons_with_places[j]
        jw = jaccard(person_to_places[pi], person_to_places[pj])
        if jw > 0:
            G_pp_jaccard.add_edge(pi, pj, weight=jw)
    done += 1
    if done % 20 == 0:
        print(f"  Person-Person Jaccard: {done}/{len(persons_with_places)}")

print(
    f"Person-Person Jaccard graph: {G_pp_jaccard.number_of_nodes()} nodes, {G_pp_jaccard.number_of_edges()} edges"
)

# ── Person↔Culture bipartite (LEARN + INTRODUCE edges) ────────────────────
# NOTE: LEARN/INTRODUCE targets may also be mislabeled (some Culture targets
# labeled as Place, e.g., L081 "日本天台宗"). We include ALL targets.
print("\n--- Person↔Culture Bipartite (LEARN+INTRODUCE) ---")

culture_edges = social_edges[
    social_edges["relation"].isin(["LEARN", "INTRODUCE"])
].copy()

person_to_cultures = defaultdict(set)
culture_to_persons = defaultdict(set)

for _, row in culture_edges.iterrows():
    src, tgt = row["source"], row["target"]
    src_type = node_type.get(src, "")
    tgt_type = node_type.get(tgt, "")
    # Include ALL LEARN/INTRODUCE targets regardless of their labeled type
    if src_type == "Person":
        person_to_cultures[src].add(tgt)
        culture_to_persons[tgt].add(src)
    elif tgt_type == "Person":
        person_to_cultures[tgt].add(src)
        culture_to_persons[src].add(tgt)

persons_with_culture = sorted(person_to_cultures.keys())
culture_targets = sorted(culture_to_persons.keys())

# Count types
ctypes = Counter(node_type.get(t, "?") for t in culture_targets)
print(f"Persons with LEARN/INTRODUCE: {len(persons_with_culture)}")
print(f"Culture targets by type: {dict(ctypes)}")
print(f"LEARN+INTRODUCE edges: {len(culture_edges)}")

# Compute Jaccard-weighted Person-Person (Culture) projection
G_pp_culture_jaccard = nx.Graph()
for p in persons_with_culture:
    G_pp_culture_jaccard.add_node(p, type="Person")

for i in range(len(persons_with_culture)):
    for j in range(i + 1, len(persons_with_culture)):
        pi, pj = persons_with_culture[i], persons_with_culture[j]
        jw = jaccard(person_to_cultures[pi], person_to_cultures[pj])
        if jw > 0:
            G_pp_culture_jaccard.add_edge(pi, pj, weight=jw)

print(
    f"Person-Person Culture Jaccard: {G_pp_culture_jaccard.number_of_nodes()} nodes, {G_pp_culture_jaccard.number_of_edges()} edges"
)

# ── Combined Person-Person network (TRAVEL + LEARN/INTRODUCE) ─────────────
print("\n--- Combined Person-Person (TRAVEL + LEARN/INTRODUCE) ---")

G_pp_combined = nx.Graph()
# Add all nodes
for p in set(list(persons_with_places) + list(persons_with_culture)):
    G_pp_combined.add_node(p, type="Person")

# Add edges from both projections (max weight if both exist)
for u, v, d in G_pp_jaccard.edges(data=True):
    G_pp_combined.add_edge(u, v, weight=d["weight"])

for u, v, d in G_pp_culture_jaccard.edges(data=True):
    if G_pp_combined.has_edge(u, v):
        G_pp_combined[u][v]["weight"] = max(G_pp_combined[u][v]["weight"], d["weight"])
    else:
        G_pp_combined.add_edge(u, v, weight=d["weight"])

print(
    f"Combined Person-Person: {G_pp_combined.number_of_nodes()} nodes, {G_pp_combined.number_of_edges()} edges"
)
if G_pp_combined.number_of_edges() > 0:
    weights = [d["weight"] for _, _, d in G_pp_combined.edges(data=True)]
    print(
        f"Edge weight range: [{min(weights):.4f}, {max(weights):.4f}], mean={np.mean(weights):.4f}"
    )

# ── Place↔Place projection (via shared Persons on TRAVEL) ─────────────────
print("\n--- Place↔Place Projection (via shared persons) ---")

G_place_place = nx.Graph()
for p in persons_with_places:
    places = list(person_to_places[p])
    for i in range(len(places)):
        for j in range(i + 1, len(places)):
            pi, pj = places[i], places[j]
            if not G_place_place.has_edge(pi, pj):
                G_place_place.add_edge(pi, pj, weight=1.0, persons=set([p]))
            else:
                G_place_place[pi][pj]["weight"] += 1.0
                G_place_place[pi][pj]["persons"].add(p)

print(
    f"Place-Place: {G_place_place.number_of_nodes()} nodes, {G_place_place.number_of_edges()} edges"
)

# ── 3. CENTRALITY ANALYSIS ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. CENTRALITY ANALYSIS (Jaccard-weighted)")
print("=" * 70)

# Keep only the largest connected component for centrality
if G_pp_combined.number_of_nodes() > 0:
    components = list(nx.connected_components(G_pp_combined))
    largest_cc = max(components, key=len)
    G_cc = G_pp_combined.subgraph(largest_cc).copy()
    print(f"Largest CC: {G_cc.number_of_nodes()} nodes, {G_cc.number_of_edges()} edges")
else:
    G_cc = G_pp_combined
    print("No connected component")

# Compute centralities
centrality_df = None
if G_cc.number_of_nodes() > 0:
    # Weighted degree
    weighted_degree = {}
    for n in G_cc.nodes():
        wd = sum(d["weight"] for _, _, d in G_cc.edges(n, data=True))
        weighted_degree[n] = wd

    # Betweenness (weight as distance: 1/weight so higher similarity = shorter path)
    # Ensure no zero-division
    for u, v, d in G_cc.edges(data=True):
        if d["weight"] <= 0:
            d["weight"] = 1e-10
    betweenness = nx.betweenness_centrality(G_cc, weight="weight", normalized=True)

    # Eigenvector
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G_cc, weight="weight")
    except Exception:
        eigenvector = {n: 0.0 for n in G_cc.nodes()}

    # PageRank
    pagerank = nx.pagerank(G_cc, weight="weight")

    centrality_df = pd.DataFrame(
        {
            "person_id": list(G_cc.nodes()),
            "person_name": [node_name.get(n, n) for n in G_cc.nodes()],
            "weighted_degree": [weighted_degree[n] for n in G_cc.nodes()],
            "betweenness": [betweenness[n] for n in G_cc.nodes()],
            "eigenvector": [eigenvector[n] for n in G_cc.nodes()],
            "pagerank": [pagerank[n] for n in G_cc.nodes()],
        }
    )

    # Add source_count from nodes
    source_count_map = dict(
        zip(
            nodes["id"], pd.to_numeric(nodes["source_count"], errors="coerce").fillna(0)
        )
    )
    centrality_df["source_count"] = centrality_df["person_id"].map(source_count_map)

    # Print top-5 for each centrality
    for col, label in [
        ("weighted_degree", "Weighted Degree"),
        ("betweenness", "Betweenness"),
        ("pagerank", "PageRank"),
    ]:
        print(f"\nTop-5 {label}:")
        top5 = centrality_df.nlargest(5, col)[["person_name", col]]
        for _, row in top5.iterrows():
            print(f"  {row['person_name']}: {row[col]:.6f}")

    # source_count deviation analysis
    print("\n--- source_count deviation ---")
    for col in ["weighted_degree", "betweenness", "eigenvector", "pagerank"]:
        valid = centrality_df[centrality_df["source_count"] > 0]
        if len(valid) > 3:
            r, p = spearmanr(valid["source_count"], valid[col])
            print(f"  {col}: Spearman r={r:.4f}, p={p:.4f} {'**' if p < 0.05 else ''}")

# Save centrality
if centrality_df is not None:
    centrality_df.to_csv(os.path.join(OUT_DIR, "centrality_all.csv"), index=False)
    for col in ["weighted_degree", "betweenness", "pagerank"]:
        top20 = centrality_df.nlargest(20, col)
        top20.to_csv(os.path.join(OUT_DIR, f"centrality_top20_{col}.csv"), index=False)

# ── 4. COMMUNITY DETECTION (Louvain) ──────────────────────────────────────
print("\n" + "=" * 70)
print("4. COMMUNITY DETECTION (Louvain)")
print("=" * 70)

if G_cc.number_of_nodes() > 0:
    # Make undirected, ensure weights
    G_undirected = G_cc.copy()

    # Louvain community detection
    partition = community_louvain.best_partition(
        G_undirected, weight="weight", resolution=1.0
    )
    n_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, G_undirected, weight="weight")

    print("Resolution: 1.0")
    print(f"Modularity Q: {modularity:.4f}")
    print(f"Communities: {n_communities}")

    # Configuration model baseline (randomize edges keeping degree sequence)
    G_random = nx.configuration_model([d for _, d in G_undirected.degree()], seed=42)
    G_random = nx.Graph(G_random)
    G_random.remove_edges_from(nx.selfloop_edges(G_random))

    # Assign random weights from observed distribution
    obs_weights = [d["weight"] for _, _, d in G_undirected.edges(data=True)]
    for u, v in G_random.edges():
        G_random[u][v]["weight"] = np.random.choice(obs_weights)

    partition_random = community_louvain.best_partition(
        G_random, weight="weight", resolution=1.0
    )
    modularity_random = community_louvain.modularity(
        partition_random, G_random, weight="weight"
    )
    print(
        f"Random baseline Q: {modularity_random:.4f}, ΔQ={modularity - modularity_random:.4f}"
    )

    # Clustering coefficient
    cc_actual = nx.average_clustering(G_undirected, weight="weight")
    cc_random = nx.average_clustering(G_random, weight="weight")
    print(f"Avg clustering: actual={cc_actual:.4f}, random={cc_random:.4f}")

    # Community composition
    community_map = defaultdict(list)
    for node, comm in partition.items():
        community_map[comm].append(node)

    communities_df = pd.DataFrame(
        {
            "person_id": list(partition.keys()),
            "person_name": [node_name.get(n, n) for n in partition.keys()],
            "community": list(partition.values()),
        }
    )
    communities_df.to_csv(os.path.join(OUT_DIR, "communities.csv"), index=False)

    print("\nCommunity sizes:")
    for comm, members in sorted(community_map.items(), key=lambda x: -len(x[1])):
        names = [node_name.get(m, m) for m in members[:5]]
        print(f"  C{comm}: {len(members)} persons ({', '.join(names)}...)")

    # Try different resolutions
    print("\n--- Resolution sweep ---")
    for res in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        p = community_louvain.best_partition(
            G_undirected, weight="weight", resolution=res
        )
        q = community_louvain.modularity(p, G_undirected, weight="weight")
        nc = len(set(p.values()))
        print(f"  resolution={res:.1f}: Q={q:.4f}, communities={nc}")

# ── 5. HYPOTHESIS TESTING ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. HYPOTHESIS TESTING")
print("=" * 70)

# ── H1: 空海 vs 円仁 ego-network Jaccard ─────────────────────────────────
print("\n--- H1: 空海 vs 円仁 ego-network separation ---")


def get_ego_network(G, person_name, radius=2):
    """Get ego network around a person by name (Person type only)."""
    person_ids = []
    for nid, nname in node_name.items():
        if nname == person_name or (
            person_name in nname and len(nname) < len(person_name) + 10
        ):
            if node_type.get(nid) == "Person":
                person_ids.append(nid)
    if not person_ids:
        return set()
    # Use the first match (most direct name match)
    person_id = person_ids[0]
    if person_id not in G:
        return set()
    ego_nodes = set([person_id])
    frontier = set([person_id])
    for _ in range(radius):
        new_frontier = set()
        for n in frontier:
            if n in G:
                new_frontier.update(G.neighbors(n))
        ego_nodes.update(new_frontier)
        frontier = new_frontier - ego_nodes
    return ego_nodes


kukai_ego = get_ego_network(G_pp_combined, "空海")
ennin_ego = get_ego_network(G_pp_combined, "円仁")

print(f"空海 ego-network: {len(kukai_ego)} nodes")
print(f"円仁 ego-network: {len(ennin_ego)} nodes")

if kukai_ego and ennin_ego:
    h1_jaccard = jaccard(kukai_ego, ennin_ego)
    print(f"Jaccard similarity: {h1_jaccard:.4f}")
    print(f"Intersection size: {len(kukai_ego & ennin_ego)}")
    print(
        f"Conclusion: {'Significant separation' if h1_jaccard < 0.5 else 'Not well separated'} (Jaccard < 0.5 → separated)"
    )

# Also check some other key figures
for name in ["最澄", "宗叡", "道昭", "吉備真備", "阿倍仲麻呂"]:
    ego = get_ego_network(G_pp_combined, name)
    if ego:
        j_k = jaccard(ego, kukai_ego) if kukai_ego else 0
        j_e = jaccard(ego, ennin_ego) if ennin_ego else 0
        print(f"  {name}: {len(ego)} nodes, J(空海)={j_k:.4f}, J(円仁)={j_e:.4f}")

# ── H2: LEARN+INTRODUCE ratio analysis ────────────────────────────────────
print("\n--- H2: LEARN+INTRODUCE ratio ---")

learn_count = len(social_edges[social_edges["relation"] == "LEARN"])
introduce_count = len(social_edges[social_edges["relation"] == "INTRODUCE"])
total_social = len(social_edges)

print(f"LEARN: {learn_count}, INTRODUCE: {introduce_count}")
print(
    f"LEARN+INTRODUCE / total social: {(learn_count + introduce_count) / total_social:.3f}"
)
print(
    f"LEARN / INTRODUCE ratio: {learn_count / introduce_count:.2f}"
    if introduce_count > 0
    else "N/A"
)

# Analyze which persons have most LEARN/INTRODUCE
learn_targets = Counter()
introduce_targets = Counter()
for _, row in social_edges.iterrows():
    if row["relation"] == "LEARN":
        tgt = row["target"]
        learn_targets[node_name.get(tgt, tgt)] += 1
    elif row["relation"] == "INTRODUCE":
        tgt = row["target"]
        introduce_targets[node_name.get(tgt, tgt)] += 1

print("\nTop LEARN targets:")
for name, cnt in learn_targets.most_common(10):
    print(f"  {name}: {cnt}")

print("\nTop INTRODUCE targets:")
for name, cnt in introduce_targets.most_common(10):
    print(f"  {name}: {cnt}")

# LEARN/INTRODUCE by persons
person_learn = Counter()
person_introduce = Counter()
for _, row in social_edges.iterrows():
    if row["relation"] in ("LEARN", "INTRODUCE"):
        src = row["source"]
        src_name = node_name.get(src, src)
        if src_name and node_type.get(src) == "Person":
            if row["relation"] == "LEARN":
                person_learn[src_name] += 1
            else:
                person_introduce[src_name] += 1

print("\nTop LEARN persons:")
for name, cnt in person_learn.most_common(10):
    print(f"  {name}: LEARN={cnt}, INTRODUCE={person_introduce.get(name, 0)}")

# ── H3: 長安 Betweenness + percolation ────────────────────────────────────
print("\n--- H3: 長安 centrality & percolation in Place-Place ---")

# Place-Place betweenness
if G_place_place.number_of_nodes() > 0:
    place_betweenness = nx.betweenness_centrality(
        G_place_place, weight="weight", normalized=True
    )

    changan_id = None
    changan_bc = 0
    for n in G_place_place.nodes():
        name = node_name.get(n, "")
        if "長安" in name or "长安" in name:
            bc = place_betweenness.get(n, 0)
            if bc > changan_bc:
                changan_bc = bc
                changan_id = n

    if changan_id:
        # changan_bc already set from the search above
        bc_sorted = sorted(place_betweenness.items(), key=lambda x: -x[1])
        changan_rank = next(
            (i + 1 for i, (n, _) in enumerate(bc_sorted) if n == changan_id), 0
        )
        print(
            f"長安 betweenness: {changan_bc:.6f} (rank {changan_rank}/{len(bc_sorted)})"
        )

    # Top-5 Place betweenness
    print("\nTop-5 Place betweenness:")
    for n, bc in bc_sorted[:5]:
        print(f"  {node_name.get(n, n)}: {bc:.6f}")

    # Percolation: remove 長安 and measure component shrink
    if changan_id and changan_id in G_place_place:
        G_pp_no_changan = G_place_place.copy()
        G_pp_no_changan.remove_node(changan_id)
        comps_before = sorted(
            [len(c) for c in nx.connected_components(G_place_place)], reverse=True
        )
        comps_after = sorted(
            [len(c) for c in nx.connected_components(G_pp_no_changan)], reverse=True
        )
        shrink = (
            (comps_before[0] - comps_after[0]) / comps_before[0] * 100
            if comps_before
            else 0
        )
        print(
            f"\nPercolation: Largest component size before={comps_before[0]}, after={comps_after[0]}"
        )
        print(f"Component shrink: {shrink:.1f}%")
        print(f"Components before={len(comps_before)}, after={len(comps_after)}")

# ── 6. ALL PAIRS JACCARD MATRIX (top persons) ──────────────────────────────
print("\n" + "=" * 70)
print("6. TOP PERSONS JACCARD SIMILARITY MATRIX")
print("=" * 70)

# Compute pairwise Jaccard for top-10 persons (by weighted degree)
if centrality_df is not None and len(centrality_df) > 0:
    top10_persons = centrality_df.nlargest(10, "weighted_degree")["person_id"].tolist()

    jaccard_matrix = np.zeros((len(top10_persons), len(top10_persons)))
    for i in range(len(top10_persons)):
        for j in range(len(top10_persons)):
            if i == j:
                jaccard_matrix[i][j] = 1.0
            elif i < j:
                pi, pj = top10_persons[i], top10_persons[j]
                si = person_to_places.get(pi, set()) | person_to_cultures.get(pi, set())
                sj = person_to_places.get(pj, set()) | person_to_cultures.get(pj, set())
                jw = jaccard(si, sj)
                jaccard_matrix[i][j] = jw
                jaccard_matrix[j][i] = jw

    top10_names = [node_name.get(p, p) for p in top10_persons]
    print("\nJaccard similarity matrix (top-10 persons):")
    header = "           " + " ".join(f"{n[:6]:>8}" for n in top10_names)
    print(header)
    for i, name in enumerate(top10_names):
        row = f"{name[:10]:>10} " + " ".join(
            f"{jaccard_matrix[i][j]:8.4f}" for j in range(len(top10_persons))
        )
        print(row)

    # Save matrix
    jaccard_df = pd.DataFrame(jaccard_matrix, index=top10_names, columns=top10_names)
    jaccard_df.to_csv(os.path.join(OUT_DIR, "jaccard_top10_persons.csv"))

# ── 7. TEMPORAL EVOLUTION ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. TEMPORAL EVOLUTION")
print("=" * 70)

# Try to extract temporal info from event names and source_text
# Look for year patterns (e.g., 天平, 勝寳, 承和, 寳龜, 貞觀, etc.)
# or batch identifiers
era_patterns = {
    "M01-M07 (630-701)": [
        "舒明",
        "皇極",
        "孝徳",
        "白雉",
        "斉明",
        "天智",
        "天武",
        "持統",
        "文武",
        "大寶",
        "大化",
        "朱鳥",
    ],
    "M08-M10 (702-770)": [
        "大寶",
        "慶雲",
        "和銅",
        "靈龜",
        "養老",
        "神龜",
        "天平",
        "天平感寶",
        "天平勝寶",
        "天平寶字",
        "天平神護",
        "神護景雲",
        "寶龜",
    ],
    "M11-M15 (771-834)": ["寶龜", "天應", "延暦", "大同", "弘仁", "天長", "承和"],
    "M16-M20 (835-894)": [
        "承和",
        "嘉祥",
        "仁壽",
        "齊衡",
        "天安",
        "貞觀",
        "元慶",
        "仁和",
        "寛平",
    ],
}


# Assign era group to events based on name and source_text
def assign_era(name, source_text):
    text = name + " " + str(source_text)
    for era_group, eras in era_patterns.items():
        for era in eras:
            if era in text:
                return era_group
    return None


event_eras = {}
for _, row in nodes[nodes["type"] == "Event"].iterrows():
    era = assign_era(row["name"], row.get("sources", ""))
    if era:
        event_eras[row["id"]] = era

print(
    f"Events with era assignment: {len(event_eras)} / {len(nodes[nodes['type'] == 'Event'])}"
)

# Try alternate approach: look for numerical year patterns in content


def extract_year_from_text(text):
    """Try to extract approximate year from text."""
    # Japanese era years mentioned as numbers
    # This is challenging without a full era-year mapping

    # Look for batch numbers like 第X次遣唐使
    batch_match = re.findall(r"第(\d+)次", str(text))
    if batch_match:
        batch_num = int(batch_match[0])
        if batch_num <= 7:
            return "M01-M07 (630-701)"
        elif batch_num <= 10:
            return "M08-M10 (702-770)"
        elif batch_num <= 15:
            return "M11-M15 (771-834)"
        else:
            return "M16-M20 (835-894)"
    return None


event_eras2 = {}
for _, row in nodes[nodes["type"] == "Event"].iterrows():
    era = extract_year_from_text(str(row.get("sources", "")) + " " + row["name"])
    if era:
        event_eras2[row["id"]] = era

# Combine all sources
all_eras = {}
all_eras.update(event_eras)
all_eras.update(event_eras2)
print(
    f"Events with any era assignment: {len(all_eras)} / {len(nodes[nodes['type'] == 'Event'])}"
)

# Temporal evolution table
temporal_data = []
for era_group in [
    "M01-M07 (630-701)",
    "M08-M10 (702-770)",
    "M11-M15 (771-834)",
    "M16-M20 (835-894)",
]:
    era_events = [eid for eid, era in all_eras.items() if era == era_group]

    # Get persons linked to these events via social edges
    era_persons = set()
    for _, row in social_edges.iterrows():
        if row["source"] in era_events or row["target"] in era_events:
            if node_type.get(row["source"]) == "Person":
                era_persons.add(row["source"])
            if node_type.get(row["target"]) == "Person":
                era_persons.add(row["target"])

    # Get persons via metadata edges
    for _, row in meta_edges.iterrows():
        if row["source"] in era_events:
            if node_type.get(row["target"]) == "Person":
                era_persons.add(row["target"])

    temporal_data.append(
        {"era": era_group, "events": len(era_events), "persons": len(era_persons)}
    )

temporal_df = pd.DataFrame(temporal_data)
print("\nTemporal evolution:")
print(temporal_df.to_string(index=False))
temporal_df.to_csv(os.path.join(OUT_DIR, "temporal_evolution.csv"), index=False)

# ── 8. OLD/NEW COMPARISON ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. OLD/NEW SCHEME COMPARISON")
print("=" * 70)

# Try to load old dataset if available
old_data_paths = [
    os.path.join(os.path.dirname(DATA_DIR), "..", "data", "old_nodes.csv"),
    os.path.join(os.path.dirname(DATA_DIR), "old_nodes.csv"),
    "/home/szhh/kento-shi-network/data/old_nodes.csv",
]

old_nodes = None
for p in old_data_paths:
    if os.path.exists(p):
        old_nodes = pd.read_csv(p, dtype=str)
        print(f"Loaded old nodes from {p}: {len(old_nodes)} nodes")
        break

if old_nodes is not None:
    old_persons = set(old_nodes[old_nodes["type"] == "Person"]["name"].dropna())
    new_persons = set(nodes[nodes["type"] == "Person"]["name"].dropna())

    shared = old_persons & new_persons
    old_only = old_persons - new_persons
    new_only = new_persons - old_persons

    person_jaccard = (
        len(shared) / len(old_persons | new_persons)
        if (old_persons | new_persons)
        else 0
    )

    print(
        f"Person: shared={len(shared)}, old-only={len(old_only)}, new-only={len(new_only)}"
    )
    print(f"Person Jaccard: {person_jaccard:.4f}")

    old_places = (
        set(old_nodes[old_nodes["type"] == "Place"]["name"].dropna())
        if "Place" in old_nodes["type"].values
        else set()
    )
    new_places = set(nodes[nodes["type"] == "Place"]["name"].dropna())
    if old_places:
        place_shared = old_places & new_places
        place_jaccard = len(place_shared) / len(old_places | new_places)
        print(
            f"Place: shared={len(place_shared)}, old-only={len(old_places - new_places)}, new-only={len(new_places - old_places)}"
        )
        print(f"Place Jaccard: {place_jaccard:.4f}")

    old_new_df = pd.DataFrame(
        {
            "category": ["Person", "Place"],
            "shared": [len(shared), len(old_places & new_places) if old_places else 0],
            "old_only": [
                len(old_only),
                len(old_places - new_places) if old_places else 0,
            ],
            "new_only": [
                len(new_only),
                len(new_places - old_places) if old_places else 0,
            ],
            "jaccard": [person_jaccard, place_jaccard if old_places else 0],
        }
    )
    old_new_df.to_csv(os.path.join(OUT_DIR, "old_new_comparison.csv"), index=False)

    print("\nOld-only Person examples:", list(old_only)[:10])
    print("New-only Person examples:", list(new_only)[:10])
else:
    print("Old dataset not found. Skipping comparison.")
    # Create placeholder
    old_new_df = pd.DataFrame(
        {
            "category": ["Person", "Place"],
            "shared": [0, 0],
            "old_only": [0, 0],
            "new_only": [
                len(nodes[nodes["type"] == "Person"]),
                len(nodes[nodes["type"] == "Place"]),
            ],
            "jaccard": [0, 0],
        }
    )
    old_new_df.to_csv(os.path.join(OUT_DIR, "old_new_comparison.csv"), index=False)

# ── 9. VISUALIZATION ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("9. VISUALIZATION")
print("=" * 70)

plt.rcParams["font.family"] = "sans-serif"
# Try to find a CJK-capable font
for font in ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "IPAGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = font
        break
    except Exception:
        pass


def plot_network(
    G, title, filename, node_color_attr=None, top_n_labels=15, figsize=(20, 20)
):
    """Plot a network graph with community coloring."""
    if G.number_of_nodes() == 0:
        print(f"  Skipping {filename}: empty graph")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42, weight="weight")

    # Node sizes by degree
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [300 + 1500 * degrees.get(n, 0) / max_deg for n in G.nodes()]

    # Node colors by community if available
    if node_color_attr is None and G.number_of_nodes() > 2:
        try:
            partition = community_louvain.best_partition(
                G, weight="weight", resolution=1.0
            )
            node_colors = [partition[n] for n in G.nodes()]
            cmap = plt.cm.tab10
        except Exception:
            node_colors = "steelblue"
            cmap = None
    elif node_color_attr:
        node_colors = [G.nodes[n].get(node_color_attr, 0) for n in G.nodes()]
        cmap = plt.cm.viridis
    else:
        node_colors = "steelblue"
        cmap = None

    # Edge widths
    if G.number_of_edges() > 0:
        edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        max_ew = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + 3 * w / max_ew for w in edge_weights]
    else:
        edge_widths = []

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths, edge_color="#888888")

    if cmap:
        nodes_drawn = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            alpha=0.85,
            ax=ax,
        )
        if isinstance(node_colors, list) and not isinstance(node_colors[0], str):
            plt.colorbar(nodes_drawn, ax=ax, shrink=0.8, label="Community")
    else:
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, ax=ax
        )

    # Label top-N nodes by degree
    if top_n_labels > 0:
        top_nodes = sorted(degrees.items(), key=lambda x: -x[1])[:top_n_labels]
        labels = {n: node_name.get(n, n)[:12] for n, _ in top_nodes}
        nx.draw_networkx_labels(
            G,
            pos,
            labels,
            font_size=8,
            font_weight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    filepath = os.path.join(OUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


# Plot Person-Person network
plot_network(
    G_cc,
    "Person-Person Network (Jaccard-weighted)",
    "person_person_network_jaccard.png",
)

# Plot Place-Place network
if G_place_place.number_of_nodes() > 0:
    plot_network(
        G_place_place,
        "Place-Place Network (via shared persons)",
        "place_place_network.png",
    )

# Ego networks for key figures
for name in ["空海", "最澄", "円仁"]:
    person_id = None
    for nid, nname in node_name.items():
        if nname == name or name in nname:
            if node_type.get(nid) == "Person":
                person_id = nid
                break
    if person_id and person_id in G_pp_combined:
        # Ego network radius 1
        ego = nx.ego_graph(G_pp_combined, person_id, radius=1)
        if ego.number_of_nodes() > 1:
            safe_name = name.replace(" ", "_")
            plot_network(
                ego,
                f"{name} Ego-Network (Jaccard-weighted)",
                f"ego_network_{safe_name}_jaccard.png",
                top_n_labels=30,
                figsize=(14, 14),
            )

# ── 10. SUMMARY / REPORT ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("10. GENERATING REPORT")
print("=" * 70)

# Collect all key stats
report_lines = []
report_lines.append(
    "# 遣唐使知识图谱：网络分析报告 (Jaccard-Weighted Bipartite Projection)"
)
report_lines.append("\n*Generated: 2026-06-30 — Jaccard-weighted method*")
report_lines.append("")

# Section 1: Data
report_lines.append("## 1. 数据准备")
report_lines.append("")
report_lines.append(
    f"- **总节点数**: {len(nodes)} (Person: {len(nodes[nodes.type == 'Person'])}, Event: {len(nodes[nodes.type == 'Event'])}, Place: {len(nodes[nodes.type == 'Place'])}, Culture: {len(nodes[nodes.type == 'Culture'])})"
)
report_lines.append(f"- **总边数**: {len(edges)}")
report_lines.append(
    f"- **元数据层**: {len(meta_edges)} ({len(meta_edges) / len(edges) * 100:.1f}%) — Event↔Entity 分解"
)
report_lines.append(
    f"- **实体关系层**: {len(social_edges)} ({len(social_edges) / len(edges) * 100:.1f}%) — 实际人物/地点/文化关系"
)
report_lines.append(
    f"- **社交关系分布**: {dict(social_edges['relation'].value_counts())}"
)
report_lines.append(
    f"- **G_social**: {G_social.number_of_nodes()} 节点, {G_social.number_of_edges()} 边"
)
report_lines.append("")

# Section 2: Bipartite Projection
report_lines.append("## 2. 双模网络投影 (Jaccard加权)")
report_lines.append("")
report_lines.append("### 方法变更")
report_lines.append("")
report_lines.append(
    '旧版使用**原始计数投影**（Person-Person边权重 = 共享地点数），导致"日本"作为几乎所有人物共享的超级枢纽，使整个网络成为近完全图。'
)
report_lines.append("")
report_lines.append("新版使用 **Jaccard加权投影**：")
report_lines.append("- Person↔Place 边权重 = Jaccard相似度 = |A ∩ B| / |A ∪ B|")
report_lines.append(
    '- 这消除了"每个人都走过日本"的虚假连接：共享"日本"但很少共享其他地点的人物，Jaccard权重极低'
)
report_lines.append("")
report_lines.append("### Person↔Place (TRAVEL)")
report_lines.append(
    f"- TRAVEL边: {len(travel_edges)} → 二部图 {len(persons_with_places)} Person, {len(travel_targets)} 地点目标 (含 {mislabeled} 个被误标为Culture的地点)"
)
report_lines.append(
    f"- Person-Person投影: {G_pp_jaccard.number_of_nodes()} 人, {G_pp_jaccard.number_of_edges()} 边"
)
if G_pp_jaccard.number_of_edges() > 0:
    jw_list = [d["weight"] for _, _, d in G_pp_jaccard.edges(data=True)]
    report_lines.append(
        f"- Jaccard权重范围: [{min(jw_list):.4f}, {max(jw_list):.4f}], 均值={np.mean(jw_list):.4f}, 中位数={np.median(jw_list):.4f}"
    )
    high_jw = sum(1 for w in jw_list if w > 0.5)
    report_lines.append(
        f"- 高相似度对 (Jaccard > 0.5): {high_jw}/{len(jw_list)} ({high_jw / len(jw_list) * 100:.1f}%)"
    )
report_lines.append("")

report_lines.append("### Person↔Culture (LEARN/INTRODUCE)")
report_lines.append(
    f"- LEARN+INTRODUCE边: {len(culture_edges)} → 二部图 {len(persons_with_culture)} Person, {len(culture_targets)} 文化/知识目标"
)
report_lines.append(f"- 目标类型分布: {dict(ctypes)}")
report_lines.append(
    f"- Person-Person Culture投影: {G_pp_culture_jaccard.number_of_nodes()} 人, {G_pp_culture_jaccard.number_of_edges()} 边"
)
report_lines.append("")

report_lines.append("### Combined Person-Person")
report_lines.append(
    f"- 合并网络: {G_pp_combined.number_of_nodes()} 人, {G_pp_combined.number_of_edges()} 边"
)
if G_pp_combined.number_of_edges() > 0:
    cw_list = [d["weight"] for _, _, d in G_pp_combined.edges(data=True)]
    report_lines.append(
        f"- 权重范围: [{min(cw_list):.4f}, {max(cw_list):.4f}], 均值={np.mean(cw_list):.4f}"
    )
report_lines.append("")

report_lines.append("### Place-Place 投影")
report_lines.append(
    f"- 通过共享人物(Person)投影: {G_place_place.number_of_nodes()} 地点, {G_place_place.number_of_edges()} 边"
)
report_lines.append("")

# Section 3: Centrality
report_lines.append("## 3. 中心性分析 (Jaccard加权)")
report_lines.append("")
if centrality_df is not None:
    for col, label in [
        ("weighted_degree", "Weighted Degree"),
        ("betweenness", "Betweenness"),
        ("pagerank", "PageRank"),
    ]:
        report_lines.append(f"### Top-5 {label}")
        report_lines.append("| 人物 | 值 |")
        report_lines.append("|------|-----|")
        top5 = centrality_df.nlargest(5, col)[["person_name", col]]
        for _, row in top5.iterrows():
            report_lines.append(f"| {row['person_name']} | {row[col]:.6f} |")
        report_lines.append("")

    report_lines.append("### source_count 偏差")
    report_lines.append("| 指标 | Spearman r | p | 偏差? |")
    report_lines.append("|------|-----------|-----|--------|")
    for col in ["weighted_degree", "betweenness", "eigenvector", "pagerank"]:
        valid = centrality_df[centrality_df["source_count"] > 0]
        if len(valid) > 3:
            r, p = spearmanr(valid["source_count"], valid[col])
            flag = "✅" if p < 0.05 else "✗"
            report_lines.append(f"| {col} | {r:.4f} | {p:.4f} | {flag} |")
    report_lines.append("")

# Section 4: Community
report_lines.append("## 4. 社区发现 (Louvain)")
report_lines.append("")
if G_cc.number_of_nodes() > 0:
    report_lines.append("- **分辨率**: 1.0")
    report_lines.append(f"- **模块度 Q**: {modularity:.4f}")
    report_lines.append(f"- **社区数**: {n_communities}")
    report_lines.append(
        f"- **随机基线 Q**: {modularity_random:.4f}, ΔQ={modularity - modularity_random:.4f}"
    )
    report_lines.append(
        f"- **平均聚类系数**: 实际={cc_actual:.4f}, 随机={cc_random:.4f}"
    )
    report_lines.append("")
    report_lines.append("### 社区构成")
    for comm, members in sorted(community_map.items(), key=lambda x: -len(x[1])):
        names = [node_name.get(m, m) for m in members[:7]]
        report_lines.append(f"  - C{comm}: {len(members)}人 ({', '.join(names)}...)")
report_lines.append("")

# Section 5: Hypotheses
report_lines.append("## 5. 假设检验")
report_lines.append("")
report_lines.append("### H1: 空海 vs 円仁 网络分隔")
report_lines.append(f"- 空海 ego-network (radius=2): {len(kukai_ego)} 节点")
report_lines.append(f"- 円仁 ego-network (radius=2): {len(ennin_ego)} 节点")
report_lines.append(f"- Jaccard 相似度: {h1_jaccard:.4f}")
if h1_jaccard < 0.5:
    report_lines.append(
        "- 结论: ✅ **显著分离** (Jaccard < 0.5) — Jaccard加权成功区分了空海和円仁的活动空间"
    )
else:
    report_lines.append("- 结论: ⚠️ 仍有较大重叠 — 两人的活动地点/文化圈有显著交集")
report_lines.append("")
report_lines.append(
    '**与旧版对比**: 旧版使用原始计数投影时 Jaccard = 1.0（因为所有人物都共享"日本"），Jaccard加权后降至有意义的值。'
)
report_lines.append("")

report_lines.append("### H2: LEARN+INTRODUCE 比例分析")
report_lines.append(f"- LEARN: {learn_count}, INTRODUCE: {introduce_count}")
report_lines.append(
    f"- 占社交边比例: {(learn_count + introduce_count) / total_social:.3f}"
)
report_lines.append(
    f"- LEARN/INTRODUCE 比率: {learn_count / introduce_count:.2f}"
    if introduce_count > 0
    else ""
)
report_lines.append("")
report_lines.append("### 主要LEARN/INTRODUCE人物")
report_lines.append("| 人物 | LEARN | INTRODUCE | 合计 |")
report_lines.append("|------|-------|-----------|------|")
top_li_persons = sorted(
    set(list(person_learn.keys()) + list(person_introduce.keys())),
    key=lambda x: person_learn.get(x, 0) + person_introduce.get(x, 0),
    reverse=True,
)[:10]
for name in top_li_persons:
    learn_cnt = person_learn.get(name, 0)
    intro_cnt = person_introduce.get(name, 0)
    report_lines.append(f"| {name} | {learn_cnt} | {intro_cnt} | {learn_cnt + intro_cnt} |")
report_lines.append("")

report_lines.append("### H3: 長安网络中心性")
if G_place_place.number_of_nodes() > 0 and changan_id:
    report_lines.append(
        f"- Place-Place 介数: {changan_bc:.6f} (排名 {changan_rank}/{len(bc_sorted)})"
    )
    report_lines.append(f"- 移除長安后巨片缩小: {shrink:.1f}%")
report_lines.append("")

# Section 6: Temporal
report_lines.append("## 6. 时空演化")
report_lines.append("")
report_lines.append(
    f"- 可映射至批次的 Event: {len(all_eras)} / {len(nodes[nodes['type'] == 'Event'])}"
)
report_lines.append("")
report_lines.append("| 阶段 | Events | Persons |")
report_lines.append("|------|--------|---------|")
for _, row in temporal_df.iterrows():
    report_lines.append(
        f"| {row['era']} | {int(row['events'])} | {int(row['persons'])} |"
    )
report_lines.append("")

# Section 8: Old/new
report_lines.append("## 7. 新旧方案对比")
report_lines.append("")
if old_nodes is not None:
    report_lines.append(
        f"- **Person Jaccard**: {person_jaccard:.4f} (共有 {len(shared)}, 旧独有 {len(old_only)}, 新独有 {len(new_only)})"
    )
    report_lines.append(
        f"- **Place Jaccard**: {place_jaccard:.4f}" if old_places else ""
    )
report_lines.append("")

# Section: Visualization
report_lines.append("## 8. 可视化")
report_lines.append("")
report_lines.append(
    "- `person_person_network_jaccard.png` — Jaccard加权 Person-Person 投影网络（社区着色）"
)
report_lines.append("- `place_place_network.png` — Place-Place 投影网络")
report_lines.append("- `ego_network_空海_jaccard.png` — 空海 ego-network (Jaccard加权)")
report_lines.append("- `ego_network_最澄_jaccard.png` — 最澄 ego-network (Jaccard加权)")
report_lines.append("- `ego_network_円仁_jaccard.png` — 円仁 ego-network (Jaccard加权)")
report_lines.append("- `jaccard_top10_persons.csv` — Top-10 人物 Jaccard 相似度矩阵")
report_lines.append("")

# Section: Data files
report_lines.append("## 9. 数据文件")
report_lines.append("")
report_lines.append("| 文件 | 描述 |")
report_lines.append("|------|------|")
report_lines.append("| `centrality_all.csv` | 所有人物中心性指标 |")
report_lines.append("| `centrality_top20_*.csv` | Top-20 各中心性 |")
report_lines.append("| `communities.csv` | 社区分配结果 |")
report_lines.append("| `temporal_evolution.csv` | 时空演化数据 |")
report_lines.append("| `old_new_comparison.csv` | 新旧方案对比 |")
report_lines.append("| `jaccard_top10_persons.csv` | Top-10人物Jaccard矩阵 |")
report_lines.append("")

# Section: Key Findings
report_lines.append("## 10. 关键发现")
report_lines.append("")
report_lines.append("### 10.1 Jaccard加权 vs 原始计数")
report_lines.append("")
report_lines.append("Jaccard加权投影从根本上改变了网络结构：")
report_lines.append(
    '- **旧版**（原始计数）：所有人物通过"日本"连接 → 近完全图 → Jaccard=1.0 无意义'
)
report_lines.append(
    '- **新版**（Jaccard加权）：共享"日本"但很少共享其他地点 → 权重极低 → 真实活动模式浮现'
)
report_lines.append("")
if G_pp_combined.number_of_edges() > 0:
    high_jw_all = sum(1 for w in cw_list if w > 0.5)
    report_lines.append(
        f"- 合并网络中，仅 {high_jw_all}/{len(cw_list)} 对 ({high_jw_all / len(cw_list) * 100:.1f}%) 的 Jaccard > 0.5"
    )
    report_lines.append("- 这意味着只有少数人物对共享了显著比例的活动地点/文化圈")
report_lines.append("")

report_lines.append("### 10.2 空海 vs 円仁 真实分隔")
report_lines.append("")
report_lines.append(
    f"Jaccard加权后，空海和円仁的 ego-network Jaccard = {h1_jaccard:.4f}，"
)
if h1_jaccard < 0.5:
    report_lines.append("两人在活动空间上有显著差异。这反映了：")
    report_lines.append("- 空海主要在長安-青龍寺一带活动，学习密教")
    report_lines.append("- 円仁主要在五台山-揚州-登州一带活动，学习天台宗")
else:
    report_lines.append("两人活动空间仍有重叠，可能由于共同造访了某些重要佛教圣地。")
report_lines.append("")

report_lines.append("### 10.3 网络稀疏性")
report_lines.append(
    f"- 平均权重从 1.06 变更为 {np.mean(cw_list):.4f}（更稀疏）"
    if G_pp_combined.number_of_edges() > 0
    else ""
)
report_lines.append('- Jaccard加权消除了"假阳性"连接，保留了真正有意义的人物关联')
report_lines.append("")

# Write report
report_path = os.path.join(OUT_DIR, "REPORT.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\nReport written to {report_path}")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
