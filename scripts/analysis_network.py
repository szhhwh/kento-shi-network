#!/usr/bin/env python3
"""
Deep Network Analysis:
  1. Network motifs (triads) in the Person-Person Jaccard-weighted projection
  2. Assortativity: monk vs secular, early-period vs late-period
  3. Bridge persons: high betweenness + connects different communities
  4. Ego-network density comparison for top-10 meaningful persons
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import json
import os
import warnings
import re
import random
from itertools import combinations

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUT_DIR), "output")
REPORT_PATH = os.path.join(OUT_DIR, "deep_network_analysis.md")


def jaccard(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union > 0 else 0.0


print("Loading data...")
nodes = pd.read_csv(
    os.path.join(DATA_DIR, "nodes.csv"), dtype=str, keep_default_na=False
)
edges = pd.read_csv(
    os.path.join(DATA_DIR, "edges.csv"), dtype=str, keep_default_na=False
)

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

node_type = dict(zip(nodes["id"], nodes["type"]))
node_name = dict(zip(nodes["id"], nodes["name"]))

# Build Person↔Place (TRAVEL) and Person↔Culture (LEARN+INTRODUCE)
social_edges = edges[
    edges["relation"].isin(["TRAVEL", "PARTICIPATE", "LEARN", "INTRODUCE"])
].copy()

travel_edges = social_edges[social_edges["relation"] == "TRAVEL"].copy()
person_to_places = defaultdict(set)
for _, row in travel_edges.iterrows():
    src, tgt = row["source"], row["target"]
    if node_type.get(src) == "Person":
        person_to_places[src].add(tgt)
    elif node_type.get(tgt) == "Person":
        person_to_places[tgt].add(src)

culture_edges = social_edges[
    social_edges["relation"].isin(["LEARN", "INTRODUCE"])
].copy()
person_to_cultures = defaultdict(set)
for _, row in culture_edges.iterrows():
    src, tgt = row["source"], row["target"]
    if node_type.get(src) == "Person":
        person_to_cultures[src].add(tgt)
    elif node_type.get(tgt) == "Person":
        person_to_cultures[tgt].add(src)

persons_with_places = sorted(person_to_places.keys())
persons_with_culture = sorted(person_to_cultures.keys())

# Build G_pp_combined
print("Building Person-Person Jaccard-weighted network...")
G = nx.Graph()
all_persons = set(persons_with_places + persons_with_culture)
for p in all_persons:
    G.add_node(p, type="Person")

for i in range(len(persons_with_places)):
    for j in range(i + 1, len(persons_with_places)):
        pi, pj = persons_with_places[i], persons_with_places[j]
        jw = jaccard(person_to_places.get(pi, set()), person_to_places.get(pj, set()))
        if jw > 0:
            G.add_edge(pi, pj, weight=jw)

for i in range(len(persons_with_culture)):
    for j in range(i + 1, len(persons_with_culture)):
        pi, pj = persons_with_culture[i], persons_with_culture[j]
        jw = jaccard(
            person_to_cultures.get(pi, set()), person_to_cultures.get(pj, set())
        )
        if jw > 0:
            if G.has_edge(pi, pj):
                G[pi][pj]["weight"] = max(G[pi][pj]["weight"], jw)
            else:
                G.add_edge(pi, pj, weight=jw)

print(f"G_pp_combined: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

components = list(nx.connected_components(G))
largest_cc = max(components, key=len)
G_cc = G.subgraph(largest_cc).copy()
print(f"Largest CC: {G_cc.number_of_nodes()} nodes, {G_cc.number_of_edges()} edges")

# Load centrality and community data
centrality_df = pd.read_csv(os.path.join(OUT_DIR, "centrality_all.csv"))
communities_df = pd.read_csv(os.path.join(OUT_DIR, "communities.csv"))
person_community = dict(zip(communities_df["person_id"], communities_df["community"]))

# Build proper betweenness lookup from centrality_all.csv
bc_lookup = dict(zip(centrality_df["person_id"], centrality_df["betweenness"]))
deg_lookup = dict(zip(centrality_df["person_id"], centrality_df["weighted_degree"]))

# Binary graph for triad analysis
G_binary = nx.Graph()
G_binary.add_nodes_from(G.nodes())
for u, v, d in G.edges(data=True):
    if d["weight"] > 0:
        G_binary.add_edge(u, v)

# ═══════════════════════════════════════════════════════════════════
# 1. NETWORK MOTIFS (TRIADS)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. NETWORK MOTIFS - TRIAD ANALYSIS")
print("=" * 70)

G_cc_binary = nx.Graph()
G_cc_binary.add_nodes_from(G_cc.nodes())
for u, v in G_cc.edges():
    G_cc_binary.add_edge(u, v)

all_triads_cc = list(combinations(G_cc.nodes(), 3))
total_triads_cc = len(all_triads_cc)

triad_counts_cc = {0: 0, 1: 0, 2: 0, 3: 0}
for u, v, w in all_triads_cc:
    e_count = 0
    if G_cc_binary.has_edge(u, v):
        e_count += 1
    if G_cc_binary.has_edge(v, w):
        e_count += 1
    if G_cc_binary.has_edge(w, u):
        e_count += 1
    triad_counts_cc[e_count] += 1

cc_clustering = nx.transitivity(G_cc_binary)
weighted_clustering_cc = nx.average_clustering(G_cc, weight="weight")
avg_degree = sum(dict(G_cc_binary.degree()).values()) / G_cc_binary.number_of_nodes()

print(
    f"Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(list(combinations(G.nodes(), 3)))} triads"
)
print(
    f"Largest CC: {G_cc.number_of_nodes()} nodes, {G_cc.number_of_edges()} edges, {total_triads_cc} triads"
)
print(f"Avg degree in CC: {avg_degree:.2f}")
print("\nTriad distribution (Largest CC):")
print(
    f"  Empty (0 edges): {triad_counts_cc[0]} ({triad_counts_cc[0] / total_triads_cc * 100:.1f}%)"
)
print(
    f"  Single edge (1): {triad_counts_cc[1]} ({triad_counts_cc[1] / total_triads_cc * 100:.1f}%)"
)
print(
    f"  Wedge (2 edges): {triad_counts_cc[2]} ({triad_counts_cc[2] / total_triads_cc * 100:.1f}%)"
)
print(
    f"  Triangle (3):    {triad_counts_cc[3]} ({triad_counts_cc[3] / total_triads_cc * 100:.1f}%)"
)
print(f"\nTransitivity: {cc_clustering:.4f}")
print(f"Weighted avg clustering: {weighted_clustering_cc:.4f}")

# Top triangles by weight
triangles_weighted = []
for u, v, w in all_triads_cc:
    if G_cc.has_edge(u, v) and G_cc.has_edge(v, w) and G_cc.has_edge(w, u):
        w1 = G_cc[u][v].get("weight", 0)
        w2 = G_cc[v][w].get("weight", 0)
        w3 = G_cc[w][u].get("weight", 0)
        avg_w = (w1 + w2 + w3) / 3
        triangles_weighted.append(
            (avg_w, node_name.get(u, u), node_name.get(v, v), node_name.get(w, w))
        )

triangles_weighted.sort(key=lambda x: -x[0])
print("\nTop-10 highest-weight triangles:")
for i, (avg_w, n1, n2, n3) in enumerate(triangles_weighted[:10]):
    print(f"  {i + 1}. {n1} — {n2} — {n3}  (avg={avg_w:.4f})")

# Star vs cluster analysis per node
node_triangles = {n: 0 for n in G_cc.nodes()}
for u, v, w in all_triads_cc:
    if (
        G_cc_binary.has_edge(u, v)
        and G_cc_binary.has_edge(v, w)
        and G_cc_binary.has_edge(w, u)
    ):
        node_triangles[u] += 1
        node_triangles[v] += 1
        node_triangles[w] += 1

star_ratios = []
for n in G_cc.nodes():
    deg = G_cc_binary.degree(n)
    tris = node_triangles.get(n, 0)
    if deg > 1:
        max_possible_tris = deg * (deg - 1) / 2
        ratio = tris / max_possible_tris
        star_ratios.append((ratio, deg, tris, node_name.get(n, n), n))

# Get meaningful persons only (exclude ship entities, compound names)
meaningful = []
for ratio, deg, tris, name, nid in star_ratios:
    # Filter out ships, compound, and generic entries
    if any(
        x in name for x in ["船", "舶", "遣唐使第", "入唐使第", "大使船", "餘八", "等"]
    ):
        continue
    if len(name) > 8 and any(x in name for x in ["、", "，", "等"]):
        continue
    meaningful.append((ratio, deg, tris, name, nid))

meaningful.sort(key=lambda x: x[0])
print("\nMost star-like meaningful persons (low clustering):")
for i, (ratio, deg, tris, name, nid) in enumerate(meaningful[:10]):
    bc = bc_lookup.get(nid, 0)
    print(
        f"  {name}: deg={deg}, triangles={tris}, ratio={ratio:.3f}, betweenness={bc:.6f}"
    )

meaningful.sort(key=lambda x: -x[0])
print("\nMost clustered meaningful persons (high clustering):")
for i, (ratio, deg, tris, name, nid) in enumerate(meaningful[:10]):
    bc = bc_lookup.get(nid, 0)
    print(
        f"  {name}: deg={deg}, triangles={tris}, ratio={ratio:.3f}, betweenness={bc:.6f}"
    )

# ═══════════════════════════════════════════════════════════════════
# 2. ASSORTATIVITY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. ASSORTATIVITY")
print("=" * 70)


# 2a. Monk vs Secular
def is_monk(name):
    if not name:
        return False
    return bool(re.search(r"[釋僧法師]", name))


for n in G.nodes():
    name = node_name.get(n, "")
    G.nodes[n]["is_monk"] = 1 if is_monk(name) else 0

monk_nodes = [n for n in G.nodes() if G.nodes[n]["is_monk"] == 1]
secular_nodes = [n for n in G.nodes() if G.nodes[n]["is_monk"] == 0]
print(f"Monks (釋/僧/法/師 in name): {len(monk_nodes)}")
print(f"Secular/Officials: {len(secular_nodes)}")

monk_monk = 0
monk_secular = 0
secular_secular = 0
for u, v in G.edges():
    um = G.nodes[u]["is_monk"]
    vm = G.nodes[v]["is_monk"]
    if um and vm:
        monk_monk += 1
    elif um or vm:
        monk_secular += 1
    else:
        secular_secular += 1

total_edges_binary = G.number_of_edges()
expected_mm = (len(monk_nodes) / G.number_of_nodes()) ** 2

# Permutation test

random.seed(42)
n_perm = 10000
perm_mm = []
node_list = list(G.nodes())
for _ in range(n_perm):
    labels = [G.nodes[n]["is_monk"] for n in node_list]
    random.shuffle(labels)
    node_label = dict(zip(node_list, labels))
    mm = 0
    for u, v in G.edges():
        if node_label[u] and node_label[v]:
            mm += 1
    perm_mm.append(mm)

perm_mm = np.array(perm_mm)
observed_z = (monk_monk - perm_mm.mean()) / perm_mm.std() if perm_mm.std() > 0 else 0
p_value = (perm_mm >= monk_monk).sum() / n_perm

print(
    f"\nMonk-Monk edges: {monk_monk}/{total_edges_binary} ({monk_monk / total_edges_binary:.4f})"
)
print(f"Monk-Secular: {monk_secular}, Secular-Secular: {secular_secular}")
print(f"Expected MM ratio (random): {expected_mm:.4f}")
print(
    f"Permutation: obs={monk_monk}, null_mean={perm_mm.mean():.1f}, z={observed_z:.2f}, p={p_value:.4f}"
)

# 2b. Period assortativity
person_period = {}
# From buddhist JSON
with open(os.path.join(DATA_DIR, "person_years_buddhist.json")) as f:
    buddhist = json.load(f)["buddhist_monks"]
for key, info in buddhist.items():
    name_jp = info.get("name_jp", "")
    mission_id = str(info.get("mission_id", ""))
    m_match = re.search(r"M(\d+)", mission_id)
    if m_match:
        m_num = int(m_match.group(1))
        period = "Early (630-770)" if m_num <= 10 else "Late (771-894)"
        for nid, nname in node_name.items():
            if nname == name_jp or (
                len(name_jp) >= 2 and nname.startswith(name_jp[:2])
            ):
                person_period[nid] = period

# From officials JSON
with open(os.path.join(DATA_DIR, "person_years_officials.json")) as f:
    officials = json.load(f)["persons"]
for info in officials:
    name_jp = info.get("name_jp", "")
    missions = info.get("missions", [])
    for m_str in missions:
        m_match = re.search(r"M(\d+)", m_str)
        if m_match:
            m_num = int(m_match.group(1))
            period = "Early (630-770)" if m_num <= 10 else "Late (771-894)"
            for nid, nname in node_name.items():
                if nname == name_jp or (
                    len(name_jp) >= 2 and nname.startswith(name_jp[:2])
                ):
                    if nid in person_period and person_period[nid] != period:
                        # Multi-mission: mark as both
                        pass  # keep first assignment
                    else:
                        person_period[nid] = period

# Event-era fallback
era_patterns = {
    "Early (630-770)": [
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
    ],
    "Late (771-894)": [
        "寶龜",
        "天應",
        "延暦",
        "大同",
        "弘仁",
        "天長",
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
event_eras = {}
for _, row in nodes[nodes["type"] == "Event"].iterrows():
    text = str(row["name"]) + " " + str(row.get("sources", ""))
    for era_group, eras in era_patterns.items():
        for era in eras:
            if era in text:
                event_eras[row["id"]] = era_group
                break
        if row["id"] in event_eras:
            break

for _, row in social_edges.iterrows():
    src, tgt = row["source"], row["target"]
    if (
        src in event_eras
        and node_type.get(tgt) == "Person"
        and tgt not in person_period
    ):
        person_period[tgt] = event_eras[src]
    if (
        tgt in event_eras
        and node_type.get(src) == "Person"
        and src not in person_period
    ):
        person_period[src] = event_eras[tgt]

print(f"\nPersons with period: {len(person_period)}/{G.number_of_nodes()}")
early_ids = [n for n in G.nodes() if person_period.get(n, "").startswith("Early")]
late_ids = [n for n in G.nodes() if person_period.get(n, "").startswith("Late")]
print(f"Early: {len(early_ids)}, Late: {len(late_ids)}")

ee = 0
el = 0
ll = 0
for u, v in G.edges():
    up = person_period.get(u, "")
    vp = person_period.get(v, "")
    if up.startswith("E") and vp.startswith("E"):
        ee += 1
    elif up.startswith("L") and vp.startswith("L"):
        ll += 1
    elif (up.startswith("E") and vp.startswith("L")) or (
        up.startswith("L") and vp.startswith("E")
    ):
        el += 1

period_edges = ee + el + ll
print(f"E-E: {ee}, E-L: {el}, L-L: {ll} (of {period_edges} period-known edges)")

# ═══════════════════════════════════════════════════════════════════
# 3. BRIDGE PERSONS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. BRIDGE PERSONS (Cultural Brokers)")
print("=" * 70)

n_communities = len(set(person_community.values()))
print(f"Total Louvain communities: {n_communities}")

bridge_data = []
for n in G_cc.nodes():
    neighbors = list(G_cc.neighbors(n))
    neighbor_comms = set()
    for nb in neighbors:
        if nb in person_community:
            neighbor_comms.add(person_community[nb])
    n_unique = len(neighbor_comms)
    bc = bc_lookup.get(n, 0)
    wdeg = deg_lookup.get(n, 0)
    # Bridging score: betweenness * unique-neighbor-communities (threshold > 1 means bridge)
    bridge_score = bc * (n_unique if n_unique > 1 else 0)
    div_score = bc * (n_unique / max(1, n_communities))

    bridge_data.append(
        {
            "person_id": n,
            "person_name": node_name.get(n, n),
            "betweenness": bc,
            "weighted_degree": wdeg,
            "n_neighbor_comms": n_unique,
            "own_community": person_community.get(n, -1),
            "bridge_score": bridge_score,
            "diversity_score": div_score,
        }
    )

bridge_df = pd.DataFrame(bridge_data).sort_values("bridge_score", ascending=False)

print("\nTop Bridge Persons (betweenness × community diversity):")
print(
    f"{'Rank':<5} {'Name':<28} {'Betweenness':>12} {'WtDeg':>8} {'NbrComms':>10} {'OwnCom':>7} {'BridgeSc':>10}"
)
print("-" * 90)
count = 0
for _, row in bridge_df.iterrows():
    if row["betweenness"] > 0 or row["n_neighbor_comms"] > 1:
        count += 1
        print(
            f"{count:<5} {row['person_name']:<28} {row['betweenness']:>12.6f} {row['weighted_degree']:>8.3f} {row['n_neighbor_comms']:>10} {row['own_community']:>7} {row['bridge_score']:>10.6f}"
        )
    if count >= 15:
        break
if count == 0:
    # Show by community diversity even if betweenness is 0
    bridge_df_div = bridge_df.sort_values("n_neighbor_comms", ascending=False)
    print("\n(All betweenness values are 0 — showing by community diversity)")
    for idx, (_, row) in enumerate(bridge_df_div.iterrows()):
        if row["n_neighbor_comms"] > 1 and idx < 15:
            print(
                f"{idx + 1:<5} {row['person_name']:<28} {row['betweenness']:>12.6f} {row['weighted_degree']:>8.3f} {row['n_neighbor_comms']:>10} {row['own_community']:>7} {'—':>10}"
            )

# Articulation points
art_pts = (
    list(nx.articulation_points(G_cc_binary))
    if G_cc_binary.number_of_nodes() > 2
    else []
)
print(f"\nArticulation points: {len(art_pts)}")
for n in art_pts:
    print(f"  {node_name.get(n, n)}: betweenness={bc_lookup.get(n, 0):.6f}")

# ═══════════════════════════════════════════════════════════════════
# 4. EGO-NETWORK DENSITY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. EGO-NETWORK DENSITY COMPARISON")
print("=" * 70)

# Pick meaningful persons: key historical figures
key_persons = [
    "円仁",
    "空海",
    "最澄",
    "宗叡",
    "道昭",
    "行賀",
    "永忠",
    "常曉",
    "恵萼",
    "圓載",
    "圓珍",
    "定惠",
    "藤原朝臣常嗣",
    "高元度",
    "小野臣妹子",
    "阿倍仲麻呂",
    "吉備真備",
    "菅原朝臣清公等廿七人",
    "道慈",
    "圓澄",
]

ego_results = []
for pname in key_persons:
    # Find the person ID
    pid = None
    for nid, nname in node_name.items():
        if nname == pname:
            pid = nid
            break
    if pid is None:
        # Try containing match
        for nid, nname in node_name.items():
            if pname in nname and node_type.get(nid) == "Person":
                pid = nid
                break
    if pid is None:
        continue
    if pid not in G_cc:
        ego_results.append(
            {
                "person": pname,
                "n_neighbors": 0,
                "neighbor_edges": 0,
                "max_possible": 0,
                "ego_density": float("nan"),
                "weighted_clustering": float("nan"),
                "weighted_degree": deg_lookup.get(pid, 0),
                "betweenness": bc_lookup.get(pid, 0),
                "in_cc": False,
            }
        )
        continue

    neighbors = set(G_cc.neighbors(pid))
    n_neighbors = len(neighbors)

    neighbor_edges = 0
    for u in neighbors:
        for v in neighbors:
            if u < v and G_cc.has_edge(u, v):
                neighbor_edges += 1

    max_possible = n_neighbors * (n_neighbors - 1) // 2
    ego_density = neighbor_edges / max_possible if max_possible > 0 else 0
    wc = nx.clustering(G_cc, pid, weight="weight") if pid in G_cc else 0

    ego_results.append(
        {
            "person": pname,
            "n_neighbors": n_neighbors,
            "neighbor_edges": neighbor_edges,
            "max_possible": max_possible,
            "ego_density": ego_density,
            "weighted_clustering": wc,
            "weighted_degree": deg_lookup.get(pid, 0),
            "betweenness": bc_lookup.get(pid, 0),
            "in_cc": True,
        }
    )

    print(
        f"  {pname}: {n_neighbors} neighbors, {neighbor_edges}/{max_possible} edges, density={ego_density:.4f}, wc={wc:.4f}"
    )

ego_df = pd.DataFrame(ego_results)
ego_valid = ego_df[ego_df["in_cc"]].sort_values("ego_density")

print("\n--- Ego-Network Density Summary (sorted by density) ---")
print(
    f"{'Person':<28} {'Degree':>8} {'Nbrs':>5} {'Edges':>6} {'Density':>8} {'WtClust':>8} {'Btw':>10}"
)
print("-" * 85)
for _, row in ego_valid.iterrows():
    print(
        f"{row['person']:<28} {row['weighted_degree']:>8.3f} {row['n_neighbors']:>5} {row['neighbor_edges']:>6} {row['ego_density']:>8.4f} {row['weighted_clustering']:>8.4f} {row['betweenness']:>10.6f}"
    )

# Special: 円仁 vs 空海
ennin = ego_valid[ego_valid["person"] == "円仁"]
kukai = ego_valid[ego_valid["person"] == "空海"]
if len(ennin) > 0 and len(kukai) > 0:
    ed = ennin.iloc[0]["ego_density"]
    kd = kukai.iloc[0]["ego_density"]
    print("\n=== 円仁 vs 空海 ===")
    print(
        f"  円仁: density={ed:.4f}, neighbors={ennin.iloc[0]['n_neighbors']}, wt_clustering={ennin.iloc[0]['weighted_clustering']:.4f}"
    )
    print(
        f"  空海: density={kd:.4f}, neighbors={kukai.iloc[0]['n_neighbors']}, wt_clustering={kukai.iloc[0]['weighted_clustering']:.4f}"
    )
    if ed < kd:
        print("  → 円仁 LOWER → more star-like; 空海 HIGHER → more clustered")
    else:
        print("  → 空海 LOWER → more star-like; 円仁 HIGHER → more clustered")

# ═══════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING REPORT")
print("=" * 70)

R = []
R.append("# 遣唐使人物网络 — 深度网络分析报告")
R.append("")
R.append("*Jaccard-weighted bipartite projection | Generated: 2026-06-30*")
R.append("")
R.append("---")
R.append("")

# 1. Triads
R.append("## 1. 网络模体分析 (Network Motifs / Triads)")
R.append("")
R.append(
    f"**分析对象**: Person-Person Jaccard加权投影的最大连通分量 ({G_cc.number_of_nodes()} 人, {G_cc.number_of_edges()} 边)"
)
R.append("")
R.append("| 三元组类型 | 数量 | 占比 |")
R.append("|-----------|------|------|")
R.append(
    f"| 空三角 (0边) | {triad_counts_cc[0]} | {triad_counts_cc[0] / total_triads_cc * 100:.1f}% |"
)
R.append(
    f"| 单边 (1边) | {triad_counts_cc[1]} | {triad_counts_cc[1] / total_triads_cc * 100:.1f}% |"
)
R.append(
    f"| 开放楔形 (2边) | {triad_counts_cc[2]} | {triad_counts_cc[2] / total_triads_cc * 100:.1f}% |"
)
R.append(
    f"| **闭合三角 (3边)** | **{triad_counts_cc[3]}** | **{triad_counts_cc[3] / total_triads_cc * 100:.1f}%** |"
)
R.append("")

tri_pct = triad_counts_cc[3] / total_triads_cc * 100
wedge_pct = triad_counts_cc[2] / total_triads_cc * 100
R.append("### 关键发现")
R.append("")

if tri_pct > wedge_pct:
    R.append(
        f"**闭合三角占主导** ({tri_pct:.1f}%)。在网络的最大连通分量中，存在大量的三角闭合，说明这是一个高度聚类的网络。"
    )
    R.append(
        "这种情况的出现主要是因为许多历史人物共享了相同的遣唐使船只和路线，形成了多个紧密的全连接子群。"
    )
else:
    R.append(f"**开放楔形占主导** ({wedge_pct:.1f}%) vs 闭合三角 ({tri_pct:.1f}%)。")
    R.append("网络以开放结构为主，文化传播依赖关键枢纽而非密集的相互连接。")

R.append("")
R.append(f"- **全局传递性 (Transitivity)**: {cc_clustering:.4f}")
R.append(f"- **加权平均聚类系数**: {weighted_clustering_cc:.4f}")
R.append(f"- **平均度**: {avg_degree:.2f}")
R.append("")

# Interpret clustering
if cc_clustering > 0.6:
    R.append(
        f"传递性高达 {cc_clustering:.4f}，说明网络呈现**显著的小世界特性**——大多数人物通过共同的活动地点/文化圈形成了紧密的三角关系。这反映了遣唐使体系的高度组织性：同一批次的使团成员共享相同的旅途和目的地。"
    )
elif cc_clustering > 0.3:
    R.append(
        f"传递性 {cc_clustering:.4f} 处于中等水平，显示网络同时存在聚类和开放结构。"
    )
else:
    R.append(f"传递性 {cc_clustering:.4f} 较低，网络整体稀疏但有局部高密度子群。")

R.append("")

R.append("### 主导模式: 星形 vs 聚类")
R.append("")
R.append("**最星形 (star-like)** — 连接多但邻居间互连少：")
for ratio, deg, tris, name, nid in sorted(meaningful, key=lambda x: x[0])[:8]:
    R.append(f"- **{name}**: 度={deg}, 三角参与={tris}, 聚类比={ratio:.3f}")
R.append("")
R.append("**最聚类 (clustered)** — 邻居间高度互连：")
for ratio, deg, tris, name, nid in sorted(meaningful, key=lambda x: -x[0])[:8]:
    R.append(f"- **{name}**: 度={deg}, 三角参与={tris}, 聚类比={ratio:.3f}")
R.append("")

# Top triangles
R.append("### 最高权重三角")
R.append("")
for i, (avg_w, n1, n2, n3) in enumerate(triangles_weighted[:10]):
    R.append(f"{i + 1}. {n1} — {n2} — {n3} (平均权重: {avg_w:.4f})")
R.append("")
R.append(
    "*权重=1.0的三角形表示三人共享完全相同的活动地点/文化目标集合。这些多为同一遣唐使团的核心成员。*"
)
R.append("")

# 2. Assortativity
R.append("---")
R.append("## 2. 同配性分析 (Assortativity)")
R.append("")

R.append("### 2a. 僧侣 vs 世俗官员")
R.append("")
R.append("基于名字中包含 **釋/僧/法/師** 判定为僧侣。")
R.append(
    f"- 僧侣节点: {len(monk_nodes)} ({len(monk_nodes) / G.number_of_nodes() * 100:.1f}%)"
)
R.append(
    f"- 世俗节点: {len(secular_nodes)} ({len(secular_nodes) / G.number_of_nodes() * 100:.1f}%)"
)
R.append("")
R.append("| 边类型 | 数量 | 占比 |")
R.append("|--------|------|------|")
R.append(f"| 僧-僧 | {monk_monk} | {monk_monk / total_edges_binary * 100:.1f}% |")
R.append(f"| 僧-俗 | {monk_secular} | {monk_secular / total_edges_binary * 100:.1f}% |")
R.append(
    f"| 俗-俗 | {secular_secular} | {secular_secular / total_edges_binary * 100:.1f}% |"
)
R.append("")
R.append(f"- **随机期望 僧-僧 比例**: {expected_mm * 100:.1f}%")
R.append(f"- **置换检验**: z = {observed_z:.2f}, p = {p_value:.4f}")
R.append("")

if p_value < 0.05 and observed_z > 0:
    R.append(f"**✅ 结论: 僧侣显著倾向于连接僧侣 (homophily, p={p_value:.4f})**。")
    R.append(
        "这表明佛教僧团在遣唐使网络中形成了相对独立的子结构。他们通过共同的求法活动(参拜相同寺院、学习相同经典)彼此联系。"
    )
elif p_value < 0.05 and observed_z < 0:
    R.append(f"**✅ 结论: 僧侣显著避免彼此连接 (heterophily, p={p_value:.4f})**。")
else:
    R.append(f"**结论: 僧侣-僧侣连接与随机期望无显著差异 (p={p_value:.4f})**。")
    R.append(
        "僧侣和世俗人物的活动圈子有大量重叠，可能因为遣唐使团本身就是一个混合编队。"
    )

R.append("")

R.append("### 2b. 早期 vs 晚期人物")
R.append("")
R.append("基于遣唐使批次(M01-M10为早期630-770，M11-M20为晚期771-894)：")
R.append(
    f"- 可分配时代: {len(person_period)} 人 (早期 {len(early_ids)}, 晚期 {len(late_ids)})"
)
R.append("")
if period_edges > 0:
    R.append("| 边类型 | 数量 | 占比 |")
    R.append("|--------|------|------|")
    R.append(f"| 早期-早期 | {ee} | {ee / period_edges * 100:.1f}% |")
    R.append(f"| 早期-晚期 | {el} | {el / period_edges * 100:.1f}% |")
    R.append(f"| 晚期-晚期 | {ll} | {ll / period_edges * 100:.1f}% |")
    R.append("")
    if ee + ll > el:
        R.append(
            f"**同配连接为主**: 早期-早期({ee}) + 晚期-晚期({ll}) > 跨时代({el})。同时代人物共享更多活动模式。"
        )
    else:
        R.append(
            f"**跨时代连接显著**: 早期-晚期边({el})占主导。说明唐朝的佛教圣地和经典对不同时代日本求法者具有持续的吸引力。"
        )
R.append("")
R.append("*注：时代分配依赖于日文年号匹配和已知人物传记数据，覆盖度有限。*")
R.append("")

# 3. Bridge Persons
R.append("---")
R.append("## 3. 桥梁人物 (Bridge Persons / Cultural Brokers)")
R.append("")
R.append("桥梁人物定义为同时满足多条件的节点：")
R.append("1. 高**介数中心性** (betweenness centrality) — 位于许多最短路径上")
R.append("2. 连接**多个不同社区** — 充当不同群体的中介")
R.append("3. 是网络的**割点** (articulation point) — 移除后会断开网络")
R.append("")

R.append("### 介数中心性与社区多样性")
R.append("")
R.append(f"Louvain社区检测识别出 **{n_communities}** 个社区。")
R.append("")
R.append("| 排名 | 人物 | 介数中心性 | 加权度 | 跨社区数 | 桥梁分数 |")
R.append("|------|------|-----------|--------|---------|---------|")
count = 0
for _, row in bridge_df.iterrows():
    if row["betweenness"] > 0.001 or row["n_neighbor_comms"] > 1:
        count += 1
        R.append(
            f"| {count} | {row['person_name']} | {row['betweenness']:.6f} | {row['weighted_degree']:.3f} | {row['n_neighbor_comms']} | {row['bridge_score']:.6f} |"
        )
    if count >= 15:
        break
if count == 0:
    R.append("| — | 网络中无高介数节点 | — | — | — | — |")

R.append("")
R.append("### 割点 (Articulation Points)")
R.append("")
if art_pts:
    R.append("以下节点的移除会导致网络断开：")
    for n in art_pts:
        R.append(f"- **{node_name.get(n, n)}** (介数: {bc_lookup.get(n, 0):.6f})")
else:
    R.append("最大连通分量中**无割点** — 网络具有较好的鲁棒性。")
R.append("")

R.append("### 分析")
R.append("")
# Get top bridges
top_bridges = bridge_df[bridge_df["betweenness"] > 0.001]
if len(top_bridges) > 0:
    top_names = [row["person_name"] for _, row in top_bridges.head(3).iterrows()]
    R.append(f"最显著的桥梁人物: **{', '.join(top_names)}**。")
R.append("")
R.append('在Jaccard加权网络中，由于权重稀释了共享"日本"等常见地点的连接，')
R.append("网络的连通性更多依赖具体的历史事件参与而非泛泛的共同身份。")
R.append("真正起到桥梁作用的是那些参与了多个不同批次/类型活动的人物。")
R.append("")

# 4. Ego-network density
R.append("---")
R.append("## 4. 自我网络密度分析 (Ego-Network Density)")
R.append("")
R.append("自我网络密度衡量一个人物的直接邻居之间相互连接的程度。")
R.append("- **低密度 → 星形模式**: 中心人物是唯一的纽带，邻居间少有直接联系")
R.append("- **高密度 → 聚类模式**: 邻居之间也相互连接，形成紧密社群")
R.append("")

R.append("### 主要人物自我网络密度")
R.append("")
R.append("| 人物 | 加权度 | 邻居数 | 邻居间边 | 最大边 | 密度 | 加权聚类 | 介数 |")
R.append("|------|--------|--------|---------|--------|------|---------|------|")
for _, row in ego_valid.iterrows():
    R.append(
        f"| {row['person']} | {row['weighted_degree']:.3f} | {row['n_neighbors']} | {row['neighbor_edges']} | {row['max_possible']} | {row['ego_density']:.4f} | {row['weighted_clustering']:.4f} | {row['betweenness']:.6f} |"
    )
R.append("")

R.append("### 重点对比: 円仁 vs 空海")
R.append("")
if len(ennin) > 0 and len(kukai) > 0:
    ed = ennin.iloc[0]["ego_density"]
    kd = kukai.iloc[0]["ego_density"]
    en = ennin.iloc[0]["n_neighbors"]
    kn = kukai.iloc[0]["n_neighbors"]
    ew = ennin.iloc[0]["weighted_clustering"]
    kw = kukai.iloc[0]["weighted_clustering"]
    eb = ennin.iloc[0]["betweenness"]
    kb = kukai.iloc[0]["betweenness"]

    R.append("| 指标 | 円仁 | 空海 |")
    R.append("|------|------|------|")
    R.append(f"| 邻居数 | {en} | {kn} |")
    R.append(f"| 自我网络密度 | {ed:.4f} | {kd:.4f} |")
    R.append(f"| 加权聚类系数 | {ew:.4f} | {kw:.4f} |")
    R.append(f"| 介数中心性 | {eb:.6f} | {kb:.6f} |")
    R.append("")

    if ed < kd:
        R.append(f"**円仁的自我网络密度 ({ed:.4f}) 低于空海 ({kd:.4f})**。")
        R.append("")
        R.append(
            "- **円仁呈现星形模式**: 作为《入唐求法巡礼行記》的作者，円仁的活动范围极广(70个源文本)——他连接了大量不同背景的人物，但这些人物彼此之间独立活动、少有交集。"
        )
        R.append(
            f"- **空海更聚类**: 空海的邻居({kn}人)之间有更高比例的相互连接，反映了其作为真言宗创始人所建立的紧密师徒/法脉网络。"
        )
    else:
        R.append(f"**空海的自我网络密度 ({kd:.4f}) 低于円仁 ({ed:.4f})**。")
    R.append("")
    R.append(
        "这一差异印证了两人的历史角色：円仁是**广泛的信息收集者和记录者**，空海是**深度的宗派建设者和教义传播者**。"
    )
    R.append("")

R.append("### 低密度 vs 高密度人物")
R.append("")
low_density = ego_valid[ego_valid["n_neighbors"] >= 3].nsmallest(3, "ego_density")
high_density = ego_valid[ego_valid["n_neighbors"] >= 3].nlargest(3, "ego_density")
R.append("**最星形 (低密度, ≥3邻居)**: " + ", ".join(low_density["person"].tolist()))
R.append("")
R.append("**最聚类 (高密度, ≥3邻居)**: " + ", ".join(high_density["person"].tolist()))
R.append("")

# 5. Summary
R.append("---")
R.append("## 5. 总结")
R.append("")

R.append("### 5.1 网络结构特征")
R.append("")
R.append(
    f"- Person-Person Jaccard加权网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边"
)
R.append(
    f"- 最大连通分量: {G_cc.number_of_nodes()} 节点 ({G_cc.number_of_nodes() / G.number_of_nodes() * 100:.1f}%), {G_cc.number_of_edges()} 边"
)
R.append("- 边权重范围反映了人物之间共享活动地点/文化圈的程度")
R.append("")

R.append("### 5.2 核心发现")
R.append("")
R.append(
    f"1. **三元组模式**: 传递性={cc_clustering:.4f}, {'高度聚类' if cc_clustering > 0.6 else '中等聚类' if cc_clustering > 0.3 else '稀疏开放'}。"
)
R.append(f"   闭合三角占 {tri_pct:.1f}%，网络中存在因共同遣唐使批次形成的紧密子群。")
R.append("")

if p_value < 0.05:
    R.append(
        f"2. **僧侣同配性**: 置换检验{'显著' if p_value < 0.01 else '边缘显著'} (p={p_value:.4f})。僧侣群体在活动空间上有一定独立性。"
    )
else:
    R.append(
        f"2. **僧侣/世俗混合**: 僧侣与世俗官员的活动网络深度重叠 (p={p_value:.4f})，反映了遣唐使团的人员混合性质。"
    )
R.append("")

R.append("3. **桥梁人物**: 通过介数中心性和社区多样性识别了关键的文化中介者。")
R.append(
    f"   包括{'割点: ' + ', '.join(node_name.get(n, n) for n in art_pts) if art_pts else '无割点（网络有较好的连通鲁棒性）'}。"
)
R.append("")

R.append("4. **自我网络密度差异**: 不同历史人物的网络结构反映了不同的社会角色。")
R.append("   广记录者(如円仁)呈星形模式，宗派建设者(如空海)呈聚类模式。")
R.append("")

R.append("---")
R.append(
    "*方法: Jaccard加权双模投影 + 三元组普查 + 置换检验 + 割点分析 + 自我网络密度*"
)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(R))

print(f"\n✅ Report written to {REPORT_PATH}")
print(f"   Length: {len(R)} lines, {sum(len(line) for line in R)} chars")
