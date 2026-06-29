#!/usr/bin/env python3
"""
Place Importance Ranking Analysis for Kentoshi Network
1. Extract all Place nodes and compute centrality measures
2. Compare Tang cities vs Japanese ports vs temples/shrines
3. Specific comparisons: 長安 vs 洛陽, 揚州 vs 明州
4. Temporal shifts: early vs late periods
5. Cross-validation with Place-Place projection
"""

import csv
import json
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations
import math

DATA = "/home/szhh/kento-shi-network/output"
ANALYSIS = "/home/szhh/kento-shi-network/analysis"

# ============================================================
# Load data
# ============================================================
nodes = {}
with open(f"{DATA}/nodes.csv") as f:
    for row in csv.DictReader(f):
        rid = row["id"].strip()
        nodes[rid] = row

edges = []
with open(f"{DATA}/edges.csv") as f:
    for row in csv.DictReader(f):
        edges.append(
            {
                "id": row["id"].strip(),
                "source": row["source"].strip(),
                "target": row["target"].strip(),
                "relation": row["relation"].strip(),
            }
        )

events = []
with open(f"{DATA}/enhanced_events.json") as f:
    events = json.load(f)

# Index events by node id
event_by_id = {}
for i, ev in enumerate(events):
    eid = f"E{i + 1:03d}"
    event_by_id[eid] = ev

# Node lookups
node_type = {}
node_name = {}
node_source_count = {}
for rid, row in nodes.items():
    node_type[rid] = row["type"].strip()
    node_name[rid] = row["name"].strip()
    node_source_count[rid] = int(row["source_count"].strip())

# Node type lists
person_nodes = [rid for rid, t in node_type.items() if t == "Person"]
place_nodes = [rid for rid, t in node_type.items() if t == "Place"]
culture_nodes = [rid for rid, t in node_type.items() if t == "Culture"]
group_nodes = [rid for rid, t in node_type.items() if t == "Group"]

print(
    f"Node counts: {len(person_nodes)} Person, {len(place_nodes)} Place, {len(culture_nodes)} Culture, {len(group_nodes)} Group"
)
print(f"Edge count: {len(edges)}")

# ============================================================
# Build full bipartite network: Person <-> Place (via TRAVEL, AT_LOCATION)
# ============================================================
G_full = nx.Graph()

# Add all Place nodes
for pid in place_nodes:
    G_full.add_node(pid, name=node_name[pid], type="Place")

# Add all Person nodes
for pid in person_nodes:
    G_full.add_node(pid, name=node_name[pid], type="Person")

# Add all Group nodes (some may also travel)
for gid in group_nodes:
    G_full.add_node(gid, name=node_name[gid], type="Group")

# Add edges: Person/Group -> Place via TRAVEL edges
travel_edges = []
for e in edges:
    if e["relation"] == "TRAVEL":
        src, tgt = e["source"], e["target"]
        src_type = node_type.get(src, "")
        tgt_type = node_type.get(tgt, "")
        if src_type in ("Person", "Group") and tgt_type == "Place":
            if G_full.has_edge(src, tgt):
                G_full[src][tgt]["weight"] += 1
            else:
                G_full.add_edge(src, tgt, weight=1, relation="TRAVEL")
            travel_edges.append((src, tgt))
        elif tgt_type in ("Person", "Group") and src_type == "Place":
            if G_full.has_edge(src, tgt):
                G_full[src][tgt]["weight"] += 1
            else:
                G_full.add_edge(src, tgt, weight=1, relation="TRAVEL")
            travel_edges.append((tgt, src))

print(f"TRAVEL edges: {len(travel_edges)}")

# Also add AT_LOCATION edges from events
at_location_count = 0
for e in edges:
    if e["relation"] == "AT_LOCATION":
        src, tgt = e["source"], e["target"]
        # src is Event, tgt is Place
        if node_type.get(tgt) == "Place":
            # Find the subject of this event
            for e2 in edges:
                if e2["source"] == src and e2["relation"] == "HAS_SUBJECT":
                    subject = e2["target"]
                    if node_type.get(subject) in ("Person", "Group"):
                        if G_full.has_edge(subject, tgt):
                            G_full[subject][tgt]["weight"] += 1
                        else:
                            G_full.add_edge(
                                subject, tgt, weight=1, relation="AT_LOCATION"
                            )
                        at_location_count += 1
                        break

print(f"AT_LOCATION edges added: {at_location_count}")

# ============================================================
# 1. CENTRALITY MEASURES FOR PLACES
# ============================================================

# 1a. Weighted degree (sum of edge weights connecting to persons/groups)
place_weighted_degree = {}
for pid in place_nodes:
    wd = (
        sum(d["weight"] for _, _, d in G_full.edges(pid, data=True))
        if pid in G_full
        else 0
    )
    place_weighted_degree[pid] = wd

# 1b. Degree (number of unique person/group connections)
place_degree = {}
for pid in place_nodes:
    place_degree[pid] = G_full.degree(pid) if pid in G_full else 0

# 1c. Betweenness centrality in the full graph
bc_full = nx.betweenness_centrality(G_full, weight="weight", normalized=True)

# 1d. PageRank in the full graph
pr_full = nx.pagerank(G_full, weight="weight")

# 1e. Place-Place projection (co-visitation by same entity)
# Collect which places each entity visits
entity_places = defaultdict(set)
for e in travel_edges:
    entity, place = e
    entity_places[entity].add(place)

G_pp = nx.Graph()
for pid in place_nodes:
    G_pp.add_node(pid, name=node_name.get(pid, pid))

for entity, places in entity_places.items():
    plist = list(places)
    for p1, p2 in combinations(plist, 2):
        if p1 in place_nodes and p2 in place_nodes:
            if G_pp.has_edge(p1, p2):
                G_pp[p1][p2]["weight"] += 1
            else:
                G_pp.add_edge(p1, p2, weight=1)

# Betweenness in Place-Place projection
bc_pp = nx.betweenness_centrality(G_pp, weight="weight", normalized=True)
# Weighted degree in PP
pp_wdeg = {
    n: sum(d["weight"] for _, _, d in G_pp.edges(n, data=True)) for n in G_pp.nodes()
}
# PageRank in PP
pr_pp = nx.pagerank(G_pp, weight="weight")

print(
    f"\nPlace-Place projection: {G_pp.number_of_nodes()} nodes, {G_pp.number_of_edges()} edges"
)

# ============================================================
# COMPILE RANKINGS
# ============================================================
print("\n" + "=" * 100)
print("PLACE IMPORTANCE RANKING")
print("=" * 100)


# Combined ranking: average of normalized scores
def rank_places(metric_dict, name):
    ranked = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)
    return ranked


# Create a composite score (average of all normalized ranks)
# For each metric, compute percentile rank
metrics = {
    "Weighted Degree (full)": place_weighted_degree,
    "Degree (full)": place_degree,
    "Betweenness (full)": bc_full,
    "PageRank (full)": pr_full,
    "Betweenness (PP)": bc_pp,
    "Weighted Degree (PP)": pp_wdeg,
    "PageRank (PP)": pr_pp,
}

# Compute percentile rank for each metric
n_places = len(place_nodes)
percentile_ranks = defaultdict(dict)
for mname, mdict in metrics.items():
    sorted_ids = sorted(mdict.keys(), key=lambda x: mdict.get(x, 0), reverse=True)
    for rank, pid in enumerate(sorted_ids):
        percentile = 1.0 - (rank / max(n_places - 1, 1))
        percentile_ranks[mname][pid] = percentile

# Composite score
composite = {}
for pid in place_nodes:
    scores = [percentile_ranks[m].get(pid, 0) for m in metrics]
    composite[pid] = sum(scores) / len(scores)

# Top-30 places
ranked_composite = sorted(composite.items(), key=lambda x: x[1], reverse=True)

print(
    f"\n{'Rank':<5} {'ID':<8} {'Name':<45} {'Composite':>10} {'WD-full':>10} {'BC-full':>10} {'PR-full':>10} {'BC-PP':>10} {'WD-PP':>10} {'PR-PP':>10} {'src':>5}"
)
print("-" * 140)

for i, (pid, comp) in enumerate(ranked_composite[:50], 1):
    name = node_name.get(pid, pid)
    wd = place_weighted_degree.get(pid, 0)
    bcf = bc_full.get(pid, 0)
    prf = pr_full.get(pid, 0)
    bcp = bc_pp.get(pid, 0)
    wdp = pp_wdeg.get(pid, 0)
    prp = pr_pp.get(pid, 0)
    sc = node_source_count.get(pid, 0)
    print(
        f"{i:<5} {pid:<8} {name:<45} {comp:>10.4f} {wd:>10.1f} {bcf:>10.4f} {prf:>10.4f} {bcp:>10.4f} {wdp:>10.1f} {prp:>10.4f} {sc:>5}"
    )

# ============================================================
# 2. CATEGORY COMPARISON
# ============================================================
print("\n" + "=" * 100)
print("2. CATEGORY COMPARISON: Tang Cities vs Japanese Ports vs Temples/Shrines")
print("=" * 100)


# Classify each place into category
def classify_place(name):
    """Classify place into broad categories"""
    name = name.strip()

    # Japanese places
    japanese_ports = [
        "難波",
        "筑紫",
        "博多",
        "肥前",
        "松浦",
        "橘浦",
        "太宰府",
        "大宰府",
        "平城京",
        "平安京",
        "大津",
        "攝津",
        "播磨",
        "備後",
        "安芸",
        "周防",
        "長門",
        "筑前",
        "筑後",
        "肥後",
        "薩摩",
        "大隅",
        "日向",
        "値嘉嶋",
        "甑島",
        "種子島",
        "屋久島",
        "奄美",
        "沖縄",
        "日本",
        "倭",
        "大和",
        "山城",
        "河内",
        "和泉",
        "攝津国",
        "筑紫国",
        "肥前国",
        "松浦郡",
    ]

    japanese_temples = [
        "東大寺",
        "興福寺",
        "薬師寺",
        "元興寺",
        "大安寺",
        "西大寺",
        "法隆寺",
        "唐招提寺",
        "延暦寺",
        "金剛峯寺",
        "比叡山",
        "高野山",
        "東寺",
        "西寺",
        "鞍馬寺",
        "清水寺",
        "四天王寺",
        "飛鳥寺",
        "川原寺",
        "橘寺",
        "當麻寺",
    ]

    japanese_other = [
        "対馬",
        "壱岐",
        "隠岐",
        "佐渡",
        "淡路",
        "伊勢",
        "志摩",
        "熊野",
        "吉野",
        "出雲",
        "春日",
        "住吉",
    ]

    # Tang cities
    tang_capitals = ["長安", "洛陽", "東都"]
    tang_major_cities = [
        "揚州",
        "明州",
        "越州",
        "蘇州",
        "杭州",
        "楚州",
        "登州",
        "萊州",
        "福州",
        "泉州",
        "広州",
        "台州",
        "温州",
        "潤州",
        "常州",
        "湖州",
        "宣州",
        "洪州",
        "鄂州",
        "潭州",
        "衡州",
        "襄州",
        "江陵",
        "成都",
        "太原",
        "幽州",
        "涼州",
        "沙州",
    ]
    tang_other = ["長安城", "洛陽城", "西京", "南京", "上都"]

    # Temples and shrines (generic, could be Tang or Japanese)
    temples = [
        "寺",
        "院",
        "廟",
        "祠",
        "觀",
        "宮",
        "臺",
        "塔",
        "堂",
        "閣",
        "庵",
        "青龍寺",
        "大興善寺",
        "興唐寺",
        "開元寺",
        "大薦福寺",
        "西明寺",
        "慈恩寺",
        "章敬寺",
        "資聖寺",
        "化度寺",
        "醴泉寺",
        "菩提寺",
        "實際寺",
        "禪定寺",
        "弘福寺",
        "大莊嚴寺",
        "總持寺",
        "空慧寺",
        "白馬寺",
        "香山寺",
        "奉先寺",
        "天宮寺",
        "敬愛寺",
        "福先寺",
        "荷澤寺",
        "龍興寺",
        "廣化寺",
        "聖善寺",
        "安国寺",
        "南臺",
        "北臺",
        "東臺",
        "西臺",
        "中臺",
        "五臺山",
        "天台山",
        "峨眉山",
        "終南山",
        "廬山",
        "衡山",
        "會昌寺",
        "玄法寺",
        "保壽寺",
        "罔極寺",
        "少林寺",
        "相国寺",
        "永平寺",
        "竹林寺",
        "靈巖寺",
        "神通寺",
        "朗公寺",
        "定水寺",
        "龍門",
        "奉天",
        "乾陵",
        "昭陵",
    ]

    # Check Tang cities first
    for city in tang_capitals:
        if city in name:
            return "Tang Capital"
    for city in tang_major_cities:
        if city in name:
            return "Tang City"
    for city in tang_other:
        if city in name:
            return "Tang City"

    # Check Japanese ports
    for port in japanese_ports:
        if port in name:
            return "Japanese Port/Region"

    # Check Japanese temples
    for t in japanese_temples:
        if t in name:
            return "Japanese Temple/Shrine"

    # Check Japanese other
    for o in japanese_other:
        if o in name:
            return "Japanese Other"

    # Generic temple check
    for t in temples:
        if t in name:
            return "Temple/Shrine"

    # Default: check if contains Japanese region markers
    jp_indicators = ["浦", "津", "嶋", "島", "国", "郡"]
    for ind in jp_indicators:
        if ind in name:
            return "Japanese Port/Region"

    # Check for Chinese city indicators
    cn_indicators = ["州", "京", "都", "府", "縣", "県"]
    for ind in cn_indicators:
        if ind in name:
            return "Tang City"

    return "Other/Unknown"


# Classify all places
place_category = {}
for pid in place_nodes:
    name = node_name.get(pid, "")
    place_category[pid] = classify_place(name)

# Count and compare
cat_counts = Counter(place_category.values())
print("\nCategory distribution:")
for cat, count in cat_counts.most_common():
    print(f"  {cat}: {count}")

# Average composite score per category
cat_scores = defaultdict(list)
for pid, cat in place_category.items():
    cat_scores[cat].append(composite[pid])

print("\nAverage composite score by category:")
for cat in sorted(
    cat_scores.keys(),
    key=lambda c: sum(cat_scores[c]) / len(cat_scores[c]),
    reverse=True,
):
    avg = sum(cat_scores[cat]) / len(cat_scores[cat])
    print(f"  {cat}: {avg:.4f} (n={len(cat_scores[cat])})")

# Top places per category
print("\nTop places per category:")
for cat in sorted(cat_scores.keys()):
    cat_places = [
        (pid, composite[pid]) for pid in place_nodes if place_category[pid] == cat
    ]
    cat_places.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  {cat} (Top 10):")
    for pid, score in cat_places[:10]:
        name = node_name.get(pid, pid)
        print(f"    {name}: {score:.4f}")

# ============================================================
# 3. SPECIFIC COMPARISONS
# ============================================================
print("\n" + "=" * 100)
print("3. SPECIFIC COMPARISONS")
print("=" * 100)

# 3a. 長安 vs 洛陽
print("\n3a. 長安 (Chang'an) vs 洛陽 (Luoyang):")
changan_ids = [pid for pid in place_nodes if "長安" in node_name.get(pid, "")]
luoyang_ids = [pid for pid in place_nodes if "洛陽" in node_name.get(pid, "")]

for label, ids in [("長安", changan_ids), ("洛陽", luoyang_ids)]:
    if ids:
        pid = ids[0]
        print(f"\n  {label} ({pid}):")
        print(f"    Composite Score: {composite[pid]:.4f}")
        print(f"    Weighted Degree (full): {place_weighted_degree.get(pid, 0):.1f}")
        print(f"    Betweenness (full): {bc_full.get(pid, 0):.4f}")
        print(f"    PageRank (full): {pr_full.get(pid, 0):.4f}")
        print(f"    Betweenness (PP): {bc_pp.get(pid, 0):.4f}")
        print(f"    Weighted Degree (PP): {pp_wdeg.get(pid, 0):.1f}")
        print(f"    PageRank (PP): {pr_pp.get(pid, 0):.4f}")
        print(f"    Source Count: {node_source_count.get(pid, 0)}")
        # Who visits this place?
        visitors = []
        for u, v, d in G_full.edges(pid, data=True):
            other = v if u == pid else u
            if node_type.get(other) in ("Person", "Group"):
                visitors.append((node_name.get(other, other), d.get("weight", 1)))
        visitors.sort(key=lambda x: x[1], reverse=True)
        print(f"    Visitors (top 10): {visitors[:10]}")

# 3b. 揚州 vs 明州
print("\n3b. 揚州 (Yangzhou) vs 明州 (Mingzhou/Ningbo):")
yangzhou_ids = [pid for pid in place_nodes if "揚州" in node_name.get(pid, "")]
mingzhou_ids = [pid for pid in place_nodes if "明州" in node_name.get(pid, "")]

for label, ids in [("揚州", yangzhou_ids), ("明州", mingzhou_ids)]:
    if ids:
        pid = ids[0]
        print(f"\n  {label} ({pid}):")
        print(f"    Composite Score: {composite[pid]:.4f}")
        print(f"    Weighted Degree (full): {place_weighted_degree.get(pid, 0):.1f}")
        print(f"    Betweenness (full): {bc_full.get(pid, 0):.4f}")
        print(f"    PageRank (full): {pr_full.get(pid, 0):.4f}")
        print(f"    Betweenness (PP): {bc_pp.get(pid, 0):.4f}")
        print(f"    Weighted Degree (PP): {pp_wdeg.get(pid, 0):.1f}")
        print(f"    Source Count: {node_source_count.get(pid, 0)}")
        visitors = []
        for u, v, d in G_full.edges(pid, data=True):
            other = v if u == pid else u
            if node_type.get(other) in ("Person", "Group"):
                visitors.append((node_name.get(other, other), d.get("weight", 1)))
        visitors.sort(key=lambda x: x[1], reverse=True)
        print(f"    Visitors (top 10): {visitors[:10]}")

# 3c. Key port comparison
print("\n3c. Key Chinese Ports comparison:")
port_names = ["揚州", "明州", "楚州", "登州", "蘇州", "越州", "福州", "広州"]
print(
    f"{'Port':<10} {'Composite':>10} {'WD-full':>10} {'BC-full':>10} {'PR-full':>10} {'BC-PP':>10} {'Contribs':>5}"
)
print("-" * 70)
for port_name in port_names:
    pid = None
    for p in place_nodes:
        if port_name in node_name.get(p, ""):
            pid = p
            break
    if pid:
        print(
            f"{port_name:<10} {composite[pid]:>10.4f} {place_weighted_degree.get(pid, 0):>10.1f} {bc_full.get(pid, 0):>10.4f} {pr_full.get(pid, 0):>10.4f} {bc_pp.get(pid, 0):>10.4f} {node_source_count.get(pid, 0):>5}"
        )

# ============================================================
# 4. TEMPORAL ANALYSIS
# ============================================================
print("\n" + "=" * 100)
print("4. TEMPORAL ANALYSIS: Place Importance by Period")
print("=" * 100)

# Extract mission years
mission_years = {}
for ev in events:
    mid = ev.get("mission_id", "")
    if mid and mid.startswith("M"):
        try:
            year_str = str(ev.get("mission_departure_year", ""))
            if year_str and year_str != "不明" and year_str != "":
                # Handle range like "805～806" or "733-734"
                y = int(year_str.split("～")[0].split("-")[0].strip())
                if mid not in mission_years or y < mission_years[mid]:
                    mission_years[mid] = y
        except (ValueError, KeyError):
            pass

print(f"Mission years extracted: {len(mission_years)}")
for mid, year in sorted(mission_years.items(), key=lambda x: x[1]):
    print(f"  {mid}: {year}")

# Build event -> mission_id mapping
event_mission = {}
for i, ev in enumerate(events):
    eid = f"E{i + 1:03d}"
    mid = ev.get("mission_id", "")
    event_mission[eid] = mid

# For each TRAVEL edge, determine the time period
# TRAVEL edges are Person->Place. Find which event generates each travel via AT_LOCATION edges.

# Collect place visits with time information
place_visits = []  # (person, place, mission_id, year)
for e in edges:
    if e["relation"] == "TRAVEL":
        src, tgt = e["source"], e["target"]
        src_type = node_type.get(src, "")
        tgt_type = node_type.get(tgt, "")
        if src_type in ("Person", "Group") and tgt_type == "Place":
            person, place = src, tgt
        elif tgt_type in ("Person", "Group") and src_type == "Place":
            person, place = tgt, src
        else:
            continue

        # Try to find the event that contains this TRAVEL edge
        # TRAVEL edges don't directly link to events. We need to find AT_LOCATION edges.
        # Actually, let's use the approach from place_culture_analysis.py: iterate events directly
        pass

# Better approach: iterate events, find subject and location
entity_travels = defaultdict(list)
for i, ev in enumerate(events):
    eid = f"E{i + 1:03d}"
    mid = ev.get("mission_id", "")
    year = mission_years.get(mid, None)

    # Find subject and location from edges
    subject = None
    location = None
    for e in edges:
        if e["source"] == eid:
            if e["relation"] == "HAS_SUBJECT":
                subj_type = node_type.get(e["target"], "")
                if subj_type in ("Person", "Group"):
                    subject = e["target"]
            elif e["relation"] == "AT_LOCATION":
                loc_type = node_type.get(e["target"], "")
                if loc_type == "Place":
                    location = e["target"]
        elif e["target"] == eid:
            if e["relation"] == "HAS_SUBJECT":
                subj_type = node_type.get(e["source"], "")
                if subj_type in ("Person", "Group"):
                    subject = e["source"]

    if subject and location:
        entity_travels[subject].append(
            {
                "place": location,
                "place_name": node_name.get(location, location),
                "time": ev.get("time_period", ""),
                "mission": mid,
                "year": year,
            }
        )

print(
    f"\nEntity travels extracted: {len(entity_travels)} entities, {sum(len(v) for v in entity_travels.values())} visits"
)

# Also add TRAVEL edges directly for those not covered by events
for e in edges:
    if e["relation"] == "TRAVEL":
        src, tgt = e["source"], e["target"]
        src_type = node_type.get(src, "")
        tgt_type = node_type.get(tgt, "")
        if src_type in ("Person", "Group") and tgt_type == "Place":
            person, place = src, tgt
        elif tgt_type in ("Person", "Group") and src_type == "Place":
            person, place = tgt, src
        else:
            continue

        # Only add if not already captured
        existing = entity_travels.get(person, [])
        if not any(t["place"] == place for t in existing):
            entity_travels[person].append(
                {
                    "place": place,
                    "place_name": node_name.get(place, place),
                    "time": "",
                    "mission": "unknown",
                    "year": None,
                }
            )

# Split into periods
# Early: 630-750 (missions M01-M09)
# Late: 751-894 (missions M10-M20)
early_cutoff = 750

early_places = Counter()
late_places = Counter()
early_entity_places = defaultdict(set)
late_entity_places = defaultdict(set)

for entity, travels in entity_travels.items():
    for t in travels:
        year = t.get("year")
        place = t["place"]
        if year is not None:
            if year <= early_cutoff:
                early_places[place] += 1
                early_entity_places[entity].add(place)
            else:
                late_places[place] += 1
                late_entity_places[entity].add(place)

print(
    f"\nEarly period (<= {early_cutoff}): {sum(early_places.values())} visits to {len(early_places)} places"
)
print(
    f"Late period (> {early_cutoff}): {sum(late_places.values())} visits to {len(late_places)} places"
)

print("\nTop 20 places in EARLY period:")
for pid, count in early_places.most_common(20):
    name = node_name.get(pid, pid)
    print(f"  {name}: {count}")

print("\nTop 20 places in LATE period:")
for pid, count in late_places.most_common(20):
    name = node_name.get(pid, pid)
    print(f"  {name}: {count}")

# Places unique to each period
early_set = set(early_places.keys())
late_set = set(late_places.keys())
only_early = early_set - late_set
only_late = late_set - early_set
both = early_set & late_set

print(f"\nPlaces visited ONLY in early period ({len(only_early)}):")
for pid in sorted(only_early, key=lambda x: early_places[x], reverse=True)[:30]:
    name = node_name.get(pid, pid)
    print(f"  {name} ({early_places[pid]})")

print(f"\nPlaces visited ONLY in late period ({len(only_late)}):")
for pid in sorted(only_late, key=lambda x: late_places[x], reverse=True)[:30]:
    name = node_name.get(pid, pid)
    print(f"  {name} ({late_places[pid]})")

print(f"\nPlaces visited in BOTH periods ({len(both)}):")
for pid in sorted(both, key=lambda x: early_places[x] + late_places[x], reverse=True)[
    :20
]:
    name = node_name.get(pid, pid)
    print(f"  {name} (early={early_places[pid]}, late={late_places[pid]})")


# Build separate Place-Place projections per period
def build_pp_projection(entity_places_dict, place_nodes_set):
    G = nx.Graph()
    for pid in place_nodes_set:
        G.add_node(pid, name=node_name.get(pid, pid))
    for entity, places in entity_places_dict.items():
        plist = list(places)
        for p1, p2 in combinations(plist, 2):
            if p1 in place_nodes_set and p2 in place_nodes_set:
                if G.has_edge(p1, p2):
                    G[p1][p2]["weight"] += 1
                else:
                    G.add_edge(p1, p2, weight=1)
    return G


G_pp_early = build_pp_projection(early_entity_places, set(place_nodes))
G_pp_late = build_pp_projection(late_entity_places, set(place_nodes))

bc_pp_early = nx.betweenness_centrality(G_pp_early, weight="weight", normalized=True)
bc_pp_late = nx.betweenness_centrality(G_pp_late, weight="weight", normalized=True)

print(
    f"\nPlace-Place networks: Early={G_pp_early.number_of_nodes()}n/{G_pp_early.number_of_edges()}e, Late={G_pp_late.number_of_nodes()}n/{G_pp_late.number_of_edges()}e"
)

# Compare top betweenness places per period
print("\nTop 15 places by Betweenness (PP) - EARLY period:")
for pid, score in sorted(bc_pp_early.items(), key=lambda x: x[1], reverse=True)[:15]:
    name = node_name.get(pid, pid)
    print(f"  {name}: {score:.6f}")

print("\nTop 15 places by Betweenness (PP) - LATE period:")
for pid, score in sorted(bc_pp_late.items(), key=lambda x: x[1], reverse=True)[:15]:
    name = node_name.get(pid, pid)
    print(f"  {name}: {score:.6f}")

# ============================================================
# 5. CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 100)
print("5. CROSS-VALIDATION")
print("=" * 100)

# Compare rankings from different methods
print("\n5a. Correlation between different centrality measures:")


# Manual Spearman correlation (avoid scipy dependency)
def spearmanr_manual(x, y):
    """Compute Spearman rank correlation coefficient and approximate p-value."""
    n = len(x)
    if n <= 2:
        return 0.0, 1.0

    # Rank the values
    def rank_values(vals):
        sorted_pairs = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0] * n
        i = 0
        while i < n:
            j = i
            while j < n and sorted_pairs[j][1] == sorted_pairs[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                ranks[sorted_pairs[k][0]] = avg_rank
            i = j
        return ranks

    rx = rank_values(x)
    ry = rank_values(y)
    # Spearman r = Pearson r on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    den_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))
    den = (den_x * den_y) ** 0.5
    if den == 0:
        return 0.0, 1.0
    r = num / den
    # t-test approximation
    if abs(r) == 1.0:
        p = 0.0
    else:
        t_stat = r * ((n - 2) / (1 - r**2)) ** 0.5
        # Approximate p-value from t-distribution (rough)

        # Welch–Satterthwaite approximation
        p = 2 * (1 - _t_cdf_approx(abs(t_stat), n - 2))
    return r, p


def _t_cdf_approx(t, df):
    """Rough approximation of t-distribution CDF."""

    x = df / (df + t**2)
    # Regularized incomplete beta function approximation
    return 1 - 0.5 * (x ** (df / 2))


# Pearson correlation
def pearsonr_manual(x, y):
    n = len(x)
    if n <= 2:
        return 0.0, 1.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    den = (den_x * den_y) ** 0.5
    if den == 0:
        return 0.0, 1.0
    r = num / den
    # Fisher z-transform for p-value approximation
    if abs(r) >= 1.0:
        p = 0.0
    else:
        z = 0.5 * math.log((1 + r) / (1 - r))
        se = 1.0 / (n - 3) ** 0.5
        z_score = abs(z) / se
        # Normal approximation
        p = 2 * (1 - _norm_cdf_approx(z_score))
    return r, p


def _norm_cdf_approx(z):
    """Abramowitz and Stegun approximation of standard normal CDF."""
    import math

    if z < 0:
        return 1 - _norm_cdf_approx(-z)
    # Constants for approximation
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    t = 1.0 / (1.0 + b0 * z)
    phi = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-(z**2) / 2)
    P = 1.0 - phi * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)
    return P


methods = {
    "WD-full": place_weighted_degree,
    "BC-full": bc_full,
    "PR-full": pr_full,
    "BC-PP": bc_pp,
    "WD-PP": pp_wdeg,
    "PR-PP": pr_pp,
    "Composite": composite,
}

method_names = list(methods.keys())
print(f"{'Method1':<15} {'Method2':<15} {'Spearman r':>12} {'Pearson r':>12}")
print("-" * 60)
for i, m1 in enumerate(method_names):
    for m2 in method_names[i + 1 :]:
        ids_common = [
            pid for pid in place_nodes if pid in methods[m1] and pid in methods[m2]
        ]
        v1 = [methods[m1][pid] for pid in ids_common]
        v2 = [methods[m2][pid] for pid in ids_common]
        if len(v1) > 2:
            sr, sp = spearmanr_manual(v1, v2)
            pr, pp = pearsonr_manual(v1, v2)
            print(f"{m1:<15} {m2:<15} {sr:>12.4f} {pr:>12.4f}")

# Check source_count bias
print("\n5b. Source count bias analysis:")
for mname, mdict in [
    ("WD-full", place_weighted_degree),
    ("BC-full", bc_full),
    ("PR-full", pr_full),
    ("BC-PP", bc_pp),
    ("Composite", composite),
]:
    ids = [pid for pid in place_nodes if pid in mdict]
    v1 = [mdict[pid] for pid in ids]
    v2 = [node_source_count.get(pid, 0) for pid in ids]
    sr, sp = spearmanr_manual(v1, v2)
    bias = "✅ BIASED" if sp < 0.05 and abs(sr) > 0.3 else "OK"
    print(f"  {mname} vs source_count: Spearman r={sr:.4f}, p={sp:.4f} {bias}")

# Compare with existing analysis
print("\n5c. Comparison with existing centrality_all.csv (Person network):")
# Read existing centrality
existing = {}
try:
    with open(f"{ANALYSIS}/centrality_all.csv") as f:
        for row in csv.DictReader(f):
            existing[row["person_id"]] = row
    print(f"  Loaded {len(existing)} entries from centrality_all.csv")
except Exception:
    print("  Could not load centrality_all.csv")

# Print all place categories for reference
print("\n" + "=" * 100)
print("APPENDIX: All Place Classifications")
print("=" * 100)
for pid in sorted(place_nodes, key=lambda x: composite.get(x, 0), reverse=True):
    name = node_name.get(pid, pid)
    cat = place_category.get(pid, "Unknown")
    comp = composite.get(pid, 0)
    print(f"{pid:<8} {name:<45} {cat:<25} {comp:.4f}")

print("\nDone!")
