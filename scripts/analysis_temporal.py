#!/usr/bin/env python3
"""
Temporal Evolution Analysis v3 — 87.7% Event Coverage
Improved: Jaccard-weighted bipartite projection, Group exclusion, per-phase networks.
Uses ONLY social edges: TRAVEL, PARTICIPATE, LEARN, INTRODUCE.
Excludes Group (G) entities from Person-Person projection.
"""

import json
import os
from collections import Counter, defaultdict
import pandas as pd
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUT_DIR), "output")
REPORT_PATH = os.path.join(OUT_DIR, "temporal_evolution_v3.md")

# ── Phase definitions ──────────────────────────────────────────────────────
PHASES = {
    "Phase1": {
        "name": "初期 (630-701)",
        "missions": ["M01", "M02", "M03", "M04", "M05", "M06", "M07"],
        "years": "630-701",
    },
    "Phase2": {
        "name": "盛期 (702-770)",
        "missions": ["M08", "M08b", "M09", "M10"],
        "years": "702-770",
    },
    "Phase3": {
        "name": "中期 (771-834)",
        "missions": ["M11", "M12", "M13", "M14", "M15"],
        "years": "771-834",
    },
    "Phase4": {
        "name": "末期 (835-894)",
        "missions": ["M16", "M17", "M18", "M19", "M20"],
        "years": "835-894",
    },
}

# ── Load data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("TEMPORAL EVOLUTION ANALYSIS v3 — 87.7% EVENT COVERAGE")
print("=" * 70)

# Load enhanced events
with open(os.path.join(DATA_DIR, "enhanced_events.json")) as f:
    events = json.load(f)
print(f"\nLoaded {len(events)} enhanced events")

# Load nodes and edges
nodes_df = pd.read_csv(
    os.path.join(DATA_DIR, "nodes.csv"), dtype=str, keep_default_na=False
)
edges_df = pd.read_csv(
    os.path.join(DATA_DIR, "edges.csv"), dtype=str, keep_default_na=False
)

# Filter to actual entities (id starting with E,P,L,C,G)
valid_nodes = nodes_df[nodes_df["id"].str.match(r"^[EPLCG]\d+$")].copy()
print(
    f"Nodes: {len(valid_nodes)} ({', '.join(f'{t}: {len(valid_nodes[valid_nodes.type == t])}' for t in ['Person', 'Event', 'Place', 'Culture', 'Group'])})"
)
print(f"Edges: {len(edges_df)}")

# Build lookups
node_type = dict(zip(valid_nodes["id"], valid_nodes["type"]))
node_name = dict(zip(valid_nodes["id"], valid_nodes["name"]))

# Map event_id -> mission_id from enhanced_events
event_nodes = valid_nodes[valid_nodes["type"] == "Event"].copy()
event_nodes = event_nodes.sort_values("id").reset_index(drop=True)
assert len(event_nodes) == len(events), (
    f"Mismatch: {len(event_nodes)} event nodes vs {len(events)} enhanced events"
)

event_to_mission = {}
for i, (_, row) in enumerate(event_nodes.iterrows()):
    if i < len(events):
        mid = events[i].get("mission_id", "")
        if mid:
            event_to_mission[row["id"]] = mid

print(
    f"Events with mission_id label: {len(event_to_mission)} ({len(event_to_mission) / len(events) * 100:.1f}%)"
)

# Count mission coverage
mission_counts = Counter(event_to_mission.values())
for m in sorted(mission_counts.keys()):
    print(f"  {m}: {mission_counts[m]} events")

# ── Edge classification ────────────────────────────────────────────────────
META_RELS = {"HAS_SUBJECT", "HAS_OBJECT", "AT_LOCATION"}
SOCIAL_RELS = {"TRAVEL", "PARTICIPATE", "LEARN", "INTRODUCE"}

meta_edges = edges_df[edges_df["relation"].isin(META_RELS)].copy()
social_edges = edges_df[edges_df["relation"].isin(SOCIAL_RELS)].copy()

print(f"\nMetadata edges: {len(meta_edges)}")
print(f"Social edges: {len(social_edges)}")
for rel in sorted(SOCIAL_RELS):
    cnt = len(social_edges[social_edges["relation"] == rel])
    print(f"  {rel}: {cnt}")

# ── Group events by phase ──────────────────────────────────────────────────
phase_events = defaultdict(set)
phase_mission_events = defaultdict(lambda: defaultdict(set))
unlabeled_events = set()

for eid in event_nodes["id"]:
    mid = event_to_mission.get(eid, "")
    if mid:
        for phase_key, phase_info in PHASES.items():
            if mid in phase_info["missions"]:
                phase_events[phase_key].add(eid)
                phase_mission_events[phase_key][mid].add(eid)
                break
    else:
        unlabeled_events.add(eid)

print(f"\nUnlabeled events: {len(unlabeled_events)}")
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    pi = PHASES[pk]
    evts = phase_events.get(pk, set())
    mission_breakdown = ", ".join(
        f"{m}:{len(phase_mission_events[pk][m])}"
        for m in sorted(phase_mission_events[pk])
    )
    print(f"  {pi['name']}: {len(evts)} events ({mission_breakdown})")


# ── Helper: Jaccard similarity ─────────────────────────────────────────────
def jaccard(a_set, b_set):
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union > 0 else 0.0


# ── Per-phase analysis ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PER-PHASE ANALYSIS")
print("=" * 70)

all_phase_data = {}

for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    pi = PHASES[phase_key]
    evts = phase_events.get(phase_key, set())

    print(f"\n--- {pi['name']} ({pi['years']}) ---")
    print(f"  Labeled events: {len(evts)}")

    # Count event types in this phase
    phase_event_types = Counter()
    for eid in sorted(evts):
        ename = node_name.get(eid, "")
        # Extract event type from name (format: "EventType：...")
        if "：" in ename:
            etype = ename.split("：")[0]
        elif ":" in ename:
            etype = ename.split(":")[0]
        else:
            etype = "Other"
        phase_event_types[etype] += 1

    print("  Event types:")
    for et, cnt in phase_event_types.most_common():
        print(f"    {et}: {cnt}")

    # Get entities linked to these events via metadata edges
    phase_persons = set()
    phase_places = set()
    phase_cultures = set()
    phase_groups = set()

    # Via HAS_SUBJECT, HAS_OBJECT, AT_LOCATION edges
    phase_meta = meta_edges[meta_edges["source"].isin(evts)].copy()
    for _, row in phase_meta.iterrows():
        tgt = row["target"]
        tgt_type = node_type.get(tgt, "")
        if tgt_type == "Person":
            phase_persons.add(tgt)
        elif tgt_type == "Place":
            phase_places.add(tgt)
        elif tgt_type == "Culture":
            phase_cultures.add(tgt)
        elif tgt_type == "Group":
            phase_groups.add(tgt)

    # Also check source column of meta edges if event is target
    phase_meta2 = meta_edges[meta_edges["target"].isin(evts)].copy()
    for _, row in phase_meta2.iterrows():
        src = row["source"]
        src_type = node_type.get(src, "")
        if src_type == "Person":
            phase_persons.add(src)
        elif src_type == "Place":
            phase_places.add(src)
        elif src_type == "Culture":
            phase_cultures.add(src)
        elif src_type == "Group":
            phase_groups.add(src)

    # Via social edges involving these events
    for _, row in social_edges.iterrows():
        src_in = row["source"] in evts
        tgt_in = row["target"] in evts
        if src_in or tgt_in:
            for col in ["source", "target"]:
                nid = row[col]
                nt = node_type.get(nid, "")
                if nt == "Person":
                    phase_persons.add(nid)
                elif nt == "Place":
                    phase_places.add(nid)
                elif nt == "Culture":
                    phase_cultures.add(nid)
                elif nt == "Group":
                    phase_groups.add(nid)

    print(f"  Persons: {len(phase_persons)}")
    print(f"  Places: {len(phase_places)}")
    print(f"  Cultures: {len(phase_cultures)}")
    print(f"  Groups: {len(phase_groups)}")

    # Build phase-specific social edges
    # Person->Place/Event (TRAVEL + PARTICIPATE)
    phase_travel_edges = []
    for _, row in social_edges[
        social_edges["relation"].isin(["TRAVEL", "PARTICIPATE"])
    ].iterrows():
        src, tgt = row["source"], row["target"]
        src_t = node_type.get(src, "")
        tgt_t = node_type.get(tgt, "")
        # At least one endpoint is a Person, and no Group involved
        if (
            (src_t == "Person" or tgt_t == "Person")
            and src_t != "Group"
            and tgt_t != "Group"
        ):
            # Only include if the person is in this phase
            if src in phase_persons or tgt in phase_persons:
                phase_travel_edges.append(row)

    # Person->Culture (LEARN + INTRODUCE)
    phase_culture_edges = []
    for _, row in social_edges[
        social_edges["relation"].isin(["LEARN", "INTRODUCE"])
    ].iterrows():
        src, tgt = row["source"], row["target"]
        src_t = node_type.get(src, "")
        tgt_t = node_type.get(tgt, "")
        if (
            (src_t == "Person" or tgt_t == "Person")
            and src_t != "Group"
            and tgt_t != "Group"
        ):
            if src in phase_persons or tgt in phase_persons:
                phase_culture_edges.append(row)

    travel_count = len(phase_travel_edges)
    learn_count = sum(1 for e in phase_culture_edges if e["relation"] == "LEARN")
    intro_count = sum(1 for e in phase_culture_edges if e["relation"] == "INTRODUCE")

    print(f"  TRAVEL+PARTICIPATE edges: {travel_count}")
    print(f"  LEARN: {learn_count}, INTRODUCE: {intro_count}")
    li_ratio = f"{learn_count / intro_count:.2f}" if intro_count > 0 else "N/A"
    if intro_count > 0:
        print(f"  LEARN/INTRODUCE ratio: {li_ratio}")

    # ── Build Person-Person Jaccard network for this phase ──
    # Person -> Places/Events (TRAVEL + PARTICIPATE)
    person_to_targets = defaultdict(set)
    for row in phase_travel_edges:
        src, tgt = row["source"], row["target"]
        src_t = node_type.get(src, "")
        tgt_t = node_type.get(tgt, "")
        if src_t == "Person" and tgt_t != "Group":
            person_to_targets[src].add(tgt)
        if tgt_t == "Person" and src_t != "Group":
            person_to_targets[tgt].add(src)

    # Person -> Cultures (LEARN + INTRODUCE)
    person_to_cultures = defaultdict(set)
    for row in phase_culture_edges:
        src, tgt = row["source"], row["target"]
        src_t = node_type.get(src, "")
        tgt_t = node_type.get(tgt, "")
        if src_t == "Person" and tgt_t != "Group":
            person_to_cultures[src].add(tgt)
        if tgt_t == "Person" and src_t != "Group":
            person_to_cultures[tgt].add(src)

    # All persons with any activity (EXCLUDING Groups)
    all_phase_persons = set()
    all_phase_persons.update(person_to_targets.keys())
    all_phase_persons.update(person_to_cultures.keys())
    # Remove any Group entities that might have slipped in
    all_phase_persons = {p for p in all_phase_persons if node_type.get(p) == "Person"}

    # Build Jaccard-weighted Person-Person network
    G_pp = nx.Graph()
    for p in all_phase_persons:
        G_pp.add_node(p, name=node_name.get(p, p), type="Person")

    p_list = sorted(all_phase_persons)
    for i in range(len(p_list)):
        for j in range(i + 1, len(p_list)):
            pa, pb = p_list[i], p_list[j]
            # Combined Jaccard: union of place+culture sets
            si = person_to_targets.get(pa, set()) | person_to_cultures.get(pa, set())
            sj = person_to_targets.get(pb, set()) | person_to_cultures.get(pb, set())
            jw = jaccard(si, sj)
            if jw > 0:
                G_pp.add_edge(pa, pb, weight=jw)

    density = nx.density(G_pp) if G_pp.number_of_nodes() > 1 else 0
    print(
        f"  Person-Person network: {G_pp.number_of_nodes()} nodes, {G_pp.number_of_edges()} edges, density={density:.6f}"
    )

    # Centrality (if network has edges) - Top 10
    top10_wdegree = []
    top10_betweenness = []
    top10_pagerank = []

    if G_pp.number_of_edges() > 0:
        # Weighted degree
        wdegree = {
            n: sum(d["weight"] for _, _, d in G_pp.edges(n, data=True))
            for n in G_pp.nodes()
        }

        # Betweenness (use weight as distance: higher weight = shorter path → invert)
        G_bt = G_pp.copy()
        for u, v, d in G_bt.edges(data=True):
            d["weight"] = 1.0 / max(d["weight"], 1e-10)
        betweenness = nx.betweenness_centrality(G_bt, weight="weight", normalized=True)

        # PageRank (higher weight = more important)
        pagerank = nx.pagerank(G_pp, weight="weight")

        top10_wdegree = sorted(wdegree.items(), key=lambda x: -x[1])[:10]
        top10_betweenness = sorted(betweenness.items(), key=lambda x: -x[1])[:10]
        top10_pagerank = sorted(pagerank.items(), key=lambda x: -x[1])[:10]

        print("  Top-10 Weighted Degree:")
        for n, v in top10_wdegree:
            print(f"    {node_name.get(n, n)}: {v:.6f}")
        print("  Top-10 Betweenness:")
        for n, v in top10_betweenness:
            print(f"    {node_name.get(n, n)}: {v:.6f}")
        print("  Top-10 PageRank:")
        for n, v in top10_pagerank:
            print(f"    {node_name.get(n, n)}: {v:.6f}")

    all_phase_data[phase_key] = {
        "phase_info": pi,
        "events": len(evts),
        "event_types": dict(phase_event_types),
        "persons": len(phase_persons),
        "places": len(phase_places),
        "cultures": len(phase_cultures),
        "groups": len(phase_groups),
        "travel_edges": travel_count,
        "learn_count": learn_count,
        "intro_count": intro_count,
        "li_ratio": li_ratio,
        "person_person_nodes": G_pp.number_of_nodes(),
        "person_person_edges": G_pp.number_of_edges(),
        "density": density,
        "top10_wdegree": top10_wdegree,
        "top10_betweenness": top10_betweenness,
        "top10_pagerank": top10_pagerank,
        "phase_persons": phase_persons,
        "phase_places": phase_places,
        "phase_cultures": phase_cultures,
        "phase_groups": phase_groups,
        "person_to_targets": person_to_targets,
        "person_to_cultures": person_to_cultures,
        "G_pp": G_pp,
    }

# ── New entities per phase ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("NEW ENTITIES PER PHASE")
print("=" * 70)

cumulative_persons = set()
cumulative_places = set()
cumulative_cultures = set()
cumulative_groups = set()
cumulative_events_set = set()

new_entity_data = {}

for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    pd_data = all_phase_data[phase_key]
    pi = pd_data["phase_info"]

    new_persons = pd_data["phase_persons"] - cumulative_persons
    new_places = pd_data["phase_places"] - cumulative_places
    new_cultures = pd_data["phase_cultures"] - cumulative_cultures
    new_groups = pd_data["phase_groups"] - cumulative_groups
    new_events = phase_events.get(phase_key, set()) - cumulative_events_set

    print(f"\n{pi['name']}:")
    print(
        f"  New events: {len(new_events)} (+{len(new_events)}, cum={len(cumulative_events_set) + len(new_events)})"
    )
    print(f"  New persons: {len(new_persons)}")
    if new_persons:
        names = [node_name.get(p, p) for p in sorted(new_persons)[:10]]
        print(f"    Examples: {', '.join(names)}")
    print(f"  New places: {len(new_places)}")
    if new_places:
        names = [node_name.get(p, p) for p in sorted(new_places)[:10]]
        print(f"    Examples: {', '.join(names)}")
    print(f"  New cultures: {len(new_cultures)}")
    if new_cultures:
        names = [node_name.get(c, c) for c in sorted(new_cultures)[:10]]
        print(f"    Examples: {', '.join(names)}")
    print(f"  New groups: {len(new_groups)}")

    cumulative_persons |= pd_data["phase_persons"]
    cumulative_places |= pd_data["phase_places"]
    cumulative_cultures |= pd_data["phase_cultures"]
    cumulative_groups |= pd_data["phase_groups"]
    cumulative_events_set |= phase_events.get(phase_key, set())

    print(
        f"  Cumulative: {len(cumulative_persons)}P, {len(cumulative_places)}L, {len(cumulative_cultures)}C, {len(cumulative_groups)}G, {len(cumulative_events_set)}E"
    )

    new_entity_data[phase_key] = {
        "new_events": len(new_events),
        "new_persons": len(new_persons),
        "new_places": len(new_places),
        "new_cultures": len(new_cultures),
        "new_groups": len(new_groups),
        "cum_events": len(cumulative_events_set),
        "cum_persons": len(cumulative_persons),
        "cum_places": len(cumulative_places),
        "cum_cultures": len(cumulative_cultures),
        "cum_groups": len(cumulative_groups),
    }

# ── Key persons across phases ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY PERSONS ACTIVITY ACROSS PHASES")
print("=" * 70)

# Identify top persons from all-phase combined data
all_person_activity = Counter()
for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    all_person_activity.update(all_phase_data[phase_key]["phase_persons"])

# Also count connections per person
person_conn_count = Counter()
for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    pd_data = all_phase_data[phase_key]
    for p in pd_data["phase_persons"]:
        person_conn_count[p] += len(pd_data["person_to_targets"].get(p, set()))
        person_conn_count[p] += len(pd_data["person_to_cultures"].get(p, set()))

# Top key persons
key_persons = sorted(
    all_person_activity.items(), key=lambda x: (-x[1], -person_conn_count.get(x[0], 0))
)[:20]
key_person_ids = [p for p, _ in key_persons]

print("\nTop key persons (by phase presence & activity):")
for pid, count in key_persons[:20]:
    name = node_name.get(pid, pid)
    print(
        f"  {name} ({pid}): {count} phases, {person_conn_count.get(pid, 0)} connections"
    )

# Activity matrix: key persons x phases
print(f"\n{'Person':<22}", end="")
for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    print(f"{PHASES[phase_key]['name']:>22}", end="")
print(f"  {'Phases':>6}")
print("-" * 110)

for pid in key_person_ids[:20]:
    name = node_name.get(pid, pid)
    print(f"{name[:20]:<22}", end="")
    phase_count = 0
    for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
        present = pid in all_phase_data[phase_key]["phase_persons"]
        if present:
            phase_count += 1
        n_conn = len(
            all_phase_data[phase_key]["person_to_targets"].get(pid, set())
        ) + len(all_phase_data[phase_key]["person_to_cultures"].get(pid, set()))
        cell = f"{'✓' if present else '-'}({n_conn})" if present else "-"
        print(f"{cell:>22}", end="")
    print(f"  {phase_count:>4}")

# ── Special tracking for named key figures ─────────────────────────────────
print("\n--- Named Key Figures Tracking ---")
key_names = [
    "空海",
    "最澄",
    "円仁",
    "道昭",
    "宗叡",
    "吉備真備",
    "阿倍仲麻呂",
    "藤原清河",
    "小野妹子",
    "高向玄理",
    "粟田真人",
    "山上憶良",
    "鑑真",
    "圓仁",
    "圓珍",
    "常暁",
    "恵萼",
]
for lookup_name in key_names:
    found = False
    for pid in key_person_ids:
        name = node_name.get(pid, "")
        if lookup_name in name:
            phases_present = []
            conns = 0
            for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
                if pid in all_phase_data[phase_key]["phase_persons"]:
                    phases_present.append(PHASES[phase_key]["name"])
                    conns += len(
                        all_phase_data[phase_key]["person_to_targets"].get(pid, set())
                    )
                    conns += len(
                        all_phase_data[phase_key]["person_to_cultures"].get(pid, set())
                    )
            print(f"  {name}: phases={phases_present}, total_connections={conns}")
            found = True
            break
    if not found:
        # Try fuzzy match
        for pid, nname in node_name.items():
            if node_type.get(pid) == "Person" and lookup_name in nname:
                print(f"  (fuzzy) {nname} ({pid})")
                found = True
                break
        if not found:
            print(f"  {lookup_name}: NOT FOUND in network")

# ── Temporal trends ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEMPORAL TRENDS")
print("=" * 70)

trend_data = []
for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[phase_key]
    nd = new_entity_data[phase_key]
    total_social = d["travel_edges"] + d["learn_count"] + d["intro_count"]
    trend_data.append(
        {
            "phase": d["phase_info"]["name"],
            "events": d["events"],
            "pp_nodes": d["person_person_nodes"],
            "pp_edges": d["person_person_edges"],
            "density": d["density"],
            "li_ratio": d["li_ratio"],
            "learn": d["learn_count"],
            "introduce": d["intro_count"],
            "travel_participate": d["travel_edges"],
            "total_social": total_social,
            "new_events": nd["new_events"],
            "new_persons": nd["new_persons"],
            "cum_events": nd["cum_events"],
            "cum_persons": nd["cum_persons"],
        }
    )

print(
    f"\n{'Phase':<22} {'Events':>7} {'PP_Nodes':>8} {'PP_Edges':>8} {'Density':>10} {'L/I':>8} {'Learn':>6} {'Intro':>6} {'Travel+P':>8} {'TotalSoc':>8} {'NewEvt':>7} {'NewPer':>7}"
)
print("-" * 130)
for t in trend_data:
    print(
        f"{t['phase']:<22} {t['events']:>7} {t['pp_nodes']:>8} {t['pp_edges']:>8} {t['density']:>10.6f} {t['li_ratio']:>8} {t['learn']:>6} {t['introduce']:>6} {t['travel_participate']:>8} {t['total_social']:>8} {t['new_events']:>7} {t['new_persons']:>7}"
    )

# ═══════════════════════════════════════════════════════════════════════════
# GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GENERATING REPORT")
print("=" * 70)

report = []
report.append("# 遣唐使网络 — 时间演化分析报告 v3 (87.7%事件覆盖率)")
report.append("")
report.append(
    "*Generated: 2026-06-30 — Using enhanced mission batch labels, Jaccard-weighted bipartite projection, Group entities excluded*"
)
report.append("")
report.append("## 1. 概述")
report.append("")
report.append(
    f"本报告使用 `enhanced_events.json` ({len(events)}个事件，其中{len(event_to_mission)}个有mission_id标注，覆盖率{len(event_to_mission) / len(events) * 100:.1f}%)，按遣唐使批次将事件分为四个时间阶段。"
)
report.append(
    "网络构建采用 **Jaccard加权二部投影**，仅使用社交边 (TRAVEL, PARTICIPATE, LEARN, INTRODUCE)，**排除Group(G)类型实体**。"
)
report.append("")
report.append(
    f"- **总节点**: {len(valid_nodes)} (Person: {len(valid_nodes[valid_nodes.type == 'Person'])}, Event: {len(valid_nodes[valid_nodes.type == 'Event'])}, Place: {len(valid_nodes[valid_nodes.type == 'Place'])}, Culture: {len(valid_nodes[valid_nodes.type == 'Culture'])}, Group: {len(valid_nodes[valid_nodes.type == 'Group'])})"
)
report.append(
    f"- **总边**: {len(edges_df)} (Metadata: {len(meta_edges)}, Social: {len(social_edges)})"
)
report.append(
    f"- **社交边分布**: TRAVEL={len(social_edges[social_edges.relation == 'TRAVEL'])}, PARTICIPATE={len(social_edges[social_edges.relation == 'PARTICIPATE'])}, LEARN={len(social_edges[social_edges.relation == 'LEARN'])}, INTRODUCE={len(social_edges[social_edges.relation == 'INTRODUCE'])}"
)
report.append(
    f"- **未标注事件**: {len(unlabeled_events)}（无mission_id，不参与阶段分析）"
)
report.append(
    f"- **Group实体**: {len(valid_nodes[valid_nodes.type == 'Group'])} (已排除，不参与Person-Person投影)"
)
report.append("")

# Phase definitions table
report.append("## 2. 时间阶段定义")
report.append("")
report.append("| 阶段 | 时期 | 年份 | 批次 (事件数) |")
report.append("|------|------|------|---------------|")
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    pi = PHASES[pk]
    evts = phase_events.get(pk, set())
    present_missions = [m for m in pi["missions"] if m in mission_counts]
    mission_detail = ", ".join(f"{m}({mission_counts[m]})" for m in present_missions)
    if not present_missions:
        mission_detail = "*(无数据)*"
    report.append(
        f"| {pi['name']} | {pi['years']} | {', '.join(pi['missions'])} | {mission_detail} (共{len(evts)}事件) |"
    )
report.append("")

# Per-phase statistics
report.append("## 3. 各阶段统计")
report.append("")
report.append("### 3.1 事件类型分布")
report.append("")
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[pk]
    report.append(f"**{d['phase_info']['name']}** ({d['phase_info']['years']}):")
    report.append("")
    report.append("| 事件类型 | 数量 |")
    report.append("|----------|------|")
    for et, cnt in sorted(d["event_types"].items(), key=lambda x: -x[1]):
        report.append(f"| {et} | {cnt} |")
    report.append("")

# Entity counts
report.append("### 3.2 实体与网络规模")
report.append("")
report.append(
    "| 阶段 | 事件 | 人物 | 地点 | 文化 | Group | TRAVEL+P | LEARN | INTRO | L/I比 | PP节点 | PP边 | 密度 |"
)
report.append(
    "|------|------|------|------|------|-------|----------|-------|-------|-------|--------|------|------|"
)
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[pk]
    report.append(
        f"| {d['phase_info']['name']} | {d['events']} | {d['persons']} | {d['places']} | {d['cultures']} | {d['groups']} | "
        f"{d['travel_edges']} | {d['learn_count']} | {d['intro_count']} | {d['li_ratio']} | "
        f"{d['person_person_nodes']} | {d['person_person_edges']} | {d['density']:.6f} |"
    )
report.append("")

# LEARN+INTRODUCE analysis
report.append("### 3.3 LEARN+INTRODUCE 边分析")
report.append("")
report.append(
    "LEARN边表示师从学习关系，INTRODUCE边表示文化传播/引入关系。L/I比反映各阶段学习传播活动的侧重。"
)
report.append("")
report.append("| 阶段 | LEARN | INTRODUCE | 合计(L+I) | L/I比 | 占社交边比例 |")
report.append("|------|-------|-----------|-----------|-------|-------------|")
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[pk]
    total_social = d["travel_edges"] + d["learn_count"] + d["intro_count"]
    li_combined = d["learn_count"] + d["intro_count"]
    li_pct = f"{li_combined / total_social * 100:.1f}%" if total_social > 0 else "N/A"
    report.append(
        f"| {d['phase_info']['name']} | {d['learn_count']} | {d['intro_count']} | {li_combined} | {d['li_ratio']} | {li_pct} |"
    )
report.append("")

# Centrality per phase - Top 10
report.append("## 4. 各阶段中心性 Top-10")
report.append("")
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[pk]
    report.append(
        f"### 4.{list(PHASES.keys()).index(pk) + 1} {d['phase_info']['name']}"
    )
    report.append("")

    if d["top10_wdegree"]:
        report.append("**加权度中心性 (Weighted Degree):**")
        report.append("")
        report.append("| 排名 | 人物 | 值 |")
        report.append("|------|------|-----|")
        for i, (n, v) in enumerate(d["top10_wdegree"], 1):
            report.append(f"| {i} | {node_name.get(n, n)} | {v:.6f} |")
        report.append("")

        report.append("**中介中心性 (Betweenness):**")
        report.append("")
        report.append("| 排名 | 人物 | 值 |")
        report.append("|------|------|-----|")
        for i, (n, v) in enumerate(d["top10_betweenness"], 1):
            report.append(f"| {i} | {node_name.get(n, n)} | {v:.6f} |")
        report.append("")

        report.append("**PageRank:**")
        report.append("")
        report.append("| 排名 | 人物 | 值 |")
        report.append("|------|------|-----|")
        for i, (n, v) in enumerate(d["top10_pagerank"], 1):
            report.append(f"| {i} | {node_name.get(n, n)} | {v:.6f} |")
        report.append("")
    else:
        report.append("*(该阶段网络无边，数据不足)*")
        report.append("")

# New entities
report.append("## 5. 新增实体与累积增长")
report.append("")
report.append(
    "| 阶段 | 新增事件 | 新增人物 | 新增地点 | 新增文化 | 新增Group | 累积事件 | 累积人物 | 累积地点 | 累积文化 | 累积Group |"
)
report.append(
    "|------|----------|----------|----------|----------|-----------|----------|----------|----------|----------|-----------|"
)
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    nd = new_entity_data[pk]
    report.append(
        f"| {all_phase_data[pk]['phase_info']['name']} | {nd['new_events']} | {nd['new_persons']} | {nd['new_places']} | {nd['new_cultures']} | {nd['new_groups']} | "
        f"{nd['cum_events']} | {nd['cum_persons']} | {nd['cum_places']} | {nd['cum_cultures']} | {nd['cum_groups']} |"
    )
report.append("")

# Cross-phase key person tracking
report.append("## 6. 跨阶段关键人物追踪")
report.append("")
report.append("### 6.1 核心人物活动矩阵")
report.append("")
report.append(
    "下表展示Top-20关键人物在各阶段的活跃情况（✓=出现在该阶段，括号内为连接数）:"
)
report.append("")
report.append(
    f"| 人物 | {PHASES['Phase1']['name']} | {PHASES['Phase2']['name']} | {PHASES['Phase3']['name']} | {PHASES['Phase4']['name']} | 活跃阶段数 |"
)
report.append("|------|" + "|".join(["------"] * 5) + "|")
for pid in key_person_ids[:20]:
    name = node_name.get(pid, pid)
    cells = []
    phase_count = 0
    for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
        present = pid in all_phase_data[phase_key]["phase_persons"]
        if present:
            phase_count += 1
            n_conn = len(
                all_phase_data[phase_key]["person_to_targets"].get(pid, set())
            ) + len(all_phase_data[phase_key]["person_to_cultures"].get(pid, set()))
            cells.append(f"✓({n_conn})")
        else:
            cells.append("-")
    report.append(f"| {name} | {' | '.join(cells)} | {phase_count} |")
report.append("")

# Named key figures
report.append("### 6.2 命名关键人物详细追踪")
report.append("")
report.append("以下是特别关注的遣唐使关键人物的阶段分布与连接数:")
report.append("")
report.append("| 人物 | 活跃阶段 | 总连接数 |")
report.append("|------|----------|----------|")
for lookup_name in [
    "空海",
    "最澄",
    "円仁",
    "道昭",
    "宗叡",
    "吉備真備",
    "阿倍仲麻呂",
    "藤原清河",
    "小野妹子",
    "高向玄理",
    "粟田真人",
    "山上憶良",
    "鑑真",
    "常暁",
    "恵萼",
]:
    found_info = None
    for pid in key_person_ids:
        name = node_name.get(pid, "")
        if lookup_name in name:
            phases_present = []
            total_conns = 0
            for phase_key in ["Phase1", "Phase2", "Phase3", "Phase4"]:
                if pid in all_phase_data[phase_key]["phase_persons"]:
                    phases_present.append(PHASES[phase_key]["name"].split(" ")[0])
                    total_conns += len(
                        all_phase_data[phase_key]["person_to_targets"].get(pid, set())
                    )
                    total_conns += len(
                        all_phase_data[phase_key]["person_to_cultures"].get(pid, set())
                    )
            found_info = (
                name,
                ", ".join(phases_present) if phases_present else "无",
                str(total_conns),
            )
            break
    if not found_info:
        # Try fuzzy
        for pid, nname in node_name.items():
            if node_type.get(pid) == "Person" and lookup_name in nname:
                found_info = (nname, "未在关键人物列表", "?")
                break
    if found_info:
        report.append(f"| {found_info[0]} | {found_info[1]} | {found_info[2]} |")
    else:
        report.append(f"| {lookup_name} | 未找到 | - |")
report.append("")

# Temporal trends
report.append("## 7. 时间趋势分析")
report.append("")
report.append("### 7.1 网络规模与密度趋势")
report.append("")
report.append("| 阶段 | PP节点 | PP边 | 密度 | 社交边总数 |")
report.append("|------|--------|------|------|-----------|")
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[pk]
    total_social = d["travel_edges"] + d["learn_count"] + d["intro_count"]
    report.append(
        f"| {d['phase_info']['name']} | {d['person_person_nodes']} | {d['person_person_edges']} | {d['density']:.6f} | {total_social} |"
    )
report.append("")

report.append("### 7.2 L/I比趋势")
report.append("")
report.append("| 阶段 | LEARN | INTRODUCE | L/I比 |")
report.append("|------|-------|-----------|-------|")
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[pk]
    report.append(
        f"| {d['phase_info']['name']} | {d['learn_count']} | {d['intro_count']} | {d['li_ratio']} |"
    )
report.append("")

report.append("### 7.3 实体增长趋势")
report.append("")
report.append(
    "| 阶段 | 新增事件 | 新增人物 | 累积事件 | 累积人物 | 事件增长率 | 人物增长率 |"
)
report.append(
    "|------|----------|----------|----------|----------|-----------|-----------|"
)
prev_cum_e = 0
prev_cum_p = 0
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    nd = new_entity_data[pk]
    e_growth = (
        f"{(nd['cum_events'] - prev_cum_e) / max(prev_cum_e, 1) * 100:.1f}%"
        if prev_cum_e > 0
        else "—"
    )
    p_growth = (
        f"{(nd['cum_persons'] - prev_cum_p) / max(prev_cum_p, 1) * 100:.1f}%"
        if prev_cum_p > 0
        else "—"
    )
    report.append(
        f"| {all_phase_data[pk]['phase_info']['name']} | {nd['new_events']} | {nd['new_persons']} | {nd['cum_events']} | {nd['cum_persons']} | {e_growth} | {p_growth} |"
    )
    prev_cum_e = nd["cum_events"]
    prev_cum_p = nd["cum_persons"]
report.append("")

# Summary findings
report.append("## 8. 主要发现与结论")
report.append("")

# Count how many phases have data
phases_with_data = sum(
    1
    for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]
    if all_phase_data[pk]["events"] > 0
)
report.append("### 8.1 数据覆盖")
report.append(
    f"- 472个事件中共有414个({414 / 472 * 100:.1f}%)事件标注了mission_id，分布于{len(mission_counts)}个遣唐使批次"
)
report.append(
    f"- 4个时间阶段中{'全部' if phases_with_data == 4 else f'{phases_with_data}个'}有数据"
)
report.append(f"- {len(unlabeled_events)}个事件未标注，不参与阶段分析")
report.append("")

report.append("### 8.2 网络演化特征")
# Find peak phase
peak_phase = max(
    ["Phase1", "Phase2", "Phase3", "Phase4"],
    key=lambda pk: all_phase_data[pk]["person_person_edges"],
)
report.append(
    f"- **网络规模**: 从Phase1的{all_phase_data['Phase1']['person_person_nodes']}人/{all_phase_data['Phase1']['person_person_edges']}边，发展到Phase4的{all_phase_data['Phase4']['person_person_nodes']}人/{all_phase_data['Phase4']['person_person_edges']}边"
)
report.append(
    f"- **网络密度**: Phase1={all_phase_data['Phase1']['density']:.6f}, Phase2={all_phase_data['Phase2']['density']:.6f}, Phase3={all_phase_data['Phase3']['density']:.6f}, Phase4={all_phase_data['Phase4']['density']:.6f}"
)
report.append(
    f"- **最大网络**: {all_phase_data[peak_phase]['phase_info']['name']} (PP边={all_phase_data[peak_phase]['person_person_edges']})"
)
report.append("")

report.append("### 8.3 LEARN/INTRODUCE 关系演变")
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[pk]
    if d["intro_count"] > 0:
        report.append(
            f"- {d['phase_info']['name']}: LEARN={d['learn_count']}, INTRODUCE={d['intro_count']}, L/I={d['li_ratio']}"
        )
    else:
        report.append(
            f"- {d['phase_info']['name']}: LEARN={d['learn_count']}, INTRODUCE={d['intro_count']}, L/I=N/A"
        )
report.append("")

report.append("### 8.4 Group实体排除说明")
report.append(
    f"- 本分析排除了所有{len(valid_nodes[valid_nodes.type == 'Group'])}个Group类型实体"
)
report.append("- Group实体代表船队、使团等集合体，不反映真实个人关系")
report.append("- 排除Group后，Person-Person网络更准确地反映个人间的历史关联")
report.append("")

# Write report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"\nReport written to {REPORT_PATH}")
print(f"Report length: {len(report)} lines")

# Also save CSV exports
# Per-phase centrality
for pk in ["Phase1", "Phase2", "Phase3", "Phase4"]:
    d = all_phase_data[pk]
    G = d["G_pp"]
    if G.number_of_edges() > 0:
        rows = []
        wdegree = {
            n: sum(dw["weight"] for _, _, dw in G.edges(n, data=True))
            for n in G.nodes()
        }
        G_bt = G.copy()
        for u, v, dw in G_bt.edges(data=True):
            dw["weight"] = 1.0 / max(dw["weight"], 1e-10)
        betweenness = nx.betweenness_centrality(G_bt, weight="weight", normalized=True)
        pagerank = nx.pagerank(G, weight="weight")
        for n in G.nodes():
            rows.append(
                {
                    "person_id": n,
                    "person_name": node_name.get(n, n),
                    "weighted_degree": wdegree.get(n, 0),
                    "betweenness": betweenness.get(n, 0),
                    "pagerank": pagerank.get(n, 0),
                }
            )
        df_c = pd.DataFrame(rows)
        df_c = df_c.sort_values("weighted_degree", ascending=False)
        df_c.to_csv(os.path.join(OUT_DIR, f"centrality_{pk}_v3.csv"), index=False)
        print(f"Saved centrality_{pk}_v3.csv ({len(df_c)} persons)")

print("\nDone!")
