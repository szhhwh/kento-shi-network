#!/usr/bin/env python3
"""
Place and Culture network analysis v2 - corrected
"""

import csv
import json
from collections import defaultdict, Counter

DATA = "/home/szhh/kento-shi-network/output"

# Load nodes
nodes = {}
with open(f"{DATA}/nodes.csv") as f:
    for row in csv.DictReader(f):
        t = row["type"].strip()
        rid = row["id"].strip()
        nodes[rid] = {"type": t, "name": row["name"].strip()}

# Load edges
edges_raw = []
with open(f"{DATA}/edges.csv") as f:
    for row in csv.DictReader(f):
        rel = row["relation"].strip()
        if rel:
            edges_raw.append(
                {
                    "source": row["source"].strip(),
                    "target": row["target"].strip(),
                    "relation": rel,
                }
            )

# Load events
with open(f"{DATA}/enhanced_events.json") as f:
    events_list = json.load(f)
event_by_id = {}
for i, ev in enumerate(events_list):
    event_by_id[f"E{i + 1:03d}"] = ev


# Node type helpers
def typ(nid):
    return nodes.get(nid, {}).get("type", "")


def nam(nid):
    return nodes.get(nid, {}).get("name", nid)


person_ids = [n for n, d in nodes.items() if d["type"] == "Person"]
place_ids = [n for n, d in nodes.items() if d["type"] == "Place"]
culture_ids = [n for n, d in nodes.items() if d["type"] == "Culture"]

# ============================================================
# Build adjacency lookups
# ============================================================
person_places = defaultdict(set)  # person -> {places} from TRAVEL
person_cultures = defaultdict(set)  # person -> {cultures} from INTRODUCE
# Also: event -> {subjects}, event -> {locations}
event_subjects = defaultdict(set)
event_locations = defaultdict(set)

for e in edges_raw:
    s, t, r = e["source"], e["target"], e["relation"]
    if r == "TRAVEL":
        if typ(s) in ("Person", "Group") and typ(t) == "Place":
            person_places[s].add(t)
        elif typ(t) in ("Person", "Group") and typ(s) == "Place":
            person_places[t].add(s)
    elif r == "INTRODUCE":
        if typ(s) == "Person" and typ(t) == "Culture":
            person_cultures[s].add(t)
        elif typ(t) == "Person" and typ(s) == "Culture":
            person_cultures[t].add(s)
    elif r == "HAS_SUBJECT":
        if s.startswith("E"):
            event_subjects[s].add(t)
    elif r == "AT_LOCATION":
        if s.startswith("E"):
            event_locations[s].add(t)

print(f"Person-place links: {sum(len(v) for v in person_places.values())}")
print(f"Person-culture links: {sum(len(v) for v in person_cultures.values())}")

# ============================================================
# 1. PLACE HUB RANKING (Place-Place via co-travel)
# ============================================================
print("\n" + "=" * 60)
print("1. PLACE HUB RANKING")
print("=" * 60)

# Build Place-Place weighted graph manually (no networkx needed)
place_edges = defaultdict(lambda: defaultdict(int))
for p, places in person_places.items():
    plist = list(places)
    for i in range(len(plist)):
        for j in range(i + 1, len(plist)):
            a, b = sorted([plist[i], plist[j]])
            place_edges[a][b] += 1

# Count unique edges and compute degree
place_degree = Counter()
for a, neighbors in place_edges.items():
    for b, w in neighbors.items():
        place_degree[a] += 1
        place_degree[b] += 1

place_wdeg = Counter()
for a, neighbors in place_edges.items():
    for b, w in neighbors.items():
        place_wdeg[a] += w
        place_wdeg[b] += w

print(
    f"Place-Place graph: {len(place_ids)} nodes, {sum(len(v) for v in place_edges.values())} edges"
)
print(f"Places with connections: {len(place_degree)}")

# Rank by weighted degree (proxy for hub-ness without betweenness)
print("\nTop 20 Places by Weighted Degree (co-travel frequency):")
for pid, wd in place_wdeg.most_common(20):
    print(f"  {nam(pid):30s}  wdeg={wd:4d}  deg={place_degree.get(pid, 0):3d}")

# Compute betweenness centrality manually (Brandes algorithm simplified)
# Actually let's just do a simple betweenness approximation
# for small graphs, or use degree as proxy.
# The graph is dense enough that degree-centrality is informative.

# Also rank by pure degree
print("\nTop 20 Places by Degree (number of co-visited places):")
for pid, d in place_degree.most_common(20):
    print(f"  {nam(pid):30s}  deg={d:3d}  wdeg={place_wdeg.get(pid, 0):4d}")

# ============================================================
# 2. CULTURE CO-OCCURRENCE
# ============================================================
print("\n" + "=" * 60)
print("2. CULTURE CO-OCCURRENCE")
print("=" * 60)

print(f"\nPersons who INTRODUCE: {len(person_cultures)}")
for p, cults in sorted(person_cultures.items(), key=lambda x: len(x[1]), reverse=True):
    cnames = [nam(c) for c in cults]
    print(f"  {nam(p)} ({len(cults)}): {', '.join(cnames)}")

# Culture-Culture edges
culture_edges = defaultdict(lambda: defaultdict(list))
for p, cults in person_cultures.items():
    clist = sorted(cults)
    for i in range(len(clist)):
        for j in range(i + 1, len(clist)):
            a, b = clist[i], clist[j]
            culture_edges[a][b].append(p)
            culture_edges[b][a].append(p)

print("\nCulture-Culture co-introduction pairs:")
edges_flat = []
for a, neighbors in culture_edges.items():
    for b, persons in neighbors.items():
        if a < b:
            edges_flat.append((a, b, len(persons), [nam(p) for p in persons]))
edges_flat.sort(key=lambda x: x[2], reverse=True)

for i, (a, b, w, ps) in enumerate(edges_flat, 1):
    print(f"  {i}. {nam(a)[:50]} + {nam(b)[:50]}  (w={w})  by: {ps}")

if not edges_flat:
    print("  (No co-introductions - each culture introduced by exactly 1 person)")

# ============================================================
# 3. ROUTE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("3. ROUTE ANALYSIS")
print("=" * 60)

# For each event, get its subjects and location, then use mission year for ordering.
# Build: (person_id, mission_year, event_index) -> place
person_event_places = defaultdict(
    list
)  # person -> [(event_idx, location, mission_id, year)]

# Parse mission years
mission_years = {}
for ev in events_list:
    mid = ev.get("mission_id", "")
    ys = ev.get("mission_departure_year", "")
    if mid and ys:
        try:
            y = int(str(ys).split("～")[0].split("-")[0].strip())
            if mid not in mission_years:
                mission_years[mid] = y
        except (ValueError, KeyError):
            pass

print(f"Mission years parsed: {len(mission_years)} missions")
for mid in sorted(mission_years, key=lambda x: mission_years[x])[:10]:
    print(f"  {mid}: {mission_years[mid]}")

# Now link persons to places via events, with event ordering
# Find: for each person, list of (event_idx, place, mission_id_year)
for eid, ev in event_by_id.items():
    ei = int(eid[1:]) - 1
    subjs = event_subjects.get(eid, set())
    locs = event_locations.get(eid, set())
    mid = ev.get("mission_id", "")
    yr = mission_years.get(mid, None)

    for subj in subjs:
        for loc in locs:
            if typ(loc) == "Place":
                person_event_places[subj].append((ei, loc, mid, yr))

print(f"\nPersons with event-linked place visits: {len(person_event_places)}")

# For each person, sort by event index and extract unique place sequences
person_routes = {}
for person, trips in person_event_places.items():
    trips.sort(key=lambda x: x[0])  # sort by event index
    route = []
    prev_place = None
    for ei, loc, mid, yr in trips:
        if loc != prev_place:
            route.append(loc)
            prev_place = loc
    if len(route) >= 2:
        person_routes[person] = route

print(f"Persons with routes (>=2 places): {len(person_routes)}")

# Count transitions
transition_counts = Counter()
for person, route in person_routes.items():
    for i in range(len(route) - 1):
        transition_counts[(route[i], route[i + 1])] += 1

print("\nTop 20 place-to-place transitions:")
for (p1, p2), cnt in transition_counts.most_common(20):
    print(f"  {nam(p1):30s} -> {nam(p2):30s}  count={cnt}")

# Show key persons' routes
print("\nKey persons' travel routes:")
key_persons = ["P022", "P107", "P045", "P028", "P199"]  # 円仁, 最澄, 圓珍, 空海, etc.
for pid in key_persons:
    if pid in person_routes:
        route_names = [nam(p) for p in person_routes[pid]]
        print(
            f"  {nam(pid)}: {' → '.join(route_names[:15])}{' ...' if len(route_names) > 15 else ''}"
        )
    elif pid in person_places:
        places = [nam(p) for p in person_places[pid]]
        print(f"  {nam(pid)} (from TRAVEL edges): {', '.join(places[:15])}")

# ============================================================
# 4. TEMPORAL DESTINATION SHIFTS
# ============================================================
print("\n" + "=" * 60)
print("4. TEMPORAL DESTINATION SHIFTS")
print("=" * 60)

# Use event-linked place visits with years
early = Counter()  # <= 750
late = Counter()  # > 750
cutoff = 750

for person, trips in person_event_places.items():
    for ei, loc, mid, yr in trips:
        if yr is None:
            continue
        if yr <= cutoff:
            early[loc] += 1
        else:
            late[loc] += 1

print(f"Early (<= {cutoff}): {sum(early.values())} visits, {len(early)} unique places")
print(f"Late (> {cutoff}): {sum(late.values())} visits, {len(late)} unique places")

print("\nTop places EARLY period:")
for pid, cnt in early.most_common(15):
    print(f"  {nam(pid):30s}  visits={cnt}")

print("\nTop places LATE period:")
for pid, cnt in late.most_common(15):
    print(f"  {nam(pid):30s}  visits={cnt}")

# Unique to each period
only_early = set(early.keys()) - set(late.keys())
only_late = set(late.keys()) - set(early.keys())
common = set(early.keys()) & set(late.keys())

print(f"\nUnique to EARLY ({len(only_early)}):")
for pid in sorted(only_early):
    print(f"  {nam(pid)} ({early[pid]})")

print(f"\nUnique to LATE ({len(only_late)}):")
for pid in sorted(only_late):
    print(f"  {nam(pid)} ({late[pid]})")

print(f"\nCommon to both periods ({len(common)}):")
for pid in sorted(common, key=lambda p: early[p] + late[p], reverse=True)[:15]:
    print(f"  {nam(pid)} (early={early[pid]}, late={late[pid]})")

# Also check by time_period in events
print("\n--- Direct location analysis from enhanced_events ---")
loc_by_year_bucket = defaultdict(Counter)
for ev in events_list:
    loc = ev.get("location", "")
    mid = ev.get("mission_id", "")
    yr = mission_years.get(mid, None)
    if not loc or yr is None:
        continue
    bucket = "pre-750" if yr <= 750 else "post-750"
    loc_by_year_bucket[bucket][loc] += 1

for bucket, locs in loc_by_year_bucket.items():
    print(f"\n{bucket} (total: {sum(locs.values())}):")
    for loc, cnt in locs.most_common(10):
        print(f"  {loc}: {cnt}")
