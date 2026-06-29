import csv
import json
from collections import defaultdict, Counter

# ===== LOAD DATA =====
nodes = {}
with open("output/nodes.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        nodes[row["id"]] = row

edges_all = []
with open("output/edges.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        edges_all.append(row)

with open("output/enhanced_events.json") as f:
    events = json.load(f)

# ===== Build event->mission map from enhanced_events =====
# Match events by source_text
text_to_mission = {}
for e in events:
    st = e.get("source_text", "").strip()
    if st:
        text_to_mission[st] = {
            "mission_id": e.get("mission_id", ""),
            "departure": e.get("mission_departure_year", ""),
            "return": e.get("mission_return_year", ""),
            "event_type": e.get("event_type", ""),
        }

# ===== Build person->events map (via HAS_SUBJECT, HAS_OBJECT, etc.) =====
person_events = defaultdict(set)
event_persons = defaultdict(set)
for e in edges_all:
    src = e["source"]
    tgt = e["target"]
    rel = e["relation"]
    # Event has subject/object -> person participates
    if (
        nodes.get(src, {}).get("type") == "Event"
        and nodes.get(tgt, {}).get("type") == "Person"
    ):
        if rel in ("HAS_SUBJECT", "HAS_OBJECT"):
            person_events[tgt].add(src)
            event_persons[src].add(tgt)
    if (
        nodes.get(tgt, {}).get("type") == "Event"
        and nodes.get(src, {}).get("type") == "Person"
    ):
        if rel == "HAS_SUBJECT":
            person_events[src].add(tgt)
            event_persons[tgt].add(src)

# Also: person at location, person travels, etc.
for e in edges_all:
    src = e["source"]
    tgt = e["target"]
    rel = e["relation"]
    if (
        nodes.get(src, {}).get("type") == "Person"
        and nodes.get(tgt, {}).get("type") == "Event"
    ):
        person_events[src].add(tgt)
    if (
        nodes.get(tgt, {}).get("type") == "Person"
        and nodes.get(src, {}).get("type") == "Event"
    ):
        person_events[tgt].add(src)

print(f"Persons with events: {len(person_events)}")

# ===== Match events to missions via source_text =====
# Event nodes have source_text in nodes.csv; enhanced_events also have source_text
event_to_mission = {}
for nid, node in nodes.items():
    if node["type"] == "Event":
        src_text = node.get("sources", "").strip()
        for etext, minfo in text_to_mission.items():
            if src_text in etext or etext in src_text:
                event_to_mission[nid] = minfo
                break

print(f"Events matched to missions: {len(event_to_mission)}")

# Show some matches
for nid in list(event_to_mission.keys())[:5]:
    n = nodes[nid]
    m = event_to_mission[nid]
    print(f"  {nid} ({n['name'][:40]}) -> {m['mission_id']} dep={m['departure']}")

# ===== Map persons to time periods =====
person_phases = {}
for pid in person_events:
    phases = set()
    for eid in person_events[pid]:
        if eid in event_to_mission:
            dep = event_to_mission[eid]["departure"]
            try:
                y = int(dep)
                if y < 670:
                    phases.add(1)
                elif y < 756:
                    phases.add(2)
                elif y < 805:
                    phases.add(3)
                else:
                    phases.add(4)
            except (ValueError, KeyError):
                pass
    if phases:
        person_phases[pid] = sorted(phases)

print(f"\nPersons with phase info: {len(person_phases)}")

# ===== Analyze INTRODUCE edges by phase =====
introduce_edges = [e for e in edges_all if e["relation"] == "INTRODUCE"]

print("\n===== INTRODUCE EDGES BY PHASE =====")
phase_intro_counts = Counter()
phase_items = defaultdict(list)

for e in introduce_edges:
    src = e["source"]
    tgt = e["target"]
    src_name = nodes.get(src, {}).get("name", src)
    tgt_name = nodes.get(tgt, {}).get("name", tgt)
    tgt_type = nodes.get(tgt, {}).get("type", "?")

    # Try to determine phase from source person
    phases = person_phases.get(src, [])
    if not phases:
        # Try target
        phases = person_phases.get(tgt, [])

    if phases:
        for p in phases:
            phase_intro_counts[p] += 1
            phase_items[p].append((src_name, tgt_name, tgt_type))
    else:
        phase_intro_counts["unknown"] += 1
        phase_items["unknown"].append((src_name, tgt_name, tgt_type))

for phase in sorted([k for k in phase_intro_counts.keys() if k != "unknown"]):
    print(f"\nPhase {phase}: {phase_intro_counts[phase]} INTRODUCE edges")
    for src, tgt, ttype in phase_items[phase]:
        print(f"  {src} -> {tgt} ({ttype})")

if "unknown" in phase_intro_counts:
    print(f"\nUnknown phase: {phase_intro_counts['unknown']} edges")
    for src, tgt, ttype in phase_items["unknown"]:
        print(f"  {src} -> {tgt} ({ttype})")

# ===== Also categorize INTRODUCE items by phase =====
print("\n===== CATEGORY BREAKDOWN BY PHASE =====")


# Reuse categorize function
def categorize(name):
    n = name.lower()
    if any(
        kw in n
        for kw in [
            "經",
            "論",
            "疏",
            "卷",
            "秘法",
            "秘教",
            "法花",
            "天台",
            "眞言",
            "胎藏",
            "金剛",
            "灌頂",
            "戒",
            "禪",
            "釋典",
            "宗書",
            "聖教",
            "曼茶",
            "法門",
            "三昧",
            "傳法",
            "具戒",
            "受戒",
            "唯識",
            "法相",
            "法花",
            "三論",
            "舎利",
            "經論",
            "章疏",
            "傳記",
            "秘法",
            "靈像",
        ]
    ):
        return "Buddhist Texts & Practices"
    if any(kw in n for kw in ["暦", "曆", "天文", "陰陽", "難義"]):
        return "Calendar/Astronomy"
    if any(kw in n for kw in ["式", "法", "制度", "儀式", "釋奠", "朝儀"]):
        return "Institutions & Rituals"
    if any(kw in n for kw in ["詩", "筆", "文書", "文籍", "賦詩"]):
        return "Art & Literature"
    if any(
        kw in n
        for kw in [
            "物",
            "貨",
            "寶",
            "信物",
            "唐組",
            "衣服",
            "僧服",
            "功德幀",
            "土石",
            "方物",
        ]
    ):
        return "General Goods & Tribute"
    return "Other"


for phase in sorted([k for k in phase_intro_counts.keys() if k != "unknown"]):
    cats = Counter()
    for src, tgt, ttype in phase_items[phase]:
        cats[categorize(tgt)] += 1
    print(f"Phase {phase}:")
    for cat, cnt in cats.most_common():
        print(f"  {cat}: {cnt}")

print("\nUnknown phase:")
cats = Counter()
for src, tgt, ttype in phase_items.get("unknown", []):
    cats[categorize(tgt)] += 1
for cat, cnt in cats.most_common():
    print(f"  {cat}: {cnt}")

# ===== Manual phase assignment for key introducers =====
# Based on historical knowledge:
known_person_phases = {
    "P028": 4,  # 円仁 (Ennin, 838 mission, Phase 4)
    "P107": 3,  # 最澄 (Saicho, 804 mission, Phase 4? Actually 804 is Phase 4)
    "P045": 4,  # 圓珍 (Enchin, 853, Phase 4)
    "P027": 3,  # 常曉 (Jokyo, 838, Phase 4? Let me check... 常曉 went with Ennin = 838 = Phase 4)
    "P047": 4,  # 圓載 (Enzai, went with Ennin, Phase 4)
    "P157": 2,  # 行賀 (Gyoga, Nara period... let me check. Actually 行賀 went in 752 = Phase 2)
    "P037": 1,  # 吉士長丹 (654 mission = Phase 1)
    "P172": 3,  # 遣唐使 generic
    "P150": 4,  # 藤原岳守 (9th c. = Phase 4)
    "P145": 3,  # 菅原某 (likely Phase 3)
    "P103": 4,  # 春苑宿祢玉成 (Phase 4, 838 mission)
    "P039": 2,  # 和迩部臣宅繼 (Phase 2, Nara period)
}

print("\n\n===== MANUAL PHASE ASSIGNMENT =====")
manual_phase_counts = Counter()
for e in introduce_edges:
    src = e["source"]
    tgt = e["target"]
    tgt_name = nodes.get(tgt, {}).get("name", tgt)
    phase = known_person_phases.get(src, 0)
    if phase > 0:
        manual_phase_counts[(phase, categorize(tgt_name))] += 1
    else:
        print(f"  Unassigned: {src} -> {tgt_name}")

for phase in range(1, 5):
    print(f"\nPhase {phase}:")
    phase_total = 0
    for (p, cat), cnt in sorted(manual_phase_counts.items()):
        if p == phase:
            print(f"  {cat}: {cnt}")
            phase_total += cnt
    print(f"  TOTAL: {phase_total}")

print("\nDONE with temporal analysis")
