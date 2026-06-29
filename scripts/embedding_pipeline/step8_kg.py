#!/usr/bin/env python3
"""Step 8: Build knowledge graph — generate nodes.csv and edges.csv from events.

Collects entities from event subjects, objects, and locations. Infers entity
types (Person/Event/Place/Culture). Generates edges based on event types.
Output format matches the old ETL pipeline for downstream compatibility.
"""

import csv
import json
import sys
from collections import defaultdict

from embedding_pipeline.config import (
    MERGED_EVENTS_JSON,
    PIPELINE_NODES_CSV,
    PIPELINE_EDGES_CSV,
)


# ── Entity type inference ───────────────────────────────────────────────────

PLACE_INDICATORS = [
    "寺",
    "山",
    "州",
    "京",
    "府",
    "縣",
    "县",
    "島",
    "岛",
    "港",
    "津",
    "關",
    "关",
    "驛",
    "驿",
    "門",
    "门",
    "院",
    "堂",
    "閣",
    "阁",
    "臺",
    "台",
    "壇",
    "坛",
    "陵",
    "墓",
    "宮",
    "宫",
    "殿",
    "城",
    "鎮",
    "镇",
    "村",
    "里",
    "坊",
    "路",
    "道",
    "海",
    "江",
    "河",
    "湖",
    "泊",
    "岸",
    "浦",
    "渡",
    "津",
]
CULTURE_INDICATORS = [
    "經",
    "经",
    "律",
    "論",
    "论",
    "疏",
    "記",
    "记",
    "曆",
    "历",
    "令",
    "式",
    "格",
    "典",
    "禮",
    "礼",
    "宗",
    "教",
    "法",
    "戒",
    "禪",
    "禅",
    "像",
    "佛",
    "曼荼羅",
    "曼荼罗",
    "曼陀羅",
    "書",
    "书",
    "集",
    "卷",
    "詩",
    "诗",
    "文集",
    "樂",
    "乐",
    "舞",
    "畫",
    "画",
    "目錄",
    "目录",
    "圖",
    "图",
    "藥",
    "药",
    "醫",
    "医",
    "針",
    "针",
    "物",
    "器",
    "具",
    "寶",
    "宝",
    "劍",
    "剑",
    "鏡",
    "镜",
    "玉",
    "香",
    "藥",
    "药",
]

# Group/collective indicators — entities that are groups, not individuals
GROUP_INDICATORS = [
    "船",
    "舶",
    "團",
    "团",
    "使團",
    "使团",
    "隊",
    "队",
    "衆",
    "众",
    "等",
    "船團",
    "船团",
    "舳",
    "艦",
    "舰",
]

# Known geopolitical entities — these are Places even without indicators
# Use exact or boundary matching to avoid compound-word false positives
# (e.g., "唐僧" should NOT match "唐" as Place)
GEO_ENTITIES = [
    "唐國",
    "唐朝",
    "大唐",
    "巨唐",
    "隋唐",
    "日本",
    "日本國",
    "倭國",
    "百濟",
    "百济",
    "新羅",
    "新罗",
    "高麗",
    "高丽",
    "渤海",
    "耽羅",
    "長安",
    "長安城",
    "洛陽",
    "洛邑",
]

# Known person names that contain place-indicator characters
# (e.g., 空海 contains 海, 道昭 contains 道)
KNOWN_PERSONS = [
    "空海",
    "最澄",
    "圓仁",
    "円仁",
    "圓珍",
    "円珍",
    "道昭",
    "道慈",
    "道旻",
    "道璿",
    "道嶋",
    "鑑真",
    "鑒真",
    "義真",
    "義眞",
    "圓行",
    "円行",
    "圓載",
    "円載",
    "常曉",
    "常暁",
    "惠運",
    "恵運",
    "惠萼",
    "恵萼",
    "眞如",
    "真如",
    "玄昉",
    "玄奘",
    "行賀",
    "善議",
    "靈仙",
    "宗叡",
    "永忠",
    "實敏",
    "常騰",
    "惠日",
    "恵日",
    "藤原清河",
    "藤原常嗣",
    "藤原葛野麻呂",
    "吉備真備",
    "阿倍仲麻呂",
    "晁衡",
    "朝衡",
    "高向玄理",
    "小野妹子",
    "犬上御田鍬",
    "粟田真人",
    "多治比廣成",
    "菅原道真",
    "大伴古麻呂",
    "坂合部石布",
    "津守吉祥",
    "藤原葛野麻呂",
    "石川道益",
    "小野篁",
    "高元度",
    "平郡廣成",
    "高丘親王",
    "惠齋",
    "惠光",
    "智通",
    "智達",
    "智鳳",
    "智鸞",
    "義德",
    "淨願",
    "清安",
    # Added: names with place-indicator chars (河, 山, 海, etc.)
    "河邊臣麻呂",
    "河邊麻呂",
    "河邊",
    "小山長丹",
    "海上三狩",
    "山代宿祢氏益",
    "山代",
    "藤原葛野麻呂",
    "石川道益",
    "下道朝臣眞備",
    "下道真備",
    "羽栗翼",
    "羽栗臣翼",
    "春苑宿祢玉成",
    "伊与部家守",
    "長岑宿禰高名",
    "長岑高名",
    "菅原朝臣清公",
    "菅原清公",
    "小野朝臣石根",
    "大神朝臣末足",
    "坂合部連石布",
    "津守連吉祥",
    "布勢朝臣人主",
    "田口朝臣養年富",
    "甘南備眞人信影",
    "紀朝臣馬主",
    "紀朝臣三寅",
    "大宅臣福主",
    "船連夫子",
    "大津造廣人",
]


def infer_entity_type(name: str) -> str:
    """Infer entity type from name: Person, Place, Culture, or Event."""
    name = name.strip()
    if not name:
        return "Person"

    # Check geo-political entities — these are Places
    for geo in GEO_ENTITIES:
        if name == geo or name.startswith(geo) or name.endswith(geo):
            return "Place"

    # Check known person names (monks, scholars) — override indicator matching
    for person_name in KNOWN_PERSONS:
        if person_name in name:
            return "Person"

    # Check place indicators
    for indicator in PLACE_INDICATORS:
        if indicator in name:
            return "Place"

    # Check group indicators
    for indicator in GROUP_INDICATORS:
        if indicator in name:
            return "Group"

    # Check culture indicators
    for indicator in CULTURE_INDICATORS:
        if indicator in name:
            return "Culture"

    # Not Person/Place/Culture/Group → default to Person
    return "Person"


# ── Event type to relation mapping ──────────────────────────────────────────

EVENT_TO_RELATION = {
    "派遣遣唐使": "PARTICIPATE",
    "到達唐朝": "TRAVEL",
    "返回日本": "TRAVEL",
    "師從學習": "LEARN",
    "受法受戒": "LEARN",
    "帶回文化物品": "INTRODUCE",
    "創立宗派": "INTRODUCE",
    "傳入制度": "INTRODUCE",
    "朝貢獻物": "PARTICIPATE",
    "接受賞賜": "PARTICIPATE",
    "前往地點": "TRAVEL",
    "賦詩唱和": "PARTICIPATE",
    "海上遭難": "TRAVEL",
    "授予官職": "PARTICIPATE",
}


def main():
    print("=" * 60)
    print("Step 8: Build Knowledge Graph")
    print("=" * 60)

    # Load merged events
    print(f"\n  Loading events from: {MERGED_EVENTS_JSON}")
    with open(MERGED_EVENTS_JSON, "r", encoding="utf-8") as f:
        events = json.load(f)
    print(f"  Total events: {len(events)}")

    if len(events) == 0:
        print("ERROR: No events to process!")
        sys.exit(1)

    # ── Collect entities ────────────────────────────────────────────────────
    print("\n  Collecting entities...")
    entity_map: dict[str, dict] = {}  # canonical_name -> entity info

    def add_entity(name: str, sources: list[str] | None = None):
        """Register an entity, tracking source texts."""
        name = name.strip()
        if not name or len(name) < 1:
            return
        if sources is None:
            sources = []
        if name not in entity_map:
            entity_map[name] = {
                "name": name,
                "type": infer_entity_type(name),
                "sources": [],
                "source_count": 0,
            }
        if sources:
            for src in sources:
                if src and src not in entity_map[name]["sources"]:
                    entity_map[name]["sources"].append(src)
            entity_map[name]["source_count"] = len(entity_map[name]["sources"])

    for event in events:
        source_text = event.get("source_text", "")
        add_entity(event.get("subject", ""), [source_text])
        add_entity(event.get("object", ""), [source_text])
        add_entity(event.get("location", ""), [source_text])

    # ── Create Event entities from extracted events ──────────────────────────
    # Each extracted event becomes an Event node for the knowledge graph.
    for i, event in enumerate(events):
        etype = event.get("event_type", "未知事件")
        subject = event.get("subject", "").strip()
        action = event.get("action", "").strip()
        source_text = event.get("source_text", "")
        # Create a descriptive name for the event
        event_name = f"{etype}：{subject}{action}" if action else f"{etype}"
        event_name = event_name[:100]  # truncate long names
        # Use a unique key that includes the index to avoid collisions
        event_key = f"__event_{i}__"
        entity_map[event_key] = {
            "name": event_name,
            "type": "Event",
            "sources": [source_text] if source_text else [],
            "source_count": 1,
        }

    print(f"  Unique entities: {len(entity_map)}")

    # ── Filter noise entities (LLM artifacts) ──────────────────────────────
    # Single-character verbs, sentence fragments, and common non-entity terms
    NOISE_ENTITIES = {
        "破",
        "沈",
        "浮",
        "沒",
        "發",
        "至",
        "到",
        "歸",
        "此發送",
        "此",
        "其",
        "之",
        "者",
        "也",
        "焉",
        "本國",
        "歸國",
        "歸朝",
        "巨唐",
        "灌頂",
        "真言",
        "神寳",
        "秘奥",
        "止觀",
        "白鹿皮一、弓三、箭八十",
        "第一船",
        "第二船",
        "第三舶",
    }
    removed = 0
    for noise in NOISE_ENTITIES:
        if noise in entity_map:
            del entity_map[noise]
            removed += 1
    # Also remove single-char entities that are not in KNOWN_PERSONS
    for key in list(entity_map.keys()):
        if len(key) == 1 and key not in KNOWN_PERSONS:
            del entity_map[key]
            removed += 1
    # Remove garbled text fragments misclassified as Person
    # (sentence-level text that leaked into subject/object fields).
    # Detect by: contains punctuation, quotes, or consecutive non-CJK chars.
    TEXT_FRAGMENT_MARKERS = {"。", "▼", "』", "：", "」", "（", "）",
                              "是日", "此", "乃", "盖", "夫", "窃以",
                              "之由", "有賊", "大使上奏", "以上奏"}
    for key in list(entity_map.keys()):
        if entity_map[key]["type"] != "Person":
            continue
        # Flag 1: contains text-fragment punctuation or "/" compound marker
        if any(m in key for m in TEXT_FRAGMENT_MARKERS) or "/" in key:
            del entity_map[key]
            removed += 1
            continue
        # Flag 2: verb phrases (contain 歸/歸國/還/漂/遊 as standalone, not in names)
        VERB_PATTERNS = ["歸國", "還至", "漂著", "遊唐"]
        if any(vp in key for vp in VERB_PATTERNS):
            del entity_map[key]
            removed += 1
            continue
        # Flag 3: very long name (>30 chars) not in KNOWN_PERSONS
        if len(key) > 30 and key not in KNOWN_PERSONS:
            del entity_map[key]
            removed += 1
    if removed:
        print(f"  Filtered {removed} noise entities")

    # Assign IDs
    type_counters = defaultdict(int)
    entity_ids: dict[str, str] = {}
    type_prefix = {
        "Person": "P",
        "Event": "E",
        "Place": "L",
        "Culture": "C",
        "Group": "G",
    }

    for name, info in sorted(entity_map.items()):
        etype = info["type"]
        type_counters[etype] += 1
        entity_ids[name] = f"{type_prefix[etype]}{type_counters[etype]:03d}"

    # ── Generate nodes ──────────────────────────────────────────────────────
    print("\n  Generating nodes...")
    nodes = []
    for key, info in sorted(entity_map.items()):
        # Use descriptive name for Event entities, key for others
        display_name = info["name"] if info["type"] == "Event" else key
        nodes.append(
            {
                "id": entity_ids[key],
                "type": info["type"],
                "name": display_name,
                "source_count": info["source_count"],
                "sources": info["sources"][:5],  # keep first 5 sources
            }
        )

    # ── Generate edges ──────────────────────────────────────────────────────
    print("  Generating edges...")
    edges = []
    edge_counter = 0

    for i, event in enumerate(events):
        etype = event.get("event_type", "")
        relation = EVENT_TO_RELATION.get(etype)
        if not relation:
            continue

        subject = event.get("subject", "").strip()
        obj = event.get("object", "").strip()
        location = event.get("location", "").strip()
        source_text = event.get("source_text", "")

        subj_id = entity_ids.get(subject)
        event_key = f"__event_{i}__"
        event_id = entity_ids.get(event_key)

        # Event node → Subject (if subject exists)
        if subj_id and event_id:
            edge_counter += 1
            edges.append(
                {
                    "id": f"E{edge_counter:03d}",
                    "source": event_id,
                    "target": subj_id,
                    "relation": "HAS_SUBJECT",
                    "source_text": source_text[:200],
                }
            )

        # Event node → Object (if object exists)
        if obj and obj in entity_ids and event_id:
            edge_counter += 1
            edges.append(
                {
                    "id": f"E{edge_counter:03d}",
                    "source": event_id,
                    "target": entity_ids[obj],
                    "relation": "HAS_OBJECT",
                    "source_text": source_text[:200],
                }
            )

        # Event node → Location (if location exists)
        if location and location in entity_ids and event_id:
            edge_counter += 1
            edges.append(
                {
                    "id": f"E{edge_counter:03d}",
                    "source": event_id,
                    "target": entity_ids[location],
                    "relation": "AT_LOCATION",
                    "source_text": source_text[:200],
                }
            )

        # Subject → Object (direct relation)
        if subj_id and obj and obj in entity_ids:
            edge_counter += 1
            edges.append(
                {
                    "id": f"E{edge_counter:03d}",
                    "source": subj_id,
                    "target": entity_ids[obj],
                    "relation": relation,
                    "source_text": source_text[:200],
                }
            )

        # Subject → Location (if location exists and is different from object)
        if subj_id and location and location in entity_ids:
            if not obj or location != obj:
                edge_counter += 1
                edges.append(
                    {
                        "id": f"E{edge_counter:03d}",
                        "source": subj_id,
                        "target": entity_ids[location],
                        "relation": "TRAVEL"
                        if etype in ("到達唐朝", "前往地點", "海上遭難")
                        else relation,
                        "source_text": source_text[:200],
                    }
                )

    # Deduplicate edges by (source, target, relation) and remove self-loops
    seen_edges = set()
    unique_edges = []
    for edge in edges:
        if edge["source"] == edge["target"]:
            continue  # skip self-referencing edges
        key = (edge["source"], edge["target"], edge["relation"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)
    edges = unique_edges

    # Reassign edge IDs
    for i, edge in enumerate(edges):
        edge["id"] = f"E{i + 1:03d}"

    # ── Context-aware reclassification ──────────────────────────────────────
    # Entities that are Place but targets of LEARN/INTRODUCE edges
    # that contain culture indicators → reclassify as Culture
    CULTURE_HINT_CHARS = {
        "宗",
        "教",
        "法",
        "經",
        "书",
        "書",
        "典",
        "制",
        "禮",
        "礼",
        "式",
        "論",
        "论",
        "卷",
        "録",
        "录",
        "戒",
        "禪",
        "禅",
        "學",
        "学",
        "曼荼",
        "儀",
        "籍",
        "釋",
        "释",
        "集",
        "詩",
        "诗",
        "曆",
        "历",
        "樂",
        "乐",
        "物",
        "寶",
        "宝",
        "道",
    }
    reclassified = 0
    for edge in edges:
        if edge["relation"] in ("LEARN", "INTRODUCE", "HAS_OBJECT"):
            target_id = edge["target"]
            for key, info in entity_map.items():
                if entity_ids.get(key) == target_id and info["type"] == "Place":
                    name = info["name"]
                    is_temple = any(c in name for c in ["寺", "院"])
                    has_culture_hint = any(c in name for c in CULTURE_HINT_CHARS)
                    is_geo_political = name in GEO_ENTITIES
                    if has_culture_hint and not is_temple:
                        info["type"] = "Culture"
                        reclassified += 1
                    elif is_geo_political and not is_temple:
                        info["type"] = "Culture"
                        reclassified += 1
                    break
    if reclassified:
        print(f"  Context reclassified: {reclassified} Place→Culture")

    # Rebuild nodes list with updated types
    nodes = []
    for key, info in sorted(entity_map.items()):
        display_name = info["name"] if info["type"] == "Event" else key
        nodes.append(
            {
                "id": entity_ids[key],
                "type": info["type"],
                "name": display_name,
                "source_count": info["source_count"],
                "sources": info["sources"][:5],
            }
        )

    # ── Statistics ──────────────────────────────────────────────────────────
    type_counts = defaultdict(int)
    for n in nodes:
        type_counts[n["type"]] += 1

    print(f"\n  Nodes: {len(nodes)}")
    for etype in ["Person", "Event", "Place", "Culture", "Group"]:
        print(f"    {etype}: {type_counts.get(etype, 0)}")
    print(f"  Edges: {len(edges)}")

    # ── Scale check ─────────────────────────────────────────────────────────
    person_count = type_counts.get("Person", 0)
    event_count = type_counts.get("Event", 0)
    relation_count = len(edges)

    print("\n  Scale check:")
    print(f"    Person  ≥ 100: {person_count} {'✓' if person_count >= 100 else '✗'}")
    print(f"    Event   ≥ 20:  {event_count} {'✓' if event_count >= 20 else '✗'}")
    print(
        f"    Relation ≥ 500: {relation_count} {'✓' if relation_count >= 500 else '✗'}"
    )

    if person_count < 100 or relation_count < 500:
        print("\n  WARNING: Scale targets not met!")
        print(f"    Gap: Person={100 - person_count}, Relation={500 - relation_count}")
        print("    Do NOT proceed to CSV output without addressing gaps.")

    # ── Write CSV ───────────────────────────────────────────────────────────
    print(f"\n  Writing nodes to: {PIPELINE_NODES_CSV}")
    with open(PIPELINE_NODES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "name", "source_count", "sources"])
        for n in nodes:
            writer.writerow(
                [
                    n["id"],
                    n["type"],
                    n["name"],
                    n["source_count"],
                    "; ".join(n["sources"]),
                ]
            )

    print(f"  Writing edges to: {PIPELINE_EDGES_CSV}")
    with open(PIPELINE_EDGES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source", "target", "relation", "source_text"])
        for e in edges:
            writer.writerow(
                [
                    e["id"],
                    e["source"],
                    e["target"],
                    e["relation"],
                    e["source_text"],
                ]
            )

    print("\nStep 8 complete.")
    return nodes, edges


if __name__ == "__main__":
    main()
