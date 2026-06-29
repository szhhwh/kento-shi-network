#!/usr/bin/env python3
"""Enhance events with kentoshi mission batch labels and temporal metadata.

Matches events to 21 missions (M01-M20+M08b) using:
1. Japanese era names in source_text (e.g., 天平勝寳 → M12)
2. Mission personnel names (ambassador, vice-ambassador, key personnel)
3. Year proximity to mission departure/return dates
"""

import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
MISSIONS_FILE = ROOT / "data" / "kentoshi_missions.json"
EVENTS_FILE = ROOT / "output" / "merged_events.json"

# Load missions
with open(MISSIONS_FILE) as f:
    missions = json.load(f)

# Build era → mission index (Japanese and Chinese eras)
era_to_missions: dict[str, list[str]] = defaultdict(list)
person_to_missions: dict[str, set[str]] = defaultdict(set)
mission_info: dict[str, dict] = {}

for m in missions:
    mid = m["id"]
    mission_info[mid] = m
    jp_era = m.get("japanese_era", "")
    cn_era = m.get("chinese_era", "")
    for era_str in [jp_era, cn_era]:
        for part in re.split(r"[→・、\s]+", era_str):
            part = part.strip().rstrip("0123456789")
            if part and len(part) >= 2:
                era_to_missions[part].append(mid)

    for name in [m["ambassador"], m.get("vice_ambassador", "")]:
        name = name.strip()
        if name:
            person_to_missions[name].add(mid)
    for name in m.get("other_key_personnel", []):
        name = name.strip()
        if name:
            person_to_missions[name].add(mid)

YEAR_PATTERN = re.compile(
    r"(舒明|白雉|齊明|天智|天武|持統|文武|大寶|大寳|慶雲|和銅|靈龜|靈亀|養老|"
    r"神龜|天平|天平感寶|天平勝寶|天平勝寳|天平寶字|天平神護|神護景雲|"
    r"寶龜|寶亀|天應|延暦|延曆|大同|弘仁|天長|承和|嘉祥|仁壽|齊衡|天安|"
    r"貞觀|元慶|仁和|寛平|昌泰|延喜)"
    r"|(貞觀|永徽|顯慶|龍朔|麟德|乾封|總章|咸亨|上元|儀鳳|調露|永隆|"
    r"開耀|永淳|弘道|嗣聖|文明|光宅|垂拱|永昌|載初|天授|如意|長壽|"
    r"延載|證聖|天冊萬歲|萬歲登封|萬歲通天|神功|聖曆|久視|大足|長安|"
    r"神龍|景龍|景雲|太極|延和|先天|天寶|至德|乾元|上元|寶應|"
    r"廣德|永泰|大曆|建中|興元|貞元|永貞|元和|長慶|寶曆|大和|開成|會昌|"
    r"大中|咸通|乾符|廣明|中和|光啟|文德|龍紀|大順|景福|乾寧|光化|"
    r"天復|天祐)"
)

for m in missions:
    mid = m["id"]
    year_d = m.get("year_depart", 0)
    year_r = (
        m.get("year_return", 0) if isinstance(m.get("year_return"), int) else year_d + 2
    )

YEAR_NUM_PATTERN = re.compile(r"(?:年|歳)[^\d]{0,5}(\d{1,4})(?:年|載|歳)")

ERA_PLACE_AMBIGUOUS = {"開元", "長安", "貞觀", "大中", "永徽", "元和", "乾元", "天寶"}
CITY_NAMES = {"長安", "洛陽"}


def match_event_to_missions(event: dict) -> list[str]:
    """Match an event to kentoshi missions. Returns list of mission IDs.
    
    Uses two-track matching: era-based matches (reliable) and person/keyword
    matches (less reliable). When era matches exist, they take priority.
    """
    era_matches = set()
    other_matches = set()
    src = event.get("source_text", "")
    subj = event.get("subject", "").strip()
    obj = event.get("object", "").strip()
    time_str = event.get("time_period", "").strip()

    # Rule 1: Year/era in source_text or time_period
    search_text = src + " " + time_str
    for match in YEAR_PATTERN.finditer(search_text):
        era = match.group(0)
        if era in CITY_NAMES:
            continue
        if era in ERA_PLACE_AMBIGUOUS:
            context = search_text[match.start() : match.end() + 3]
            if re.search(rf"{era}[寺院城門宮殿閣塔]", context):
                continue
        if era in era_to_missions:
            # Skip ambiguous era names (中日同名年号) — defer to Rule 2
            if era in ERA_PLACE_AMBIGUOUS:
                continue
            for mid in era_to_missions[era]:
                era_matches.add(mid)

    # Rule 2: Era + year → actual year → mission proximity
    ERA_START = {
        "大化": 645, "朱鳥": 686, "持統": 687, "文武": 697,
        "舒明": 629, "白雉": 650, "齊明": 655, "天智": 662,
        "大寶": 701, "大寳": 701, "慶雲": 704, "和銅": 708,
        "靈龜": 715, "靈亀": 715, "養老": 717, "神龜": 724,
        "天平": 729, "天平勝寶": 749, "天平勝寳": 749,
        "天平寶字": 757, "寶龜": 770, "寶亀": 770,
        "延暦": 782, "延曆": 782, "弘仁": 810, "天長": 824,
        "承和": 834, "嘉祥": 848, "仁壽": 851, "齊衡": 854,
        "天安": 857,
        "貞觀_jp": 859, "元慶": 877, "寛平": 889,
        "貞觀_cn": 627,
        "永徽": 650, "顯慶": 656, "天寶": 742, "貞元": 785,
        "元和": 806, "長慶": 821, "開成": 836, "會昌": 841,
        "大中": 847,
    }
    # Japanese-specific context markers for disambiguating 貞觀 etc.
    JP_MARKERS = [
        "圓仁", "圓珍", "清和", "陽成", "光孝", "宇多",
        "延曆寺", "三代實錄", "日本紀略", "元慶", "仁和",
        "寛平", "齊衡", "天安", "仁壽", "比叡山", "日本",
    ]
    
    for m in re.finditer(r"([\u4e00-\u9fff]{2,4})(\d+)年", search_text):
        era_name = m.group(1)
        era_year = int(m.group(2))
        if era_name in ERA_PLACE_AMBIGUOUS:
            context = search_text[m.start() : m.end() + 3]
            if re.search(rf"{era_name}[寺院城門宮殿閣]", context):
                continue
        era_start = ERA_START.get(era_name)
        if era_start is None:
            jp_key = era_name + "_jp"
            cn_key = era_name + "_cn"
            if jp_key in ERA_START and cn_key in ERA_START:
                if any(mk in search_text for mk in JP_MARKERS):
                    era_start = ERA_START[jp_key]
                else:
                    era_start = ERA_START[cn_key]
            elif jp_key in ERA_START:
                era_start = ERA_START[jp_key]
            elif cn_key in ERA_START:
                era_start = ERA_START[cn_key]
        if era_start is not None:
            actual_year = era_start + era_year - 1
            for mission in missions:
                dep = mission["year_depart"]
                ret_raw = mission.get("year_return", dep + 2)
                if isinstance(ret_raw, str) or not ret_raw:
                    ret = dep + 2
                else:
                    ret = int(ret_raw)
                if dep - 5 <= actual_year <= ret + 5:
                    era_matches.add(mission["id"])

    # Rule 3: Personnel name match
    for name in [subj, obj]:
        if name in person_to_missions:
            other_matches.update(person_to_missions[name])
        for pname, mids in person_to_missions.items():
            if len(pname) >= 2 and pname in name:
                other_matches.update(mids)

    # Rule 4: Keyword-based mission matching
    mission_keywords = {
        "M01": ["犬上", "藥師惠日"],
        "M02": ["吉士長丹", "道昭", "白雉"],
        "M04": ["坂合部"],
        "M08": ["粟田真人", "大寶令", "大寳令", "長安中"],
        "M09": ["阿倍仲麻呂", "吉備真備", "吉備朝臣", "下道朝臣",
                "靈龜", "養老", "玄昉", "多治比縣守"],
        "M10": ["多治比廣成"],
        "M11": ["石上乙麻呂"],
        "M12": ["藤原清河", "藤原河清", "鑑真", "天平勝寶", "天平勝寳",
                 "朝衡", "晁衡"],
        "M13": ["高元度", "迎入唐大使"],
        "M16": ["佐伯今毛人", "小野石根", "小野朝臣石根", "寶龜", "寶亀"],
        "M17": ["布勢清直"],
        "M18": ["藤原葛野", "藤原葛野麻呂", "空海", "最澄", "延暦",
                 "菅原清公", "菅原朝臣清公"],
        "M19": ["藤原常嗣", "円仁", "圓仁", "承和", "開成"],
        "M20": ["菅原道真", "寛平"],
    }
    for mid, kws in mission_keywords.items():
        for kw in kws:
            if kw in search_text:
                other_matches.add(mid)
                break

    # Save to module scope for context backtracking
    global _ERA_START, _MISSION_KEYWORDS
    _ERA_START = ERA_START
    _MISSION_KEYWORDS = mission_keywords

    # Prefer era-based matches when available
    final = era_matches if era_matches else other_matches
    return sorted(final)


# ── Chinese numeral → integer ──────────────────────────────────────────────
CN_NUM = {
    "元": 1, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
    "廿": 20, "卅": 30,
}
ERA_YEAR_RE = re.compile(r"([\u4e00-\u9fff]{2,4})([\d一二三四五六七八九十廿卅]+)年")

_ERA_START = {}
_MISSION_KEYWORDS = {}

# ── Context backtracking ────────────────────────────────────────────────────
_chunks_cache = None
_sections_cache = None


def _load_chunks():
    global _chunks_cache, _sections_cache
    if _chunks_cache is not None:
        return
    chunks_file = ROOT / "output" / "chunks.jsonl"
    _chunks_cache = {}
    _sections_cache = defaultdict(list)
    with open(chunks_file) as f:
        for line in f:
            c = json.loads(line)
            _chunks_cache[c["chunk_id"]] = c
            _sections_cache[c.get("section", "")].append(c["chunk_id"])


def match_by_context(event: dict) -> set[str]:
    """Second-pass matching using section-level context (±5 chunks)."""
    _load_chunks()
    matches = set()
    cid = event.get("chunk_id")
    src = event.get("source_text", "").strip()

    if cid is None:
        src_prefix = src[:30].replace("\n", "").replace(" ", "")
        for test_cid, c in _chunks_cache.items():
            if src_prefix and src_prefix[:15] in c["text"].replace("\n", ""):
                cid = test_cid
                break

    if cid is None or cid not in _chunks_cache:
        return matches

    sec = _chunks_cache[cid].get("section", "")
    sec_chunks = _sections_cache.get(sec, [])
    try:
        idx = sec_chunks.index(cid)
    except ValueError:
        return matches

    window = 8 if len(sec_chunks) > 20 else 5
    ctx = ""
    for off in range(-window, window + 1):
        ni = idx + off
        if 0 <= ni < len(sec_chunks):
            ctx += _chunks_cache[sec_chunks[ni]]["text"] + "\n"

    for m in ERA_YEAR_RE.finditer(ctx):
        era = m.group(1)
        yr_str = m.group(2)
        if era in _ERA_START:
            yr = CN_NUM.get(yr_str, None)
            if yr is None:
                try:
                    yr = int(yr_str)
                except (ValueError, TypeError):
                    continue
            actual_year = _ERA_START[era] + yr - 1
            for mission in missions:
                dep = mission["year_depart"]
                ret_raw = mission.get("year_return", dep + 2)
                if isinstance(ret_raw, str) or not ret_raw:
                    ret = dep + 2
                else:
                    ret = int(ret_raw)
                if dep - 5 <= actual_year <= ret + 5:
                    matches.add(mission["id"])

    for mid, kws in _MISSION_KEYWORDS.items():
        for kw in kws:
            if kw in ctx and mid not in matches:
                if kw not in src:
                    matches.add(mid)
                break

    return matches


# ── Main processing ─────────────────────────────────────────────────────────
with open(EVENTS_FILE) as f:
    events = json.load(f)

enhanced = []
matched_count = 0
multi_match = 0

for event in events:
    mid_list = match_event_to_missions(event)
    if mid_list:
        matched_count += 1
        if len(mid_list) > 1:
            multi_match += 1
        primary = mission_info[mid_list[0]]
        event["mission_id"] = mid_list[0]
        event["mission_departure_year"] = primary["year_depart"]
        event["mission_return_year"] = primary.get("year_return", "")
    else:
        ctx_matches = match_by_context(event)
        if ctx_matches:
            matched_count += 1
            if len(ctx_matches) > 1:
                multi_match += 1
            primary = mission_info[list(ctx_matches)[0]]
            event["mission_id"] = list(ctx_matches)[0]
            event["mission_departure_year"] = primary["year_depart"]
            event["mission_return_year"] = primary.get("year_return", "")
        else:
            event["mission_id"] = ""
            event["mission_departure_year"] = ""
            event["mission_return_year"] = ""

    enhanced.append(event)

ENHANCED_FILE = ROOT / "output" / "enhanced_events.json"
with open(ENHANCED_FILE, "w", encoding="utf-8") as f:
    json.dump(enhanced, f, ensure_ascii=False, indent=2)

print(f"Total events: {len(enhanced)}")
print(f"Matched to missions: {matched_count} ({matched_count / len(enhanced) * 100:.1f}%)")
print(f"Multi-match: {multi_match}")
print(f"Saved: {ENHANCED_FILE}")

mission_event_counts = defaultdict(int)
for e in enhanced:
    if e.get("mission_id"):
        mission_event_counts[e["mission_id"]] += 1
print("\nEvents per mission:")
for mid in sorted(mission_event_counts.keys(), key=lambda x: int(x[1:].replace("b", ""))):
    print(f"  {mid}: {mission_event_counts[mid]} events")
