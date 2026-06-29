#!/usr/bin/env python3
"""Step 7: Merge, deduplicate, and disambiguate extracted events.

Reads all batch output files from step 6, merges events, deduplicates by
(source_text fingerprint + event_type + subject + object), and standardizes
entity names using the alias map.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

from embedding_pipeline.config import EXTRACTED_EVENTS_DIR, MERGED_EVENTS_JSON


# ── Import alias map from old ETL config ────────────────────────────────────
def load_alias_map() -> dict[str, str]:
    """Load ALIAS_MAP from old ETL config, or return a built-in copy."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from etl.config import ALIAS_MAP

        return dict(ALIAS_MAP)
    except (ImportError, AttributeError):
        pass

    # Built-in fallback
    return {
        "空海": "空海",
        "弘法大师": "空海",
        "弘法大師": "空海",
        "遍照金刚": "空海",
        "遍照金剛": "空海",
        "最澄": "最澄",
        "伝教大师": "最澄",
        "傳教大師": "最澄",
        "円仁": "円仁",
        "圆仁": "円仁",
        "圓仁": "円仁",
        "慈觉大师": "円仁",
        "慈覺大師": "円仁",
        "阿倍仲麻吕": "阿倍仲麻呂",
        "阿倍仲麻呂": "阿倍仲麻呂",
        "晁衡": "阿倍仲麻呂",
        "朝衡": "阿倍仲麻呂",
        "吉备真备": "吉備真備",
        "吉備真備": "吉備真備",
        "鉴真": "鑑真",
        "鑑真": "鑑真",
        "鑒真": "鑑真",
        "藤原清河": "藤原清河",
        "藤原葛野麻呂": "藤原葛野麻呂",
        "藤原常嗣": "藤原常嗣",
        "菅原道真": "菅原道真",
        "小野妹子": "小野妹子",
        "犬上御田鍬": "犬上御田鍬",
        "藥師惠日": "薬師恵日",
        "药師惠日": "薬師恵日",
        "高向玄理": "高向玄理",
        "道昭": "道昭",
        "円行": "円行",
        "圆行": "円行",
        "円載": "円載",
        "圆载": "円載",
        "常暁": "常暁",
        "常晓": "常暁",
        "恵運": "恵運",
        "惠运": "恵運",
        "恵萼": "恵萼",
        "惠萼": "恵萼",
        "真如": "真如",
        "円珍": "円珍",
        "圆珍": "円珍",
        "義真": "義真",
        "义真": "義真",
        # Fragment normalization (name + suffix/title → canonical)
        "空海法師": "空海",
        "空海上表": "空海",
        "釋空海": "空海",
        "釋空海從遣唐使": "空海",
        "最澄法師": "最澄",
        "最澄賜": "最澄",
        "最澄闍梨": "最澄",
        "傳教大師": "最澄",
        "伝教大师": "最澄",
        "円仁法師": "円仁",
        "圓仁法師": "円仁",
        "釋圓仁": "円仁",
        "釋円仁": "円仁",
        "釋道昭": "道昭",
        "道昭法師": "道昭",
        "鑑真大和上": "鑑真",
        "鑑真和尚": "鑑真",
        "大和上鑑真": "鑑真",
        "吉備朝臣眞備": "吉備真備",
        "吉備朝臣真備": "吉備真備",
        "阿倍朝臣仲麻呂": "阿倍仲麻呂",
        "藤原朝臣常嗣": "藤原常嗣",
        "藤原朝臣葛野麻呂": "藤原葛野麻呂",
        "鑑真之父": "鑑真之父",  # keep as distinct entity
    }


def text_fingerprint(text: str, n_chars: int = 30) -> str:
    """Create a fingerprint from the first N chars of source text."""
    cleaned = text.strip().replace("\n", "").replace(" ", "")
    return cleaned[:n_chars]


def normalize_name(name: str, alias_map: dict[str, str]) -> str:
    """Normalize an entity name using the alias map."""
    name = name.strip()
    if not name:
        return name
    if name in alias_map:
        return alias_map[name]
    # Check if name contains any alias as substring
    for alias, canonical in sorted(alias_map.items(), key=lambda x: -len(x[0])):
        if alias != canonical and alias in name:
            return canonical
    # Also check: does name contain a canonical form + suffix?
    # e.g. "最澄法師" contains "最澄", "空海上表" contains "空海"
    for canonical in sorted(alias_map.values(), key=lambda x: -len(x)):
        if canonical in name and canonical != name:
            return canonical
    return name


def load_batch_outputs() -> list[dict]:
    """Load all batch and merged group output JSON files."""
    events = []

    # Load batch output files (from delegate_task)
    batch_files = sorted(EXTRACTED_EVENTS_DIR.glob("batch_*_output.json"))
    print(f"  Found {len(batch_files)} batch output files")
    for bf in batch_files:
        try:
            with open(bf, "r", encoding="utf-8") as f:
                batch_events = json.load(f)
            # Handle both dict format {"events": [...]} and list format [...]
            if isinstance(batch_events, dict) and "events" in batch_events:
                batch_events = batch_events["events"]
            if isinstance(batch_events, list):
                events.extend(batch_events)
                if len(batch_events) > 0:
                    print(f"    {bf.name}: {len(batch_events)} events")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"    {bf.name}: ERROR — {e}")

    # Load merged group output files (from cronjob)
    merged_dir = EXTRACTED_EVENTS_DIR / "merged"
    if merged_dir.exists():
        group_files = sorted(merged_dir.glob("group_*_output.json"))
        print(f"  Found {len(group_files)} merged group output files")
        for gf in group_files:
            try:
                with open(gf, "r", encoding="utf-8") as f:
                    group_events = json.load(f)
                if isinstance(group_events, list):
                    events.extend(group_events)
                    if len(group_events) > 0:
                        print(f"    {gf.name}: {len(group_events)} events")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"    {gf.name}: ERROR — {e}")

    if not events:
        print("  WARNING: No events found in any output files!")
    return events


def deduplicate_events(events: list[dict]) -> list[dict]:
    """Deduplicate events by fingerprint + event_type + subject + object."""
    seen = set()
    unique = []

    for event in events:
        fp = text_fingerprint(event.get("source_text", ""))
        key = (
            fp,
            event.get("event_type", ""),
            event.get("subject", "").strip(),
            event.get("object", "").strip(),
        )
        if key not in seen:
            seen.add(key)
            unique.append(event)

    print(f"  Dedup: {len(events)} → {len(unique)} events")
    return unique


def main():
    print("=" * 60)
    print("Step 7: Merge, Dedup & Disambiguate")
    print("=" * 60)

    # Load alias map
    alias_map = load_alias_map()
    print(f"\n  Alias map: {len(alias_map)} entries")

    # Load all events
    print("\n  Loading batch outputs...")
    events = load_batch_outputs()
    print(f"  Total raw events: {len(events)}")

    if len(events) == 0:
        print("\n  WARNING: No events to process. Skipping.")
        MERGED_EVENTS_JSON.write_text("[]", encoding="utf-8")
        return

    # Normalize entity names
    print("\n  Normalizing entity names...")
    # Common LLM artifact suffixes to strip from entity names
    ARTIFACT_SUFFIXES = [
        "薨",
        "卒",
        "死",
        "亡",
        "等",
        "曰",
        "云",
        "言",
        "歸",
        "還",
        "入",
        "至",
    ]
    for event in events:
        event["subject"] = normalize_name(event.get("subject", ""), alias_map)
        event["object"] = normalize_name(event.get("object", ""), alias_map)
        # Clean entity names: strip single-char artifact suffixes
        for field in ("subject", "object"):
            val = event.get(field, "")
            # Strip trailing artifact chars if the name ends with them
            # but only if the remaining name is still meaningful (>2 chars)
            for suffix in ARTIFACT_SUFFIXES:
                if val.endswith(suffix) and len(val) - len(suffix) >= 2:
                    val = val[: -len(suffix)]
            event[field] = val.strip()

    # Fill empty subjects from source_text context
    # Many events lose their subject due to 200-char chunking boundaries
    KNOWN_SUBJECTS = ["空海", "最澄", "円仁", "圓仁", "道昭", "鑑真", "宗叡",
                      "常暁", "円珍", "圓珍", "吉備真備", "阿倍仲麻呂",
                      "藤原常嗣", "藤原清河", "藤原葛野麻呂", "恵運",
                      "圓載", "円載", "行賀", "玄昉", "真如"]
    for event in events:
        if not event.get("subject", "").strip():
            src = event.get("source_text", "")
            for name in sorted(KNOWN_SUBJECTS, key=len, reverse=True):
                if name in src:
                    event["subject"] = name
                    break

    # Deduplicate
    print("\n  Deduplicating...")
    events = deduplicate_events(events)

    # Statistics
    event_types = defaultdict(int)
    for e in events:
        event_types[e.get("event_type", "UNKNOWN")] += 1

    print("\n  Events by type:")
    for etype, count in sorted(event_types.items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count}")

    # Save
    print(f"\n  Saving {len(events)} events to: {MERGED_EVENTS_JSON}")
    with open(MERGED_EVENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print("Step 7 complete.")
    return events


if __name__ == "__main__":
    main()
