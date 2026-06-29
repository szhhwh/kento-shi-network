#!/usr/bin/env python3
"""Phase transition detection: natural breakpoints, diplomacy→Buddhism shift, inflection points."""

import json
from collections import defaultdict

# Load data
with open(
    "/home/szhh/kento-shi-network/output/enhanced_events.json"
) as f:
    events = json.load(f)

# ============================================================
# 1. Per-mission statistics
# ============================================================
print("=" * 80)
print("1. PER-MISSION EVENT STATISTICS")
print("=" * 80)

mission_data = {}
for e in events:
    mid = e.get("mission_id", "")
    if not mid:
        continue
    if mid not in mission_data:
        mission_data[mid] = {
            "events": 0,
            "event_types": defaultdict(int),
            "subjects": set(),
            "locations": set(),
            "departure_year": None,
            "return_year": None,
            "diplomacy_count": 0,
            "buddhist_count": 0,
            "travel_count": 0,
            "return_count": 0,
        }
    d = mission_data[mid]
    d["events"] += 1
    et = e.get("event_type", "")
    d["event_types"][et] += 1
    d["subjects"].add(e.get("subject", ""))
    d["locations"].add(e.get("location", ""))

    if e.get("mission_departure_year") and not d["departure_year"]:
        d["departure_year"] = e["mission_departure_year"]
    if e.get("mission_return_year") and not d["return_year"]:
        d["return_year"] = e["mission_return_year"]

    # Categorize: diplomacy vs. Buddhist
    if et in ["派遣遣唐使", "朝貢獻物", "接受賞賜", "授予官職", "返回日本"]:
        d["diplomacy_count"] += 1
    if et in [
        "師從學習",
        "受法受戒",
        "前往地點",
        "創立宗派",
        "傳入制度",
        "帶回文化物品",
    ]:
        d["buddhist_count"] += 1
    if et == "前往地點":
        d["travel_count"] += 1
    if et == "返回日本":
        d["return_count"] += 1

# Sort missions by departure year
sorted_missions = sorted(
    mission_data.items(),
    key=lambda x: (
        x[1].get("departure_year") if x[1].get("departure_year") is not None else 9999,
        x[0],
    ),
)

for mid, data in sorted_missions:
    total = data["events"]
    dip = data["diplomacy_count"]
    bud = data["buddhist_count"]
    dip_pct = dip / total * 100 if total else 0
    bud_pct = bud / total * 100 if total else 0
    ratio = bud / dip if dip > 0 else float("inf")
    print(
        f"\n{mid} | Departure: {data['departure_year']} | Events: {total} | "
        f"Diplomacy: {dip} ({dip_pct:.0f}%) | Buddhist: {bud} ({bud_pct:.0f}%) | "
        f"B/D ratio: {ratio:.2f}"
    )
    # Top event types
    top_types = sorted(data["event_types"].items(), key=lambda x: -x[1])[:5]
    type_str = ", ".join(f"{t}({c})" for t, c in top_types)
    print(f"  Top types: {type_str}")

# ============================================================
# 2. Natural breakpoint detection
# ============================================================
print("\n" + "=" * 80)
print("2. NATURAL BREAKPOINT DETECTION")
print("=" * 80)

# Build mission-order arrays
missions_ordered = []
event_counts = []
dip_counts = []
bud_counts = []
travel_counts = []
for mid, data in sorted_missions:
    missions_ordered.append(mid)
    event_counts.append(data["events"])
    dip_counts.append(data["diplomacy_count"])
    bud_counts.append(data["buddhist_count"])
    travel_counts.append(data["travel_count"])

n = len(missions_ordered)


def cumsum(arr):
    s = 0
    result = []
    for x in arr:
        s += x
        result.append(s)
    return result


def diff(arr):
    return [arr[i] - arr[i - 1] for i in range(1, len(arr))]


# Cumulative metrics
cum_events = cumsum(event_counts)
cum_dip = cumsum(dip_counts)
cum_bud = cumsum(bud_counts)

# Compute gradients (differences between consecutive missions)
event_diffs = diff(event_counts)
dip_diffs = diff(dip_counts)
bud_diffs = diff(bud_counts)

print("\n--- Mission-by-mission event counts ---")
for i, (mid, ec) in enumerate(zip(missions_ordered, event_counts)):
    diff_str = f" (Δ{event_diffs[i - 1]:+d})" if i > 0 else ""
    print(f"  {mid}: {ec} events{diff_str}")

print("\n--- Consecutive-mission differences (gradients) ---")
for i in range(len(event_diffs)):
    print(
        f"  {missions_ordered[i]}→{missions_ordered[i + 1]}: "
        f"Δevents={event_diffs[i]:+d}, Δdip={dip_diffs[i]:+d}, Δbud={bud_diffs[i]:+d}"
    )

# Find natural breakpoints: where event count jumps > 2x relative to previous
print("\n--- Natural breakpoints (event count > 2x previous mission) ---")
for i in range(1, n):
    if event_counts[i - 1] > 0 and event_counts[i] / event_counts[i - 1] >= 2.0:
        print(
            f"  BREAKPOINT: {missions_ordered[i - 1]}→{missions_ordered[i]} "
            f"({event_counts[i - 1]}→{event_counts[i]} events, {event_counts[i] / event_counts[i - 1]:.1f}x)"
        )

# CUSUM-like detection: deviations from mean
mean_events = sum(event_counts) / n if n > 0 else 0
print(f"\n--- Mean event count per mission: {mean_events:.1f} ---")
cusum = cumsum([ec - mean_events for ec in event_counts])
print("CUSUM values (deviation from mean):")
for i, (mid, cs) in enumerate(zip(missions_ordered, cusum)):
    print(f"  {mid}: CUSUM={cs:+.1f}")

# ============================================================
# 3. Diplomacy → Buddhism shift detection
# ============================================================
print("\n" + "=" * 80)
print("3. DIPLOMACY → BUDDHISM SHIFT DETECTION")
print("=" * 80)

# Define a "Buddhism-dominant" mission as one where buddhist_count > diplomacy_count
# Track the exact point where this shift happens
print("\n--- Per-mission B/D balance ---")
shift_mission = None
shift_found = False
for mid, data in sorted_missions:
    dip = data["diplomacy_count"]
    bud = data["buddhist_count"]
    total = data["events"]
    label = ""
    bd_ratio = bud / dip if dip > 0 else float("inf")
    if bud > dip:
        label = " ← BUDDHIST-DOMINANT"
        if not shift_found:
            shift_mission = mid
            shift_found = True
            label += " *** FIRST SHIFT ***"
    print(
        f"  {mid} (yr {data['departure_year']}): dip={dip}, bud={bud}, "
        f"bud/dip={bd_ratio:.2f}{' (BUDDHIST)' if bud > dip else ''}{' (DIPLOMACY)' if dip > bud else ''}{label}"
    )

print(f"\nFirst Buddhist-dominant mission: {shift_mission}")

# More nuanced: look at event type shift
print("\n--- Detailed event type evolution ---")
diplo_types = ["派遣遣唐使", "朝貢獻物", "接受賞賜", "返回日本"]
bud_types = ["師從學習", "受法受戒", "創立宗派", "傳入制度", "帶回文化物品"]
travel_type = "前往地點"
disaster_type = "海上遭難"
arrive_type = "到達唐朝"

for mid, data in sorted_missions:
    types = data["event_types"]
    print(
        f"  {mid}: 派遣={types.get('派遣遣唐使', 0)}, 貢獻={types.get('朝貢獻物', 0)}, "
        f"賞賜={types.get('接受賞賜', 0)}, 返回={types.get('返回日本', 0)}, "
        f"學習={types.get('師從學習', 0)}, 受法={types.get('受法受戒', 0)}, "
        f"創派={types.get('創立宗派', 0)}, 傳制={types.get('傳入制度', 0)}, "
        f"帶回={types.get('帶回文化物品', 0)}, 前往={types.get('前往地點', 0)}, "
        f"遭難={types.get('海上遭難', 0)}, 到達={types.get('到達唐朝', 0)}"
    )

# ============================================================
# 4. Bridge missions identification
# ============================================================
print("\n" + "=" * 80)
print("4. BRIDGE MISSIONS (MARKING TRANSITIONS)")
print("=" * 80)

# A bridge mission has significant presence in both diplomatic and Buddhist events
print("\n--- Missions with mixed diplomatic + Buddhist character ---")
for mid, data in sorted_missions:
    dip = data["diplomacy_count"]
    bud = data["buddhist_count"]
    total = data["events"]
    if total >= 5 and dip >= 2 and bud >= 2:
        print(
            f"  BRIDGE: {mid} (yr {data['departure_year']}): "
            f"total={total}, dip={dip}, bud={bud} — mixed character"
        )

# Also check for the first appearance of Buddhist events
print("\n--- First appearance of Buddhist-related events ---")
for et in ["師從學習", "受法受戒", "創立宗派", "傳入制度", "帶回文化物品"]:
    first_found = False
    for mid, data in sorted_missions:
        if data["event_types"].get(et, 0) > 0 and not first_found:
            print(f"  '{et}' first appears in {mid} (yr {data['departure_year']})")
            first_found = True

# ============================================================
# 5. Cumulative metrics and inflection points
# ============================================================
print("\n" + "=" * 80)
print("5. CUMULATIVE METRICS AND INFLECTION POINTS")
print("=" * 80)

print("\n--- Cumulative counts per mission ---")
print(
    f"{'Mission':>8} {'cum_events':>12} {'cum_dip':>10} {'cum_bud':>10} {'cum_dip%':>10} {'cum_bud%':>10}"
)
for i, mid in enumerate(missions_ordered):
    cum_dip_pct = cum_dip[i] / cum_events[i] * 100 if cum_events[i] else 0
    cum_bud_pct = cum_bud[i] / cum_events[i] * 100 if cum_events[i] else 0
    print(
        f"  {mid:>6} {cum_events[i]:>12} {cum_dip[i]:>10} {cum_bud[i]:>10} {cum_dip_pct:>9.1f}% {cum_bud_pct:>9.1f}%"
    )

# Detect inflection points in cumulative buddhist curve
# Inflection = where second derivative changes sign
if n >= 3:
    bud_first_diff = diff(cum_bud)
    print("\n--- First differences of cumulative Buddhist events ---")
    for i in range(len(bud_first_diff)):
        print(
            f"  {missions_ordered[i]}→{missions_ordered[i + 1]}: Δ={bud_first_diff[i]:+.0f}"
        )

    if n >= 4:
        bud_second_diff = diff(bud_first_diff)
        print(
            "\n--- Second differences of cumulative Buddhist (inflection detection) ---"
        )
        for i in range(len(bud_second_diff)):
            change = "↑ accelerating" if bud_second_diff[i] > 0 else "↓ decelerating"
            print(
                f"  {missions_ordered[i + 1]}: δ²={bud_second_diff[i]:+.0f} ({change})"
            )

        # Find inflection (sign change in second derivative)
        for i in range(1, len(bud_second_diff)):
            if bud_second_diff[i - 1] * bud_second_diff[i] < 0:
                print(
                    f"  >>> INFLECTION POINT at {missions_ordered[i + 1]} "
                    f"(second derivative changes sign from {bud_second_diff[i - 1]:+d} to {bud_second_diff[i]:+d})"
                )

# Same for events
if n >= 4:
    event_first_diff = diff(cum_events)
    event_second_diff = diff(event_first_diff)
    print("\n--- Second differences of cumulative events (inflection detection) ---")
    for i in range(len(event_second_diff)):
        change = "↑ accelerating" if event_second_diff[i] > 0 else "↓ decelerating"
        print(f"  {missions_ordered[i + 1]}: δ²={event_second_diff[i]:+.0f} ({change})")

    for i in range(1, len(event_second_diff)):
        if event_second_diff[i - 1] * event_second_diff[i] < 0:
            print(
                f"  >>> INFLECTION POINT at {missions_ordered[i + 1]} "
                f"(second derivative changes sign from {event_second_diff[i - 1]:+d} to {event_second_diff[i]:+d})"
            )

# ============================================================
# 6. Entity type distribution shifts
# ============================================================
print("\n" + "=" * 80)
print("6. ENTITY TYPE DISTRIBUTION SHIFTS")
print("=" * 80)

# Track per-mission unique subjects and locations
for mid, data in sorted_missions:
    print(
        f"  {mid}: {len(data['subjects'])} unique subjects, {len(data['locations'])} unique locations, "
        f"{data['events']} events"
    )

# Detect where subject diversity jumps
print("\n--- Entity diversity jumps (>2x) ---")
for i in range(1, n):
    prev_subj = len(sorted_missions[i - 1][1]["subjects"])
    curr_subj = len(sorted_missions[i][1]["subjects"])
    if prev_subj > 0 and curr_subj / prev_subj >= 2.0:
        print(
            f"  Subject jump: {sorted_missions[i - 1][0]}→{sorted_missions[i][0]} "
            f"({prev_subj}→{curr_subj} subjects, {curr_subj / prev_subj:.1f}x)"
        )

# ============================================================
# 7. Network density / connectivity analysis from raw events
# ============================================================
print("\n" + "=" * 80)
print("7. EVENT DENSITY RATIOS (as proxy for network density)")
print("=" * 80)

# Events per subject (higher = denser interactions)
for mid, data in sorted_missions:
    n_subj = len(data["subjects"]) if data["subjects"] else 1
    density_proxy = data["events"] / n_subj
    print(
        f"  {mid}: events/subject ratio = {density_proxy:.2f} "
        f"({data['events']} events / {len(data['subjects'])} subjects)"
    )

print("\n--- Done ---")
