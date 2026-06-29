# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Kento-shi Network** (遣唐使ネットワーク) — a knowledge graph of Japanese missions to Tang China (630–894 CE). Extracts entities (Person, Event, Place, Culture) and relations (PARTICIPATE, LEARN, TRAVEL, INTRODUCE) from Japanese and Chinese classical historical texts. Final output: `nodes.csv` + `edges.csv` for NetworkX/Neo4j/Gephi analysis.

**Hard constraints** (from `00_PROMPT.md`):
- All relations must have traceable source text — no inferred/fabricated data
- Minimum scale targets: Person ≥ 100, Event ≥ 20, Relation ≥ 500
- If any dimension fails, do NOT proceed to Phase 5 (CSV output)

## Commands

```bash
# Install dependencies
pixi install

# Run specific ETL phases
pixi run python -m etl.phase1_structured_extract
pixi run python -m etl.phase2_chronicle_match
pixi run python -m etl.phase3_prep_chunks
pixi run python -m etl.phase3_merge_results
pixi run python -m etl.phase4_transform
pixi run python -m etl.phase5_load

# Full pipeline (Phases 1-2, then prompts for Phase 3 Workflow, then 4-5)
pixi run python -m etl.orchestrator

# Phase 4+5 only (after Phase 3 Workflow completed)
pixi run python -m etl.orchestrator --phase4-5

# Check Phase 3 status
pixi run python -m etl.orchestrator --phase3
```

## Architecture

### Data Flow
```
kentoshi_missions.json ──→ Phase 1 (structured extraction) ──→ phase1_nodes/edges.json
japan_clean.txt          ──→ Phase 2 (pattern matching)      ──→ phase2_nodes/edges.json
china_clean.txt          ──┘
ennin diary + zenrin     ──→ Phase 3 (Workflow LLM extraction) → phase3_nodes/edges.json
                              ↓
                         Phase 4 (disambiguation, dedup, ID remap) → nodes/edges.json
                              ↓ (scale check: Person≥100, Event≥20, Relation≥500)
                         Phase 5 (CSV output) → nodes.csv, edges.csv
```

### Data Sources (`data/`)

| Source | Size | Description |
|--------|------|-------------|
| `kentoshi_missions.json` | 16K | 21 mission records (M01–M20+M08b), clean JSON |
| `cleaned/japan_clean.txt` | 4.3M | 六国史 190 volumes with `# work 卷XX` headers |
| `cleaned/china_clean.txt` | 14M | 舊唐書+新唐書 with `# 書名` section markers |
| `cleaned/supplementary/ennin/` | ~100K | 入唐求法巡礼行記 (Ennin's diary, 4 vols), **clean** |
| `cleaned/supplementary/zenrin_kokuhoki.txt` | ~35K | 善鄰國寶記, **clean** |
| `cleaned/supplementary/kukai_shorai_mokuroku.txt` | ~20K | 御請來目録 (Kūkai's catalog), recovered from SAT Daizokyo |
| `cleaned/supplementary/toudaiwajo_toseiden.txt` | ~34K | 唐大和上東征傳 (Ganjin biography), recovered from Wikisource |
| `cleaned/supplementary/nittou_gokaden_eun.txt` | ~2K | 入唐五家傳 慧運傳, recovered from supplementary_clean.txt |
| `cleaned/supplementary/enchin_den.txt` | — | **Removed** (corrupted, pending recovery) |
| `cleaned/supplementary/nittou_gokaden_shinnyo.txt` | — | **Removed** (corrupted, pending recovery) |

### Key Design Decisions

**Phase 3 uses Workflow sub-agents, NOT the Anthropic API.** The `phase3_merge_results.py` script reads checkpoint JSON (written by Workflow agents) and converts triples to nodes/edges. Workflow scripts are plain JS without Node.js APIs — pass data via `args`, not `require('fs')`. Use `schema` parameter in `agent()` calls for structured triple extraction output.

**Encoding corruption handling**: 2 of 9 supplementary files still corrupted (enchin_den, nittou_gokaden_shinnyo). 3 recovered from online sources.

**Phase 2 edge dedup**: Edges from chronicle matching are deduped by `(source_id, target_id, relation)` only — not by source_text. This prevents 60%+ noise from multiple mentions of the same entity pair.

**Place name normalization**: Simplified Chinese place names are mapped to traditional canonical forms via `PLACE_NORMALIZE` in `config.py` (e.g., 长安→長安, 洛阳→洛陽).

**Person name splitting**: The `_split_person_names()` function in Phase 1 handles compound entries like `"大伴山守（大使）、藤原馬養（副使）"` — splitting on `、` before stripping role parens.

**Phase 4 disambiguation log** only records groups where actual merging happened (different original IDs for the same canonical name). Single-person groups are not logged.

### Scale Check Gate

Between Phase 4 and Phase 5, `check_scale()` validates:
```python
Person ≥ 100, Event ≥ 20, Relation ≥ 500
```
If any fails, Phase 5 is blocked. The orchestrator prints gap analysis and instructions.

### Entity Type Prefixes
- `P001–P999` — Person
- `E001–E999` — Event
- `L001–L999` — Place
- `C001–C999` — Culture

IDs are reassigned sequentially by type during Phase 4. Intermediate phases use provisional IDs; only `nodes.json`/`nodes.csv` from Phase 4+ have final canonical IDs.

### Dependency Management
- **pixi** for Python packages (pandas, python-dotenv)
- **mise** manages the pixi tool itself (`mise.toml`)
- No pip/conda — use `pixi install` and `pixi run python`
