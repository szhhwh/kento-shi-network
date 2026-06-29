#!/usr/bin/env python3
"""Step 6: Event extraction — prepares batches and event extraction prompts.

This script:
1. Reads retrieved.jsonl (output of step 5)
2. Splits chunks into batches of EXTRACTION_BATCH_SIZE
3. Creates batch task files with system prompt + chunk texts
4. Outputs a manifest for the orchestrator to process

The actual LLM extraction is done by the Hermes agent via delegate_task
for each batch. This script only prepares the data.
"""

import json
import sys

from embedding_pipeline.config import (
    RETRIEVED_JSONL,
    EXTRACTED_EVENTS_DIR,
    EXTRACTION_BATCH_SIZE,
)

# ── Event Extraction System Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """你是一位精通中國唐代歷史和日本古代史的古籍信息抽取專家。你正在閱讀遣唐使相關的史料段落，需要從中提取結構化的事件信息。

## 事件類型定義

你只能提取以下十四種事件類型，每種類型有嚴格的定義和示例：

### 1. 派遣遣唐使
定義：**官方任命大使/副使、遣唐使團正式出發**的記載。必須有任命或使團出發的明確動作。
示例：天皇任命藤原常嗣為遣唐大使。
關鍵詞：任命~大使、拜遣唐使、為遣唐大使、遣唐使出發、節刀、遣於唐國
**注意**：如果原文只說某人"入唐"、"赴唐"、"渡海至唐"而沒有任命或遣唐使團出發，應歸類為"到達唐朝"或"前往地點"，而非此類型。

### 2. 到達唐朝
定義：遣唐使或個人**到達唐朝境內任何地點**（長安、洛陽、揚州、明州等）的記載。包括個人跟隨使團入唐。
示例：遣唐使到達長安。空海跟隨遣唐使入唐，到達福州。
關鍵詞：到達、至、入唐、抵達某地、著某州、到某州
**注意**：個人"入唐"、"赴唐"、"隨使入唐"屬於此類型。

### 3. 返回日本
定義：遣唐使或個人從唐朝返回日本的記載。
示例：空海攜帶經論歸國。
關鍵詞：歸國、返日、回國、歸朝、還至

### 4. 師從學習
定義：日本人在唐朝明確向某人學習的記載，要求原文有明確的師從或學習關係。
示例：空海師從惠果學習密教。
關鍵詞：師從、從某學、受業、師事、稟受

### 5. 受法受戒
定義：日本人在唐朝接受佛法傳授或戒律的記載。
示例：最澄在天台山受菩薩戒。
關鍵詞：受戒、受法、灌頂、受菩薩戒、受具足戒、受兩部大法

### 6. 帶回文化物品
定義：日本人從唐朝帶回書籍、經典、佛像、法器等文化物品的記載。
示例：吉備真備攜《唐禮》歸國。
關鍵詞：請來、將來、將來經論、請來佛像、齎來、攜歸

### 7. 創立宗派
定義：日本人在日本創立佛教宗派的記載（通常基於從唐朝學到的內容）。
示例：最澄創立日本天台宗。
關鍵詞：創立、開創、開宗、建立宗派

### 8. 傳入制度
定義：日本人將唐朝制度、律令、曆法、官制等引入日本的記載。
示例：高向玄理參與大化改新引入唐制。
關鍵詞：引入、傳入、採用唐、仿唐、行唐制

### 9. 朝貢獻物
定義：遣唐使向唐朝皇帝進貢物品的記載。
示例：遣唐使向唐朝皇帝進貢。
關鍵詞：朝貢、進貢、貢獻、獻上

### 10. 接受賞賜
定義：唐朝皇帝賞賜物品給遣唐使或日本人的記載。
示例：唐朝皇帝賜物給遣唐使。
關鍵詞：賞賜、賜、賜物、賜姓、賜官

### 11. 前往地點
定義：日本人在唐朝境內移動到特定地點的記載（寺廟、名山等）。
示例：圓仁到達五台山。
關鍵詞：前往、至、到、登、入（某寺某山）

### 12. 賦詩唱和
定義：遣唐使或日本人與唐朝文人之間賦詩唱和的記載。
示例：阿倍仲麻呂與唐朝詩人賦詩唱和。
關鍵詞：賦詩、唱和、贈詩、和詩、詩會

### 13. 海上遭難
定義：遣唐使船在海上遭遇風暴、漂流、覆沒的記載。
示例：遣唐使船在海上遭難。
關鍵詞：遇難、遭風、漂著、覆沒、沉沒、遇風

### 14. 授予官職
定義：唐朝皇帝授予日本人官職的記載。
示例：唐朝授予阿倍仲麻呂官職。
關鍵詞：授官、拜官、除授、賜官

## 抽取原則（必須嚴格遵守）

1. **必須有原文依據**：每一條提取的事件都必須能在給定的段落原文中找到直接的文字支持。絕對不要推斷或猜測。
2. **不要因詞語共現就判定關係**：不可以僅因為「某個人名和某個關鍵詞同時出現」就認為有關係。必須有明確的動作動詞。
3. **使用原文中的名稱**：事件的主體和客體使用原文中的稱呼，不要翻譯或改寫成現代名稱。
4. **同一段多個事件分別提取**：一個段落中可能包含多個事件，需要逐一提取。
5. **模糊記載不要猜測**：如果原文只說了年份但沒有具體人名，不要猜測人名填進去。
6. **沒有事件就返回空數組**：如果段落中沒有符合上述類型的事件，返回空數組 []。

## 輸出格式

對於每個事件，輸出以下 JSON 結構：
{
  "event_type": "事件類型名稱（如：派遣遣唐使）",
  "subject": "事件主體（人名或團名，使用原文稱呼）",
  "action": "動作描述（簡短概括）",
  "object": "事件客體（人名、地名、物品名，可為空字符串）",
  "location": "發生地點（可為空字符串）",
  "time_period": "時間信息（原文中出現的年號或年份，可為空字符串）",
  "source_text": "支撐該事件的原文摘錄（從段落中摘取的最相關的一兩句話）"
}

返回一個 JSON 數組，包含該批次所有段落中提取的所有事件。如果某個段落沒有事件，就不要為該段落產生任何輸出。
"""


def prepare_batches():
    """Read retrieved chunks, split into batches, create task files."""
    print("=" * 60)
    print("Step 6: Event Extraction — Preparation")
    print("=" * 60)

    # Load retrieved chunks
    print(f"\n  Loading retrieved chunks from: {RETRIEVED_JSONL}")
    chunks = []
    with open(RETRIEVED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  Total retrieved chunks: {len(chunks)}")

    if len(chunks) == 0:
        print("ERROR: No chunks to extract from!")
        sys.exit(1)

    # Split into batches
    batches = []
    for i in range(0, len(chunks), EXTRACTION_BATCH_SIZE):
        batch_chunks = chunks[i : i + EXTRACTION_BATCH_SIZE]
        batches.append(batch_chunks)

    print(f"  Batches: {len(batches)} (size={EXTRACTION_BATCH_SIZE})")

    # Create batch task files
    EXTRACTED_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for batch_idx, batch_chunks in enumerate(batches):
        # Build the user prompt for this batch
        chunks_text = ""
        for ci, chunk in enumerate(batch_chunks):
            local_id = batch_idx * EXTRACTION_BATCH_SIZE + ci
            chunks_text += (
                f"--- 段落 {local_id} ---\n"
                f"來源：{chunk.get('source_file', '未知')}\n"
                f"章節：{chunk.get('section', '未知')}\n"
                f"文本：\n{chunk['text']}\n\n"
            )

        batch_file = EXTRACTED_EVENTS_DIR / f"batch_{batch_idx:04d}_input.json"
        batch_output = EXTRACTED_EVENTS_DIR / f"batch_{batch_idx:04d}_output.json"

        batch_data = {
            "batch_id": batch_idx,
            "total_batches": len(batches),
            "system_prompt": SYSTEM_PROMPT,
            "chunks": [
                {
                    "local_id": batch_idx * EXTRACTION_BATCH_SIZE + ci,
                    "chunk_id": ch["chunk_id"],
                    "text": ch["text"],
                    "source_file": ch.get("source_file", ""),
                    "section": ch.get("section", ""),
                }
                for ci, ch in enumerate(batch_chunks)
            ],
            "expected_output": str(batch_output),
        }

        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)

        manifest.append(
            {
                "batch_id": batch_idx,
                "input_file": str(batch_file),
                "output_file": str(batch_output),
                "chunk_count": len(batch_chunks),
                "chunk_ids": [ch["chunk_id"] for ch in batch_chunks],
            }
        )

    # Write manifest
    manifest_file = EXTRACTED_EVENTS_DIR / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  Batch files: {len(batches)} in {EXTRACTED_EVENTS_DIR}")
    print(f"  Manifest: {manifest_file}")
    print("\n  Next: process batches using delegate_task or direct API calls")
    print("Step 6 preparation complete.")

    return manifest


if __name__ == "__main__":
    prepare_batches()
