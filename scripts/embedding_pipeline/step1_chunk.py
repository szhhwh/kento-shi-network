#!/usr/bin/env python3
"""Step 1: Split source texts into overlapping CJK-character chunks.

Reads all clean source files, splits into 200-CJK-character chunks with
50-character overlap, tracks source and section headers, outputs JSONL.
"""

import json
import re
import sys
from pathlib import Path

from embedding_pipeline.config import (
    SOURCE_FILES,
    SKIP_FILES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
    CHUNKS_JSONL,
    is_cjk_char,
)


def extract_cjk_chars(text: str) -> list[int]:
    """Return indices of all CJK characters in text."""
    return [i for i, ch in enumerate(text) if is_cjk_char(ch)]


def chunk_text(
    text: str,
    source_file: str,
    base_offset: int = 0,
    section: str = "",
) -> list[dict]:
    """Split text into overlapping CJK-character chunks.

    Args:
        text: Input text to chunk
        source_file: Source filename for tracking
        base_offset: Character offset in the source file where this text begins
        section: Current section/volume header (e.g., '# montoku 卷01')

    Returns:
        List of chunk dicts with: id, text, source_file, offset, section, cjk_len
    """
    cjk_indices = extract_cjk_chars(text)
    total_cjk = len(cjk_indices)

    if total_cjk < MIN_CHUNK_LENGTH:
        return []

    step = CHUNK_SIZE - CHUNK_OVERLAP  # 150 chars forward per chunk
    chunks = []

    chunk_idx = 0
    for start_cjk_pos in range(0, total_cjk, step):
        end_cjk_pos = min(start_cjk_pos + CHUNK_SIZE, total_cjk)

        # Map CJK positions back to original string offsets
        char_start = cjk_indices[start_cjk_pos]
        if end_cjk_pos < total_cjk:
            char_end = cjk_indices[end_cjk_pos]
        else:
            char_end = len(text)

        chunk_text_val = text[char_start:char_end]

        # Count actual CJK chars in this chunk
        cjk_count = sum(1 for ch in chunk_text_val if is_cjk_char(ch))

        if cjk_count < MIN_CHUNK_LENGTH:
            continue

        chunks.append(
            {
                "chunk_id": chunk_idx,
                "text": chunk_text_val.strip(),
                "source_file": source_file,
                "source_offset": base_offset + char_start,
                "section": section,
                "cjk_length": cjk_count,
            }
        )
        chunk_idx += 1

    return chunks


def read_source_file(filepath: Path) -> list[dict]:
    """Read a source file and chunk it, tracking section headers.

    For combined files (japan_clean.txt, china_clean.txt), lines starting
    with '# ' are section markers that update the current section context.
    """
    filename = filepath.name

    if filename in SKIP_FILES:
        print(f"  [SKIP] {filepath} (corrupted, in skip list)")
        return []

    # Check for U+FFFD corruption
    try:
        raw = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"  [SKIP] {filepath} (encoding error)")
        return []

    if "\ufffd" in raw[:10000]:  # check first 10K chars
        print(f"  [SKIP] {filepath} (contains U+FFFD replacement chars)")
        return []

    # For combined files with section headers, split by sections first
    section_pattern = re.compile(r"^(# .+)$", re.MULTILINE)

    if "# " in raw[:200] or bool(section_pattern.search(raw)):
        # This file has section headers; split into sections
        return _chunk_with_sections(raw, str(filepath), section_pattern)
    else:
        # Simple file, no section headers
        chunks = chunk_text(raw, str(filepath), section="")
        print(f"  {filepath}: {len(chunks)} chunks from {len(raw)} chars")
        return chunks


def _chunk_with_sections(
    text: str, source_name: str, section_pattern: re.Pattern
) -> list[dict]:
    """Chunk a file that has '# section' headers, preserving section context."""
    all_chunks = []
    sections = []

    # Find all section headers and their positions
    for m in section_pattern.finditer(text):
        sections.append((m.start(), m.group(1)))

    if not sections:
        return chunk_text(text, source_name, section="")

    # Process text between section markers
    for i, (sec_start, sec_header) in enumerate(sections):
        # Text from this header to the next header (or end of file)
        sec_text_start = sec_start + len(sec_header) + 1  # skip header line + newline
        if i + 1 < len(sections):
            sec_text_end = sections[i + 1][0]
        else:
            sec_text_end = len(text)

        sec_text = text[sec_text_start:sec_text_end]

        chunks = chunk_text(
            sec_text,
            source_name,
            base_offset=sec_text_start,
            section=sec_header,
        )
        all_chunks.extend(chunks)

    # Reassign chunk IDs sequentially
    for new_id, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = new_id

    print(f"  {source_name}: {len(all_chunks)} chunks across {len(sections)} sections")
    return all_chunks


def main():
    """Run text chunking on all source files."""
    print("=" * 60)
    print("Step 1: Text Chunking")
    print("=" * 60)
    print(f"  Chunk size: {CHUNK_SIZE} CJK chars")
    print(f"  Overlap:    {CHUNK_OVERLAP} CJK chars")
    print(f"  Min length: {MIN_CHUNK_LENGTH} CJK chars")
    print()

    all_chunks = []

    for src_file in SOURCE_FILES:
        if not src_file.exists():
            print(f"  [MISSING] {src_file}")
            continue
        chunks = read_source_file(src_file)
        all_chunks.extend(chunks)

    # Reassign global sequential IDs
    for new_id, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = new_id

    # Write JSONL output
    CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Summary
    total_cjk = sum(ch["cjk_length"] for ch in all_chunks)
    total_chars = sum(len(ch["text"]) for ch in all_chunks)
    unique_sources = sorted(set(ch["source_file"] for ch in all_chunks))
    unique_sections = sorted(set(ch["section"] for ch in all_chunks if ch["section"]))

    print()
    print(f"  Total chunks:       {len(all_chunks)}")
    print(f"  Total CJK chars:    {total_cjk:,}")
    print(f"  Total raw chars:    {total_chars:,}")
    print(f"  Unique sources:     {len(unique_sources)}")
    print(f"  Unique sections:    {len(unique_sections)}")
    print(f"  Output:             {CHUNKS_JSONL}")
    print()

    if len(all_chunks) == 0:
        print("ERROR: No chunks generated!", file=sys.stderr)
        sys.exit(1)

    print("Step 1 complete.")
    return all_chunks


if __name__ == "__main__":
    main()
