"""Configuration for the embedding pipeline."""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = ROOT / "output"

# Source files to process (combined + clean supplementary)
SOURCE_FILES: list[Path] = [
    CLEANED_DIR / "japan_clean.txt",
    CLEANED_DIR / "china_clean.txt",
    CLEANED_DIR / "supplementary" / "ennin" / "ennin_vol01.txt",
    CLEANED_DIR / "supplementary" / "ennin" / "ennin_vol02.txt",
    CLEANED_DIR / "supplementary" / "ennin" / "ennin_vol03.txt",
    CLEANED_DIR / "supplementary" / "ennin" / "ennin_vol04.txt",
    CLEANED_DIR / "supplementary" / "zenrin_kokuhoki.txt",
    CLEANED_DIR / "supplementary" / "kukai_shorai_mokuroku.txt",
    CLEANED_DIR / "supplementary" / "toudaiwajo_toseiden.txt",
    CLEANED_DIR / "supplementary" / "nittou_gokaden_eun.txt",
]

# Files to skip (still corrupted / irrecoverable)
SKIP_FILES = {
    "enchin_den.txt",
    "nittou_gokaden_shinnyo.txt",
    "supplementary_clean.txt",
}

# Chunking parameters
CHUNK_SIZE = 200  # number of CJK characters per chunk
CHUNK_OVERLAP = 50  # overlap in CJK characters
MIN_CHUNK_LENGTH = 10  # minimum CJK characters to keep a chunk

# Output files
CHUNKS_JSONL = OUTPUT_DIR / "chunks.jsonl"
VECTORS_NPY = OUTPUT_DIR / "vectors.npy"
FAISS_INDEX = OUTPUT_DIR / "faiss.index"
RETRIEVED_JSONL = OUTPUT_DIR / "retrieved.jsonl"
EXTRACTED_EVENTS_DIR = OUTPUT_DIR / "extracted_events"
MERGED_EVENTS_JSON = OUTPUT_DIR / "merged_events.json"
PIPELINE_NODES_CSV = OUTPUT_DIR / "nodes.csv"
PIPELINE_EDGES_CSV = OUTPUT_DIR / "edges.csv"

# Model config
MODEL_NAME = "SIKU-BERT/sikuroberta"
MODEL_FALLBACK = "iic/nlp_sikuroberta_fill-mask_chinese-base"  # ModelScope mirror
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512

# Retrieval config
# Higher TOP_K and lower threshold per user directive: prefer recall over precision
TOP_K_PER_QUERY = 800  # top results per query (was 500)
SIMILARITY_THRESHOLD = 0.45  # minimum cosine similarity (was 0.5)

# Post-retrieval filter: require at least one Tang-specific multi-character term.
# Prevents SikuRoBERTa from conflating kentoshi with Korean peninsula diplomacy.
# Note: bare '唐' removed — matches too many non-kentoshi strings (surnames, etc.)
TANG_KEYWORDS = [
    "遣唐",
    "入唐",
    "大唐",
    "唐朝",
    "唐人",
    "長安",
    "洛陽",
    "遣唐使",
    "聘唐",
    "唐使",
    "唐客",
    "唐舶",
    "來朝長安",
    "入唐求法",
    "遣使入唐",
    "隨使入唐",
    "入唐學問",
    "入唐留学",
    "入唐學",
    "唐國",
    "唐制",
    "唐禮",
    "在唐",
    "至唐",
]

# Event extraction
EXTRACTION_BATCH_SIZE = 8  # chunks per LLM call

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def is_cjk_char(ch: str) -> bool:
    """Check if a character is a CJK unified ideograph or compatible character.

    Covers:
    - CJK Unified Ideographs (U+4E00-U+9FFF)
    - CJK Unified Ideographs Extension A (U+3400-U+4DBF)
    - CJK Compatibility Ideographs (U+F900-U+FAFF)
    - CJK Unified Ideographs Extensions B-F (U+20000-U+2EBEF)
    - CJK Compatibility Ideographs Supplement (U+2F800-U+2FA1F)
    - CJK Unified Ideographs Extensions G-H (U+30000-U+323AF)
    - CJK Unified Ideographs Extension I (U+2EBF0-U+2EE5F)
    """
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x20000 <= cp <= 0x2EBEF)
        or (0x2F800 <= cp <= 0x2FA1F)
        or (0x30000 <= cp <= 0x323AF)
        or (0x2EBF0 <= cp <= 0x2EE5F)
    )
