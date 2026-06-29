#!/usr/bin/env python3
"""Orchestrator for the embedding retrieval + event extraction pipeline.

Usage:
    # Run all steps
    pixi run python -m embedding_pipeline.orchestrator

    # Run specific steps
    pixi run python -m embedding_pipeline.orchestrator --step1   # chunking
    pixi run python -m embedding_pipeline.orchestrator --step3   # encoding
    pixi run python -m embedding_pipeline.orchestrator --step4   # FAISS index
    pixi run python -m embedding_pipeline.orchestrator --step5   # retrieval
    pixi run python -m embedding_pipeline.orchestrator --step6-prep  # prepare batches
    pixi run python -m embedding_pipeline.orchestrator --step7   # merge
    pixi run python -m embedding_pipeline.orchestrator --step8   # build KG

    # Resume from a step
    pixi run python -m embedding_pipeline.orchestrator --from step3
"""

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = {
    "step1": {
        "module": "embedding_pipeline.step1_chunk",
        "description": "Text chunking (200 CJK chars, 50 overlap)",
        "output": "output/chunks.jsonl",
    },
    "step3": {
        "module": "embedding_pipeline.step3_encode",
        "description": "Text encoding (SikuRoBERTa -> 768-dim vectors)",
        "output": "output/vectors.npy",
        "requires": ["step1"],
    },
    "step4": {
        "module": "embedding_pipeline.step4_index",
        "description": "Build FAISS FlatIP index",
        "output": "output/faiss.index",
        "requires": ["step3"],
    },
    "step5": {
        "module": "embedding_pipeline.step5_retrieve",
        "description": "Semantic retrieval (5 query groups)",
        "output": "output/retrieved.jsonl",
        "requires": ["step4"],
    },
    "step6-prep": {
        "module": "embedding_pipeline.step6_extract",
        "description": "Prepare event extraction batches",
        "output": "output/extracted_events/manifest.json",
        "requires": ["step5"],
    },
    "step7": {
        "module": "embedding_pipeline.step7_merge",
        "description": "Merge, dedup & disambiguate events",
        "output": "output/merged_events.json",
        "requires": ["step6-prep"],
    },
    "step8": {
        "module": "embedding_pipeline.step8_kg",
        "description": "Build knowledge graph (nodes.csv + edges.csv)",
        "output": "output/nodes.csv",
        "requires": ["step7"],
    },
}

STEP_ORDER = ["step1", "step3", "step4", "step5", "step6-prep", "step7", "step8"]


def run_module(module: str) -> bool:
    """Run a Python module and return True if successful."""
    print(f"\n{'=' * 60}")
    print(f"  Running: {module}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=Path(__file__).resolve().parent.parent,
    )
    return result.returncode == 0


def check_output(output_rel: str, base_dir: Path) -> bool:
    """Check if an output file exists."""
    path = base_dir / output_rel
    return path.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Kento-shi Embedding Pipeline Orchestrator"
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        choices=STEP_ORDER,
        help="Resume from a specific step",
    )
    for step_name, info in STEPS.items():
        parser.add_argument(
            f"--{step_name}",
            action="store_true",
            help=f"Run only {step_name}: {info['description']}",
        )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent

    # Determine which step(s) to run
    single_steps = [
        name for name in STEPS if getattr(args, name.replace("-", "_"), False)
    ]

    if single_steps:
        # Run only the specified step(s)
        for step_name in single_steps:
            info = STEPS[step_name]
            print(f"Step: {step_name} — {info['description']}")
            if not run_module(info["module"]):
                print(f"\nERROR: {step_name} failed!")
                sys.exit(1)
            print(f"\n{step_name} completed successfully.")
        return

    # Run full pipeline or from a specific step
    start_idx = 0
    if args.from_step:
        start_idx = STEP_ORDER.index(args.from_step)
        print(f"Resuming from: {args.from_step}")

    print("Kento-shi Embedding Pipeline")
    print("=" * 60)

    for step_name in STEP_ORDER[start_idx:]:
        info = STEPS[step_name]

        # Check if output already exists
        output_path = base_dir / info["output"]
        if output_path.exists():
            print(f"\n[{step_name}] Output already exists: {info['output']}")
            print(f"  Skipping. Delete {info['output']} to re-run.")
            continue

        # Check requirements
        skip = False
        for req in info.get("requires", []):
            req_info = STEPS[req]
            req_path = base_dir / req_info["output"]
            if not req_path.exists():
                print(
                    f"\n[{step_name}] Requirement '{req}' not met "
                    f"(missing {req_info['output']})"
                )
                print(f"  Run '{req}' first or check output.")
                skip = True
                break
        if skip:
            sys.exit(1)

        print(f"\n{'─' * 60}")
        print(f"  Step: {step_name}")
        print(f"  {info['description']}")
        print(f"{'─' * 60}")

        if not run_module(info["module"]):
            print(f"\nERROR: {step_name} failed!")
            sys.exit(1)

        print(f"\n{step_name} completed successfully.")

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print(f"{'=' * 60}")
    print("\nNote: Step 6 (event extraction) requires LLM processing.")
    print("After step6-prep, process batches manually or via delegate_task.")
    print("Then run step7 and step8 to generate the knowledge graph.")


if __name__ == "__main__":
    main()
