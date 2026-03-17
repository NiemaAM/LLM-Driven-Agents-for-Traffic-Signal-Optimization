"""
Full Milestone 3 data pipeline orchestrator.

Runs all steps in order and logs timing for each step.
Uses ZenML decorators for tracking when available,
falls back to direct execution if ZenML server is not running.
"""

import os
import time
import traceback

PYTHONPATH_SET = os.environ.get("PYTHONPATH", "")
if "." not in PYTHONPATH_SET:
    os.environ["PYTHONPATH"] = "." + os.pathsep + PYTHONPATH_SET


def run_step(name: str, fn) -> bool:
    """Run a pipeline step, log timing, catch and report errors."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"\n  ✅ {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ {name} FAILED after {elapsed:.1f}s")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def step_ingest():
    from pipeline.ingest import ingest
    train_df, eval_df = ingest()
    print(f"  → {len(train_df)} train rows, {len(eval_df)} eval rows")


def step_validate():
    from pipeline.validate import run_validation
    schema = run_validation()
    print(f"  → Schema saved with {schema['row_count']} total rows")


def step_transform():
    from pipeline.transform import transform
    df = transform()
    print(f"  → {len(df)} rows transformed, {len(df.columns)} features")


def step_feature_store():
    from pipeline.feature_store import (
        prepare_parquet, write_feature_repo, apply_and_materialize
    )
    parquet_path = prepare_parquet()
    write_feature_repo(parquet_path)
    apply_and_materialize()


def run_pipeline():
    print("\n" + "="*60)
    print("  MILESTONE 3 — DATA PIPELINE")
    print("  Traffic Signal Optimization Project")
    print("="*60)

    steps = [
        ("1. Data Ingestion",       step_ingest),
        ("2. Data Validation",      step_validate),
        ("3. Feature Engineering",  step_transform),
        ("4. Feature Store",        step_feature_store),
    ]

    results = {}
    pipeline_start = time.time()

    for name, fn in steps:
        results[name] = run_step(name, fn)

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}  {name}")

    passed = sum(results.values())
    total  = len(results)
    print(f"\n  {passed}/{total} steps passed  |  total time: {total_elapsed:.1f}s")
    print("="*60)

    if passed == total:
        print("\n  🎉 All steps completed successfully!")
        print("  Outputs:")
        print("    data/processed/train.parquet")
        print("    data/processed/eval.parquet")
        print("    data/processed/transformed_dataset.csv")
        print("    data/processed/feature_metadata.json")
        print("    data/schema/schema.json")
        print("    data/schema/anomalies.json")
        print("    data/feature_store/vehicle_features.parquet")
        print("    data/feature_store/registry.db")
    else:
        failed = [n for n, s in results.items() if not s]
        print(f"\n  ⚠ Failed steps: {', '.join(failed)}")

    return passed == total


if __name__ == "__main__":
    success = run_pipeline()
    exit(0 if success else 1)