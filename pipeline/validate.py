"""Step 2 – Data validation using Great Expectations v0.18."""

import os
import json
import pandas as pd
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

RAW_CSV    = os.path.join("data", "raw", "generated_dataset.csv")
SCHEMA_DIR = os.path.join("data", "schema")
os.makedirs(SCHEMA_DIR, exist_ok=True)


def validate_dataframe(df: pd.DataFrame, name: str) -> dict:
    """
    Run all expectations directly on a PandasDataset (no context needed).
    Returns a dict with success flag and list of failed expectations.
    """
    gdf = PandasDataset(df)

    results = []

    def check(result, description):
        success = result["success"]
        results.append({
            "expectation": description,
            "success":     success,
            "details":     str(result.get("result", {})),
        })
        status = "✅" if success else "⚠"
        print(f"  {status} {description}")
        return success

    print(f"\n--- Validating {name} ({len(df)} rows) ---")

    # ── Schema: required columns ──────────────────────────────────────────
    for col in ["vehicle_id", "lane", "speed", "distance_to_intersection",
                "direction", "destination", "is_conflict", "number_of_conflicts"]:
        check(gdf.expect_column_to_exist(col), f"column '{col}' exists")

    # ── No nulls ──────────────────────────────────────────────────────────
    for col in ["vehicle_id", "speed", "distance_to_intersection",
                "direction", "destination", "is_conflict"]:
        check(gdf.expect_column_values_to_not_be_null(col),
              f"no nulls in '{col}'")

    # ── Lane: 1–8 ─────────────────────────────────────────────────────────
    check(gdf.expect_column_values_to_be_between("lane", 1, 8),
          "lane between 1 and 8")

    # ── Speed: 0–200 km/h ─────────────────────────────────────────────────
    check(gdf.expect_column_values_to_be_between("speed", 0, 200),
          "speed between 0 and 200")

    # ── Distance: 0–5000 m ────────────────────────────────────────────────
    check(gdf.expect_column_values_to_be_between(
              "distance_to_intersection", 0, 5000),
          "distance_to_intersection between 0 and 5000")

    # ── number_of_conflicts: 0–10 ─────────────────────────────────────────
    check(gdf.expect_column_values_to_be_between("number_of_conflicts", 0, 10),
          "number_of_conflicts between 0 and 10")

    # ── Direction: valid values only ──────────────────────────────────────
    check(gdf.expect_column_values_to_be_in_set(
              "direction", ["north", "south", "east", "west"]),
          "direction in {north, south, east, west}")

    # ── is_conflict: yes/no ───────────────────────────────────────────────
    check(gdf.expect_column_values_to_be_in_set(
              "is_conflict", ["yes", "no"]),
          "is_conflict in {yes, no}")

    # ── Row count ─────────────────────────────────────────────────────────
    check(gdf.expect_table_row_count_to_be_between(100, 100000),
          "row count between 100 and 100000")

    passed  = sum(1 for r in results if r["success"])
    failed  = [r for r in results if not r["success"]]
    total   = len(results)
    overall = len(failed) == 0

    print(f"\n  Result: {passed}/{total} passed  |  overall success: {overall}")
    return {"success": overall, "passed": passed, "total": total, "failures": failed}


def run_validation() -> dict:
    # ── Load and flatten ──────────────────────────────────────────────────
    from pipeline.ingest import parse_scenarios
    df_raw = pd.read_csv(RAW_CSV)
    df     = parse_scenarios(df_raw)

    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].copy()
    eval_df   = df.iloc[split_idx:].copy()

    print(f"Total vehicle rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # ── 1. Print statistics ───────────────────────────────────────────────
    print("\n=== DATA STATISTICS (Training Set) ===")
    print(train_df[["speed", "distance_to_intersection",
                     "lane", "number_of_conflicts"]].describe().to_string())

    print("\n=== COLUMN VALUE COUNTS ===")
    for col in ["direction", "destination", "is_conflict"]:
        print(f"\n{col}:\n{train_df[col].value_counts().to_string()}")

    # ── 2. Validate train and eval sets ──────────────────────────────────
    train_result = validate_dataframe(train_df, "Training Set")
    eval_result  = validate_dataframe(eval_df,  "Evaluation Set")

    # ── 3. Anomaly report ─────────────────────────────────────────────────
    print("\n=== ANOMALY REPORT (Evaluation Set) ===")
    if eval_result["failures"]:
        for f in eval_result["failures"]:
            print(f"  ⚠ ANOMALY  [{f['expectation']}]  {f['details']}")
    else:
        print("  ✅ No anomalies detected in evaluation set.")

    # ── 4. Save schema ────────────────────────────────────────────────────
    schema = {
        "columns":    list(df.columns),
        "dtypes":     {c: str(df[c].dtype) for c in df.columns},
        "row_count":  len(df),
        "train_rows": len(train_df),
        "eval_rows":  len(eval_df),
        "value_ranges": {
            "speed": {
                "min":  float(df["speed"].min()),
                "max":  float(df["speed"].max()),
                "mean": float(df["speed"].mean()),
            },
            "distance_to_intersection": {
                "min":  float(df["distance_to_intersection"].min()),
                "max":  float(df["distance_to_intersection"].max()),
                "mean": float(df["distance_to_intersection"].mean()),
            },
            "lane":                {"min": int(df["lane"].min()),
                                    "max": int(df["lane"].max())},
            "number_of_conflicts": {"min": int(df["number_of_conflicts"].min()),
                                    "max": int(df["number_of_conflicts"].max())},
        },
        "valid_directions":   sorted(df["direction"].unique().tolist()),
        "valid_destinations": sorted(df["destination"].unique().tolist()),
        "conflict_balance":   df["is_conflict"].value_counts().to_dict(),
    }

    with open(os.path.join(SCHEMA_DIR, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)
    print(f"\nSchema saved     → data/schema/schema.json")

    anomalies = eval_result["failures"]
    with open(os.path.join(SCHEMA_DIR, "anomalies.json"), "w") as f:
        json.dump(anomalies, f, indent=2)
    print(f"Anomaly report   → data/schema/anomalies.json")

    return schema


if __name__ == "__main__":
    run_validation()