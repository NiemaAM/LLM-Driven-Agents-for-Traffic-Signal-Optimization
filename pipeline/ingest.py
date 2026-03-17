"""Step 1 – Raw data ingestion: parse nested scenario JSON, flatten, split into train/eval."""

import os
import json
import numpy as np
import pandas as pd

RAW_CSV    = os.path.join("data", "raw", "generated_dataset.csv")
OUTPUT_DIR = os.path.join("data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row in the raw CSV has a 'scenario' column containing a JSON string
    with a list of vehicles. This function explodes each scenario into one row
    per vehicle, preserving the scenario-level labels.

    Input columns:
        scenario, is_conflict, number_of_conflicts, places_of_conflicts,
        conflict_vehicles, decisions, priority_order, waiting_times

    Output columns (one row per vehicle):
        scenario_id, vehicle_id, lane, speed, distance_to_intersection,
        direction, destination,
        is_conflict, number_of_conflicts, places_of_conflicts,
        conflict_vehicles, decisions, priority_order, waiting_times
    """
    rows = []
    for scenario_id, row in df.iterrows():
        try:
            scenario = json.loads(row["scenario"])
            vehicles = scenario.get("vehicles_scenario", [])
        except (json.JSONDecodeError, TypeError):
            continue

        for vehicle in vehicles:
            rows.append({
                "scenario_id":              scenario_id,
                "vehicle_id":               vehicle.get("vehicle_id", ""),
                "lane":                     int(vehicle.get("lane", 0)),
                "speed":                    float(vehicle.get("speed", 0.0)),
                "distance_to_intersection": float(vehicle.get("distance_to_intersection", 0.0)),
                "direction":                vehicle.get("direction", ""),
                "destination":              vehicle.get("destination", ""),
                "is_conflict":              row["is_conflict"],
                "number_of_conflicts":      int(row["number_of_conflicts"]),
                "places_of_conflicts":      str(row["places_of_conflicts"]),
                "conflict_vehicles":        str(row["conflict_vehicles"]),
                "decisions":                str(row["decisions"]),
                "priority_order":           str(row["priority_order"]),
                "waiting_times":            str(row["waiting_times"]),
            })

    return pd.DataFrame(rows)


def ingest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV, parse nested JSON, flatten to one row per vehicle,
    perform 80/20 train/eval split (split by scenario to avoid leakage),
    save both splits as parquet.
    """
    print(f"Loading {RAW_CSV} ...")
    df_raw = pd.read_csv(RAW_CSV)
    print(f"  Raw rows (scenarios): {len(df_raw)}")

    print("Parsing and flattening vehicle records ...")
    df = parse_scenarios(df_raw)
    print(f"  Flattened rows (vehicles): {len(df)}")

    # Split by scenario_id to avoid data leakage between train and eval
    unique_scenarios = df["scenario_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_scenarios)

    split_idx = int(len(unique_scenarios) * 0.8)
    train_ids = set(unique_scenarios[:split_idx])
    eval_ids  = set(unique_scenarios[split_idx:])

    train_df = df[df["scenario_id"].isin(train_ids)].copy()
    eval_df  = df[df["scenario_id"].isin(eval_ids)].copy()

    train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
    eval_df.to_parquet(os.path.join(OUTPUT_DIR,  "eval.parquet"),  index=False)

    df.to_csv(os.path.join(OUTPUT_DIR, "flattened_dataset.csv"), index=False)

    print(f"\nTrain: {len(train_df)} vehicle rows → data/processed/train.parquet")
    print(f"Eval:  {len(eval_df)}  vehicle rows → data/processed/eval.parquet")
    print(f"Full:  {len(df)}      vehicle rows → data/processed/flattened_dataset.csv")

    return train_df, eval_df


if __name__ == "__main__":
    train_df, eval_df = ingest()
    print("\nSample:")
    print(train_df[["vehicle_id", "lane", "speed", "distance_to_intersection",
                     "direction", "destination", "is_conflict"]].head(5).to_string())