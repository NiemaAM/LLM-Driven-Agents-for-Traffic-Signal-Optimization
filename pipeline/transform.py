"""Step 3 – Feature engineering using scikit-learn on flattened vehicle data."""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

OUTPUT_DIR    = os.path.join("data", "processed")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def transform() -> pd.DataFrame:
    # ── Load flattened dataset produced by ingest.py ──────────────────────
    flat_csv = os.path.join(OUTPUT_DIR, "flattened_dataset.csv")
    if not os.path.exists(flat_csv):
        print("Flattened dataset not found — running ingest first ...")
        from pipeline.ingest import ingest
        ingest()

    df = pd.read_csv(flat_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} rows for transformation.")

    # ── 1. Encode label: yes → 1, no → 0 ─────────────────────────────────
    df["is_conflict_encoded"] = (
        df["is_conflict"].map({"yes": 1, "no": 0}).fillna(0).astype(int)
    )

    # ── 2. Z-score normalise continuous features ──────────────────────────
    scaler = StandardScaler()
    df[["speed_scaled", "distance_scaled"]] = scaler.fit_transform(
        df[["speed", "distance_to_intersection"]]
    )
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    print("  Saved scaler → data/processed/artifacts/scaler.pkl")

    # ── 3. Scale lane to [0, 1] over known range 1–8 ─────────────────────
    df["lane_scaled"] = (df["lane"].astype(float) - 1.0) / 7.0

    # ── 4. Encode categorical features ───────────────────────────────────
    for col in ["direction", "destination"]:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        joblib.dump(le, os.path.join(ARTIFACTS_DIR, f"encoder_{col}.pkl"))
        print(f"  {col} classes: {list(le.classes_)} → encoder_{col}.pkl")

    # ── 5. Derived feature: time_to_intersection (seconds) ────────────────
    speed_ms = df["speed"].astype(float) * 1000.0 / 3600.0
    df["time_to_intersection"] = (
        df["distance_to_intersection"].astype(float) /
        speed_ms.replace(0, np.nan)
    )

    # ── 6. Derived feature: vehicles per scenario ─────────────────────────
    df["vehicles_in_scenario"] = df.groupby("scenario_id")["scenario_id"].transform("count")

    # ── 7. Save transformed dataset ───────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, "transformed_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"\nTransformed dataset → {output_path}")
    print(f"Final columns: {list(df.columns)}")

    # ── 8. Print sample ───────────────────────────────────────────────────
    print("\nSample of engineered features:")
    print(df[[
        "vehicle_id", "speed", "speed_scaled",
        "distance_to_intersection", "distance_scaled",
        "lane", "lane_scaled",
        "direction", "direction_encoded",
        "destination", "destination_encoded",
        "time_to_intersection",
        "vehicles_in_scenario",
        "is_conflict", "is_conflict_encoded",
    ]].head(5).to_string())

    # ── 9. Save feature metadata ──────────────────────────────────────────
    metadata = {
        "numeric_features":     ["speed_scaled", "distance_scaled",
                                  "lane_scaled", "time_to_intersection"],
        "categorical_features": ["direction_encoded", "destination_encoded"],
        "derived_features":     ["time_to_intersection", "vehicles_in_scenario"],
        "label":                "is_conflict_encoded",
        "artifacts": {
            "scaler":             "artifacts/scaler.pkl",
            "encoder_direction":  "artifacts/encoder_direction.pkl",
            "encoder_destination":"artifacts/encoder_destination.pkl",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nFeature metadata → data/processed/feature_metadata.json")

    return df


if __name__ == "__main__":
    transform()