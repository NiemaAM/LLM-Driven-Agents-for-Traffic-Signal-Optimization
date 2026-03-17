"""Step 4 – Feature store using Feast (local mode)."""

import os
import importlib.util
import pandas as pd
from datetime import datetime, timezone

FEATURE_STORE_DIR = os.path.abspath(os.path.join("data", "feature_store"))
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

TRANSFORMED_CSV = os.path.join("data", "processed", "transformed_dataset.csv")


def prepare_parquet() -> str:
    """Convert transformed dataset to parquet for Feast offline source."""
    df = pd.read_csv(TRANSFORMED_CSV)
    df["vehicle_entity_id"] = range(len(df))
    df["event_timestamp"]   = datetime.now(tz=timezone.utc)

    feast_cols = [
        "vehicle_entity_id", "event_timestamp",
        "speed", "distance_to_intersection", "lane",
        "speed_scaled", "distance_scaled", "lane_scaled",
        "time_to_intersection", "vehicles_in_scenario",
        "direction_encoded", "destination_encoded",
        "is_conflict_encoded",
    ]
    df_feast = df[feast_cols].copy()

    df_feast["speed"]                    = df_feast["speed"].astype("float32")
    df_feast["distance_to_intersection"] = df_feast["distance_to_intersection"].astype("float32")
    df_feast["speed_scaled"]             = df_feast["speed_scaled"].astype("float32")
    df_feast["distance_scaled"]          = df_feast["distance_scaled"].astype("float32")
    df_feast["lane_scaled"]              = df_feast["lane_scaled"].astype("float32")
    df_feast["time_to_intersection"]     = df_feast["time_to_intersection"].astype("float32")
    df_feast["vehicles_in_scenario"]     = df_feast["vehicles_in_scenario"].astype("float32")
    df_feast["lane"]                     = df_feast["lane"].astype("int64")
    df_feast["direction_encoded"]        = df_feast["direction_encoded"].astype("int64")
    df_feast["destination_encoded"]      = df_feast["destination_encoded"].astype("int64")
    df_feast["is_conflict_encoded"]      = df_feast["is_conflict_encoded"].astype("int64")
    df_feast["vehicle_entity_id"]        = df_feast["vehicle_entity_id"].astype("int64")

    parquet_path = os.path.join(FEATURE_STORE_DIR, "vehicle_features.parquet")
    df_feast.to_parquet(parquet_path, index=False)
    print(f"Feature parquet written → {parquet_path}  ({len(df_feast)} rows)")
    return parquet_path


def write_feature_repo(parquet_path: str) -> None:
    """
    Write feature_store.yaml using relative paths only.
    Feast on Windows rejects absolute paths starting with 'C:/' as
    it mistakes the drive letter for a URL scheme.
    All paths in the yaml must be relative to FEATURE_STORE_DIR.
    """
    # Relative path from FEATURE_STORE_DIR to the parquet file
    parquet_rel = os.path.relpath(parquet_path, FEATURE_STORE_DIR).replace("\\", "/")

    yaml_content = f"""project: traffic_conflict_detection
registry: registry.db
provider: local
online_store:
    type: sqlite
    path: online_store.db
"""
    yaml_path = os.path.join(FEATURE_STORE_DIR, "feature_store.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"feature_store.yaml → {yaml_path}")

    features_py = f'''from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

vehicle = Entity(
    name="vehicle_entity_id",
    join_keys=["vehicle_entity_id"],
    value_type=None,
)

source = FileSource(
    path="{parquet_rel}",
    timestamp_field="event_timestamp",
)

vehicle_features = FeatureView(
    name="vehicle_features",
    entities=[vehicle],
    ttl=timedelta(days=365),
    schema=[
        Field(name="speed",                    dtype=Float32),
        Field(name="distance_to_intersection", dtype=Float32),
        Field(name="lane",                     dtype=Int64),
        Field(name="speed_scaled",             dtype=Float32),
        Field(name="distance_scaled",          dtype=Float32),
        Field(name="lane_scaled",              dtype=Float32),
        Field(name="time_to_intersection",     dtype=Float32),
        Field(name="vehicles_in_scenario",     dtype=Float32),
        Field(name="direction_encoded",        dtype=Int64),
        Field(name="destination_encoded",      dtype=Int64),
        Field(name="is_conflict_encoded",      dtype=Int64),
    ],
    source=source,
)
'''
    features_path = os.path.join(FEATURE_STORE_DIR, "features.py")
    with open(features_path, "w") as f:
        f.write(features_py)
    print(f"features.py        → {features_path}")


def apply_and_materialize() -> None:
    """Apply feature definitions and materialise into the online store."""
    from feast import FeatureStore

    # FeatureStore must be initialised from FEATURE_STORE_DIR
    # so that relative paths in feature_store.yaml resolve correctly
    store = FeatureStore(repo_path=FEATURE_STORE_DIR)

    # Load entity and feature view from features.py
    spec = importlib.util.spec_from_file_location(
        "features", os.path.join(FEATURE_STORE_DIR, "features.py")
    )
    features_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features_mod)

    store.apply([features_mod.vehicle, features_mod.vehicle_features])
    print("Feast apply complete.")

    store.materialize_incremental(end_date=datetime.now(tz=timezone.utc))
    print("Features materialised into online store.")

    # Quick retrieval test
    entity_df = pd.DataFrame({
        "vehicle_entity_id": [0, 1, 2],
        "event_timestamp":   [datetime.now(tz=timezone.utc)] * 3,
    })
    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "vehicle_features:speed_scaled",
            "vehicle_features:is_conflict_encoded",
        ],
    ).to_df()
    print("\nFeature retrieval test (3 rows):")
    print(features.to_string())


if __name__ == "__main__":
    parquet_path = prepare_parquet()
    write_feature_repo(parquet_path)
    apply_and_materialize()