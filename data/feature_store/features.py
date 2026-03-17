from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

vehicle = Entity(
    name="vehicle_entity_id",
    join_keys=["vehicle_entity_id"],
    value_type=None,
)

source = FileSource(
    path="vehicle_features.parquet",
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
