"""Streamlit app for Intersection Conflict Detection.

Entrypoint that ties together data generation, scenario input, conflict
detection and visualization. Designed to be run with ``streamlit run app.py``.
"""

import json
import os
from typing import Optional

import pandas as pd
import streamlit as st

from src.conflict_detection import (
    parse_vehicles,
    detect_conflicts,
    parse_intersection_layout,
)

st.set_page_config(page_title="Intersection Conflict Detection", layout="wide")

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
LAYOUT_PATH  = os.path.join(DATA_DIR, "intersection_layout.json")
DATASET_PATH = os.path.join(DATA_DIR, "generated_dataset.csv")

LANE_DIRECTION = {
    1: "north", 2: "north",
    3: "east",  4: "east",
    5: "south", 6: "south",
    7: "west",  8: "west",
}

_DEFAULT_VEHICLES = [
    {"vehicle_id": "V001", "lane": 1, "speed": 50,
     "distance_to_intersection": 100, "direction": "north", "destination": "F"},
    {"vehicle_id": "V002", "lane": 3, "speed": 50,
     "distance_to_intersection": 100, "direction": "east",  "destination": "B"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_layout(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return parse_intersection_layout(json.load(f))
    except FileNotFoundError:
        st.error(f"Intersection layout not found at {path}")
    except Exception as e:
        st.error(f"Unable to load intersection layout: {e}")
    return None


def load_dataset(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Failed to read dataset: {e}")
    return None


def get_destinations_for_lane(layout: dict, lane: int) -> list[str]:
    direction = LANE_DIRECTION.get(lane)
    if direction is None:
        return []
    return layout.get(direction, {}).get(str(lane), [])


def _vehicles_to_json(vehicles: list) -> str:
    return json.dumps({"vehicles_scenario": vehicles}, indent=2)


def _parse_json_safe(text: str) -> tuple[list | None, str | None]:
    """Parse JSON text → (vehicles list, error message).  Returns (None, err) on failure."""
    try:
        data = json.loads(text)
        if "vehicles_scenario" not in data:
            return None, "JSON must have a 'vehicles_scenario' key."
        if not isinstance(data["vehicles_scenario"], list):
            return None, "'vehicles_scenario' must be a list."
        return data["vehicles_scenario"], None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "vehicles" not in st.session_state:
        st.session_state.vehicles = list(_DEFAULT_VEHICLES)
    # The live JSON string is always recomputed from vehicles on each render;
    # we store it so the textarea key stays stable across reruns.
    if "json_text" not in st.session_state:
        st.session_state.json_text = _vehicles_to_json(st.session_state.vehicles)


def _sync_json_from_vehicles() -> None:
    """Overwrite the JSON textarea content from the current vehicle list."""
    st.session_state.json_text = _vehicles_to_json(st.session_state.vehicles)


def _on_json_edit() -> None:
    """Called by on_change on the JSON textarea.
    Parses the current text and updates st.session_state.vehicles if valid."""
    text = st.session_state.json_text          # updated by Streamlit before callback
    vehicles, err = _parse_json_safe(text)
    if vehicles is not None:
        st.session_state.vehicles = vehicles   # keep UI list in sync
        st.session_state._json_error = None
    else:
        st.session_state._json_error = err     # show inline error, keep old vehicles


# ---------------------------------------------------------------------------
# Vehicle builder UI
# ---------------------------------------------------------------------------

def vehicle_builder(layout: dict) -> None:
    st.subheader("🚗 Vehicle Scenario Builder")

    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: form + vehicle list ──────────────────────────────────────────
    with left:
        with st.expander("➕ Add a vehicle", expanded=True):
            c1, c2, c3 = st.columns(3)

            with c1:
                new_id = st.text_input(
                    "Vehicle ID",
                    value=f"V{len(st.session_state.vehicles) + 1:03d}",
                    key="form_id",
                )
                new_speed = st.number_input(
                    "Speed (km/h)", min_value=1, max_value=200,
                    value=50, step=5, key="form_speed",
                )

            with c2:
                new_lane = st.selectbox(
                    "Lane",
                    options=list(range(1, 9)),
                    format_func=lambda l: f"Lane {l}  ({LANE_DIRECTION[l]})",
                    key="form_lane",
                )
                new_dist = st.number_input(
                    "Distance to intersection (m)", min_value=1, max_value=5000,
                    value=100, step=10, key="form_dist",
                )

            with c3:
                auto_direction = LANE_DIRECTION[new_lane]
                st.text_input(
                    "Direction (auto from lane)",
                    value=auto_direction,
                    disabled=True,
                    key="form_dir",
                )
                dest_options = get_destinations_for_lane(layout, new_lane)
                new_dest = st.selectbox(
                    "Destination",
                    options=dest_options if dest_options else ["—"],
                    key="form_dest",
                )

            if st.button("Add Vehicle", type="primary", use_container_width=True):
                existing_ids = [v["vehicle_id"] for v in st.session_state.vehicles]
                if not new_id.strip():
                    st.error("Vehicle ID cannot be empty.")
                elif new_id.strip() in existing_ids:
                    st.error(f"Vehicle ID '{new_id}' already exists.")
                elif new_dest == "—":
                    st.error("No valid destination for this lane.")
                else:
                    st.session_state.vehicles.append({
                        "vehicle_id":               new_id.strip(),
                        "lane":                     new_lane,
                        "speed":                    new_speed,
                        "distance_to_intersection": new_dist,
                        "direction":                auto_direction,
                        "destination":              new_dest,
                    })
                    _sync_json_from_vehicles()   # ← push to JSON panel immediately
                    st.rerun()

        # Vehicle list
        if st.session_state.vehicles:
            st.markdown("#### 🗂 Current vehicles")
            for idx, v in enumerate(st.session_state.vehicles):
                cols = st.columns([1.2, 1.4, 1, 1.6, 0.8, 0.5])
                cols[0].markdown(f"**{v['vehicle_id']}**")
                cols[1].caption(f"Lane {v['lane']} · {v['direction'].capitalize()}")
                cols[2].caption(f"{v['speed']} km/h")
                cols[3].caption(f"{v['distance_to_intersection']} m")
                cols[4].caption(f"→ **{v['destination']}**")
                if cols[5].button("🗑", key=f"del_{idx}", help=f"Remove {v['vehicle_id']}"):
                    st.session_state.vehicles.pop(idx)
                    _sync_json_from_vehicles()   # ← push to JSON panel immediately
                    st.rerun()

            if st.button("🗑 Clear all", use_container_width=True):
                st.session_state.vehicles = []
                _sync_json_from_vehicles()
                st.rerun()
        else:
            st.info("No vehicles yet. Use the form above to add vehicles.")

    # ── RIGHT: live JSON panel ─────────────────────────────────────────────
    with right:
        st.markdown("#### 📋 Live JSON scenario")
        st.caption(
            "This JSON updates in real time as you add/remove vehicles. "
            "You can also edit it directly — valid changes sync back to the vehicle list."
        )

        st.text_area(
            label="vehicles_scenario JSON",
            key="json_text",                 # bound directly to session state
            height=400,
            on_change=_on_json_edit,         # parse & sync on every keystroke change
            label_visibility="collapsed",
        )

        # Show inline parse error if JSON is currently invalid
        if st.session_state.get("_json_error"):
            st.error(f"⚠ {st.session_state._json_error}  — fix the JSON or use the form.")
        else:
            st.success("✅ JSON is valid and synced.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Intersection Conflict Detection")

    _init_state()

    # Sidebar: dataset controls
    st.sidebar.header("Dataset")
    if st.sidebar.button("Generate sample dataset (1000)"):
        from src.data_generation import generate_dataset
        ds = generate_dataset(total_records=1000, num_vehicles=5, fixed_vehicle_count=False)
        os.makedirs(DATA_DIR, exist_ok=True)
        ds.to_csv(DATASET_PATH, index=False)
        st.sidebar.success(f"Dataset written to {DATASET_PATH}")

    dataset = load_dataset(DATASET_PATH)
    if dataset is not None:
        st.sidebar.write("Loaded dataset preview:")
        st.sidebar.dataframe(dataset.head())
    else:
        st.sidebar.write("No dataset found. Generate one above.")

    layout = load_layout(LAYOUT_PATH)
    if layout is None:
        return

    # Vehicle builder (form + live JSON side by side)
    vehicle_builder(layout)

    st.divider()

    # Detect conflicts button — always reads from the live JSON textarea
    if not st.session_state.get("vehicles"):
        st.info("Add at least one vehicle to run conflict detection.")
        return

    if st.button("🔍 Detect Conflicts", type="primary", use_container_width=True):

        # Always parse from the JSON textarea so any direct edits are included
        vehicles_raw, err = _parse_json_safe(st.session_state.json_text)
        if err:
            st.error(f"Cannot run: {err}")
            return

        scenario_data = {"vehicles_scenario": vehicles_raw}

        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                vehicles = parse_vehicles(scenario_data, layout)
                conflicts = detect_conflicts(vehicles)
            except Exception as e:
                st.warning(f"Failed to compute conflicts: {e}")
                return

        for w in caught:
            st.warning(str(w.message))

        try:
            from src.visualization import visualize_intersection
            st.subheader("Intersection Visualization")
            visualize_intersection(layout, vehicles, steps=40, interval=80)
        except Exception as e:
            st.warning(f"Visualization failed: {e}")

        st.subheader("Conflict Detection Results")
        st.write(conflicts)

        try:
            from src.visualization import visualize_solution
            st.subheader("Conflict Resolution Visualization")
            visualize_solution(layout, vehicles, conflicts, steps=50, interval=80)
        except Exception as e:
            st.warning(f"Solution visualization failed: {e}")


if __name__ == "__main__":
    main()