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

st.set_page_config(page_title="Intersection Conflict Detection")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LAYOUT_PATH = os.path.join(DATA_DIR, "intersection_layout.json")
DATASET_PATH = os.path.join(DATA_DIR, "generated_dataset.csv")


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


def main() -> None:
    st.title("Intersection Conflict Detection")

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
        st.sidebar.write("No dataset found. Generate one using the button above.")

    # Load intersection layout
    layout = load_layout(LAYOUT_PATH)
    if layout is None:
        return

    # Scenario input
    default_example = {
        "vehicles_scenario": [
            {"vehicle_id": "V001", "lane": 1, "speed": 50, "distance_to_intersection": 100, "direction": "north", "destination": "F"},
            {"vehicle_id": "V002", "lane": 3, "speed": 50, "distance_to_intersection": 100, "direction": "east", "destination": "B"},
        ]
    }

    scenario_text = st.text_area("Paste vehicles scenario JSON", value=json.dumps(default_example, indent=2), height=260)

    if st.button("Detect Conflicts"):

        try:
            scenario_data = json.loads(scenario_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return

        # run detection inside a warnings catcher so that model-generated
        # warnings (UserWarning) are surfaced to the UI
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                vehicles = parse_vehicles(scenario_data, layout)
                conflicts = detect_conflicts(vehicles)
            except Exception as e:
                st.warning(f"Failed to compute conflicts: {e}")
                return

        # display any warnings produced by the logic
        for w in caught:
            st.warning(str(w.message))
        
        # problem Visualization (animated with Plotly)
        try:
            from src.visualization import visualize_intersection
            st.subheader("Intersection Visualization")
            visualize_intersection(layout, vehicles, steps=30, interval=100)
        except Exception as e:
            st.warning(f"Visualization failed: {e}")

        st.subheader("Conflict Detection Results")
        st.write(conflicts)

        # solution visualization (animated with Plotly)
        try:
            from src.visualization import visualize_solution
            st.subheader("Conflict Resolution Visualization")
            visualize_solution(layout, vehicles, conflicts, steps=30, interval=100)
        except Exception as e:
            st.warning(f"Solution visualization failed: {e}")


if __name__ == "__main__":
    main()
