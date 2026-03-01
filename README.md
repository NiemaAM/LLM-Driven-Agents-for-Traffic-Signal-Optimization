**🏷️ Tags:** `traffic-control` `intelligent-transportation` `large-language-model` `safety-detection` `human-in-the-loop` `smart-city`
<div align="center">
<h1>🚦 LLM-Driven Agents for Traffic Signal Optimization 🚗</h1>
CSC5382 – AI for Digital Transformation
</div>

# Table of content
[1. Milestone 1: Project Inception](#1-milestone-1-project-inception)
- [1.1. Framing the Business Idea as an ML Problem](#11-framing-the-business-idea-as-an-ml-problem)
  - [Business Case Description](#-buisness-case-description)
  - [Business Value of Using ML](#-business-value-of-using-ml)
  - [Data Overview](#data-overview)
  - [Project Archetype](#-project-archetype)
- [1.2. Feasibility Analysis](#12-feasibility-analysis)
  - [Literature Review](#-literature-review)
  - [Model Choice / Specification of a Baseline](#-model-choice-specification-of-a-baseline)
  - [Metrics for Business Goal Evaluation](#-metrics-for-business-goal-evaluation)

[2. Milestone 2: Baseline Proof-of-Concept (PoC)](#2-milestone-2-baseline-proof-of-concept-poc)
- [2.1. Overview](#21-overview)
- [2.2. App Structure](#22-app-structure)
- [2.3. Features](#23-features)
- [2.4. Intersection Layout](#24-intersection-layout)
- [2.5. Baseline Conflict Detection Logic](#25-baseline-conflict-detection-logic)
- [2.6. Speed-Aware Animation Model](#26-speed-aware-animation-model)
- [2.7. Running Locally](#27-running-locally)
- [2.8. Deployment](#28-deployment)
- [2.9. Example Scenario](#29-example-scenario)

[3. Milestone 3: Data Ingestion & Validation Pipeline](#3-milestone-3-data-ingestion--validation-pipeline)

[4. Milestone 4: Model Training & Experiment Tracking](#4-milestone-4-model-training--experiment-tracking)

[5. Milestone 5: Deployment & API Serving](#5-milestone-5-deployment--api-serving)

[6. Milestone 6: Monitoring & Continual Learning](#6-milestone-6-monitoring--continual-learning)

---

# 1. Milestone 1: Project Inception
> **Report:** [HERE](docs/trafficLLM-Milestone1.pdf)
> 
> **Notebook:** [HERE](LLM_Driven_Agents_for_Traffic_Signal_Optimization.ipynb)
> 
> **Video presentation:**

[![Watch the video](https://img.youtube.com/vi/Mm5viEheXXs/0.jpg)](https://youtu.be/Mm5viEheXXs)

---

## 1.1. Framing the Business Idea as an ML Problem  
### 📄 Buisness Case Description

Urban congestion is one of the most pressing challenges in modern cities, leading to increased travel times, fuel consumption, air pollution, and economic losses. Traditional traffic signal control systems rely on fixed timing plans or rule-based adaptive logic, which often fail to respond efficiently to dynamic traffic patterns.
This project proposes the development of LLM-driven agents for traffic signal optimization, where Large Language Models (LLMs) are used to generate, refine, and optimize traffic signal control policies. The system leverages traffic datasets and simulation environments to evaluate and iteratively improve signal timing strategies.

The LLM functions as an intelligent decision-support agent capable of:

- Generating traffic signal control logic.
- Interpreting traffic state representations.
- Proposing optimized signal phase transitions.
- Iteratively refining policies based on performance feedback.

The system is designed for urban planners, municipalities, and smart city operators seeking AI-assisted traffic optimization solutions.

### 🧠 Business value of using ML
Applying ML and LLM-based agents to traffic signal optimization provides measurable value:
 
**🚦 Operational Value**

- Smoother traffic flow
- Reduced congestion at intersections
- Shorter vehicle waiting times
- Increased road network throughput

**🌳 Environmental Value**

- Reduced CO₂ emissions
- Reduced idle engine time
- Improved urban air quality

**📈 Economic Value**

- Reduced fuel consumption
- Lower operational costs
- Reduced time lost due to congestion
- Improved public transportation reliability

**🏙️ Strategic Value**

- Scalable across cities
- Adaptable to changing traffic patterns
- Reduced dependency on manually engineered traffic rules
 
Compared to fixed-time or manually optimized systems, LLM-driven systems can adapt faster and propose alternative strategies automatically.

### Data Overview

**📌 Source:** https://doi.org/10.5281/zenodo.14171745

This dataset contains simulated multi-lane intersection traffic scenarios with annotated conflict events. It provides structured traffic state variables alongside labeled conflict occurrences and recommended control actions. The dataset is primarily used for conflict detection and safety-aware traffic signal decision modeling. 

#### Data Generation Process:

**📍 Synthetic Scenario Generation**

The data is generated using a [script](src/generate_data.py) that programmatically generates vehicle traffic scenarios for intersections. Each scenario consists of random or parameterized sets of vehicles with attributes like: `vehicle_id`, `lane`, `speed`, `distance_to_intersection`, `direction`, `destination`.

This simulated traffic data is created without real world measurements and is used as training/test input for conflict detection and classification.

**📍 Intersection Layout Specification**

The generated data uses a predefined intersection layout stored in [data/intersection_layout.json](data/intersection_layout.json). This defines:

  - Incoming directions (north, south, east, west)
  - Lane configurations
  - Valid destinations/exits

The generator uses this layout to assign realistic trajectories and entry/exit paths to vehicles within each scenario.

**📍 Random & Controlled Variation**

The generation script usually supports arguments to control:

  - Total number of records
  - Number of vehicles per scenario (fixed or variable)

This allows creation of large datasets covering diverse intersection traffic situations for conflict detection experiments.

**📍 Export to Disk**

After generation, the created dataset is exported to CSV format within [data/generate_data.py](data/generated_dataset.csv).

#### Data description:

- 🚗 Input Data Table (Vehicle Scenario Features):

| Feature                    | Description                      | Example  |
| -------------------------- | -------------------------------- | -------- |
| `vehicle_id`               | Unique vehicle identifier        | `V7657`  |
| `lane`                     | Lane number where the vehicle is | `6`      |
| `speed`                    | Vehicle speed (km/h or m/s)      | `62.36`  |
| `distance_to_intersection` | Distance to intersection (m)     | `319.51` |
| `direction`                | Travel direction                 | `south`  |
| `destination`              | Intended exit / destination      | `A`      |

- ⚠️ Labels Table (Conflict & Control Information):

| Label                 | Description                                 | Example                                                             |
| --------------------- | ------------------------------------------- | ------------------------------------------------------------------- |
| `is_conflict`         | Whether a traffic conflict occurs           | `yes`                                                               |
| `number_of_conflicts` | Number of conflicts in the scenario         | `1`                                                                 |
| `places_of_conflicts` | Locations of conflicts                      | `['intersection']`                                                  |
| `conflict_vehicles`   | Vehicle pairs involved in conflicts         | `[{'vehicle1_id': 'V7657', 'vehicle2_id': 'V4314'}]`                |
| `decisions`           | Recommended actions for vehicles            | `['Potential conflict: Vehicle V7657 must yield to Vehicle V4314']` |
| `priority_order`      | Vehicle priority ranking (1 = highest)      | `{'V4314': 1, 'V7657': 2, 'V5246': None, 'V2448': None}`            |
| `waiting_times`       | Vehicle waiting time (relative or absolute) | `{'V4314': 0, 'V7657': 2, 'V5246': 0, 'V2448': 0}`                  |

### 🔷 Project Archetype

This project follows the Human-in-the-Loop AI System archetype characterized by:

- The LLM generates signal control strategies.
- A simulation engine evaluates their performance.
- Engineers validate or refine outputs.
- Iterative feedback improves performance.

This positions the system as an AI-Orchestrated Decision Support System for Intelligent Transportation Infrastructure. The LLM does not directly control physical infrastructure without validation but operates within a supervised optimization loop.

---

## 1.2. Feasibility Analysis
### 📚 Literature review

Recent 2025 research demonstrates successful use of LLMs for traffic signal control:

- **Masri et al. (2025)** formalized LLMs as centralized traffic controllers using a 4D system (Detect, Decide, Disseminate, Deploy).  
- **Li et al. (2025)** introduced LLM-TrafficBrain, a semantic reasoning framework for dynamic signal control.  
- **Lai et al. (2025)** proposed LLMLight, directly employing LLMs as traffic signal control agents.  
- **Wang et al. (2025)** developed LLM-DCTSC for coordinated signal phase and duration optimization.

In a subsequent study, Masri et al. (2025) formalize the role of LLMs as centralized traffic controllers through a 4D system model (Detect, Decide, Disseminate, and Deploy) [1]. This methodology integrates traditionally disconnected control processes into a single LLM-driven architecture that can process heterogeneous data from GPS, video imaging, and loop detectors. The authors utilized fine-tuned models and ROUGE-L metrics to confirm that GPT-4o-mini excels in priority assignment and waiting time optimization. This new paradigm demonstrates that LLMs can provide precise, context-aware recommendations that align with established traffic regulations while enhancing overall intersection safety.

Li et al. present LLM-TrafficBrain, an information-centric framework designed for dynamic signal control through semantic reasoning [2]. This architecture transforms structured sensor data—including queue lengths and special events—into semantically rich natural language prompts for processing by an LLM. The framework operates in a closed-loop feedback system, allowing the model to self-correct and adjust timing strategies based on real-time performance metrics like vehicle throughput and average delay. Evaluation via the SUMO simulator showed that the system is highly responsive to emergency vehicle priority requests and unpredictable traffic spikes. 

Lai et al. introduce LLMLight, the first framework to directly employ LLMs as decision-making agents for TSC rather than just auxiliary tools [3]. The authors developed LightGPT, a specialized backbone LLM optimized through imitation fine-tuning and a critic-guided policy refinement process. This approach leverages Chain-of-Thought (CoT) reasoning to analyze traffic conditions and execute optimal signal phases. Extensive testing across ten datasets demonstrated that LightGPT offers superior generalization and interpretability compared to traditional heuristic and RL-based methods.

Wang et al. introduce LLM-DCTSC [4], an agent-driven framework that jointly optimizes signal phases and durations to improve traffic management granularity. By incorporating neighboring intersection data, the system avoids local optima and enhances global coordination. The framework employs a two-stage training pipeline featuring supervised fine-tuning and Direct Preference Optimization (DPO), guided by a reinforcement learning reward model. Utilizing Chain-of-Thought reasoning, LLM-DCTSC delivers interpretable decisions and achieves state-of-the-art performance in travel time and queue reduction across varied traffic conditions.

Full references are provided [Here](references/references.bib).

| Approach | Direct Agent Decision-Making | Chain-of-Thought (CoT) | Fine-Tuned Model | Actionable Driver Guidance | Emergency Vehicle Priority | Closed-Loop Feedback | RL Integration | Conflict Detection & Resolution | Natural Language Rationales |
|-----------|-----------------------------|-------------------------|------------------|----------------------------|---------------------------|---------------------|---------------|-------------------------------|----------------------------|
| Masri et al. (2025) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| LLM-TrafficBrain (2025) | ✓ | ◐ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ |
| LLMLight (2025) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ◐ | ✗ | ✓ |
| LLM-DCTSC (2025) | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ◐ | ✗ | ✓ |
 
<div align="center"> ✓ = Fully Supported  ◐ = Partially Supported  ✗ = Not Supported </div>

### 🤖 Model choice/ Specification of a baseline*

- **Model Name:** Intersection Conflict Detection LLM
- **Repository:** https://github.com/sarimasri3/Intersection-Conflict-Detection/ 
- **Reference Paper:** Masri et al. (2025) 
- Full description available in [Model Card](model/model_card.md).

**Arguments for Baseline Choice**

- The baseline uses existing repo code and synthetic data, meeting the requirement of having both the binary model and the notebook/code to retrain.
- Provides clear separation of conflict vs no-conflict, allowing future model improvements (e.g., deep learning, probabilistic forecasting) to be benchmarked.
- Leverages state-of-the-art simulation-based ML techniques from recent traffic research while remaining reproducible in Colab.

The fine-tuning process was performed through an API-based workflow, meaning the trained model is hosted by the provider and not stored locally as downloadable weights.

Because the model was fine-tuned via an external API, raw model weights are not exportable. However, we provide:

- The complete fine-tuning script
- The training dataset
- The evaluation pipeline
- Configuration parameters

This ensures full reproducibility of the experiment.

**Summary of Model Comparisons (from evaluation)**

| Model                           | Setting               | Best Result                   |
| ------------------------------- | --------------------- | ----------------------------- |
| **GPT-mini (fine-tuned)**       | Mixed vehicle dataset | ~83 % accuracy (best overall) |
| GPT-4o-mini (fine-tuned)        | Four vehicles         | ~81 % accuracy                |
| GPT-4o-mini (fine-tuned)        | Eight vehicles        | ~71 % accuracy                |
| GPT-mini (zero-shot)            | Mixed scenarios       | ~62 % accuracy                |
| Meta-LLaMA-3.1 variants         | Fine-tuned            | ~51 % accuracy (moderate)     |
| Gemini (fine-tuned & zero-shot) | Various               | Lower performance overall     |

In the context of this project and its evaluations, the fine-tuned GPT-mini model delivered the best conflict-detection performance among the tested LLMs.

### 📊 Metrics for business goal evaluation

Model evaluation must align with business objectives.

#### **⚠️ Safety-Oriented Metrics:**

Since intersection safety is a primary concern:

- **Precision:** Minimizes false conflict alarms
- **Recall:** Minimizes missed conflicts (critical for safety)
- **F1-Score:** Balanced safety-performance tradeoff
- **False Negative Rate (FNR):** Directly related to accident risk

Reducing false negatives is directly linked to preventing potential collisions, which aligns with the safety and liability reduction objectives of municipalities.

#### **🚘 Operational Efficiency Metrics:**

To measure congestion reduction:

- **Average Waiting Time (AWT):** The average amount of time vehicles spends stopped or delayed at an intersection before proceeding.
- **Average Travel Time (ATT):** The average total time it takes for a vehicle to pass through the intersection or road segment.
- **Average Queue Length (AQL):** The average number of vehicles waiting in line at an intersection during a given time period.
- **Intersection Throughput:** The total number of vehicles that successfully pass through an intersection within a specified time interval.

Reducing AWT and AQL directly supports:

- Reduced fuel consumption
- Reduced CO₂ emissions
- Improved commuter satisfaction

These metrics reflect measurable economic and environmental impact.

#### **💰 Cost-Sensitive Evaluation:**

Different errors have different consequences:

- **False negatives:** Safety risk
- **False positives:** Unnecessary traffic delays

A cost matrix will be used to assign higher penalties to safety-critical errors.

---
# 2. Milestone 2: Baseline Proof-of-Concept (PoC)

> **Live Demo:** [trafficllm.streamlit.app](https://trafficllm.streamlit.app)
> 
> **App source:** [`Streamlit/`](./Streamlit/)
>
> **Report:** [HERE](docs/trafficLLM-Milestone2.pdf)
>
> **Video Presentation:**

---

## 2.1. Overview

Milestone 2 delivers a fully interactive **Streamlit web application** that implements the baseline conflict detection system from [Masri et al. (2025)](https://arxiv.org/abs/2411.10869) as a visual Proof-of-Concept. The app allows users to build custom vehicle scenarios, run the rule-based conflict detection engine, and observe both the detected conflicts and their resolution through real-time animated intersection visualizations.

The PoC demonstrates the feasibility of the core pipeline:

```
Scenario Input → Conflict Detection → Priority Assignment → Wait Time Computation → Animated Visualization
```

---

## 2.2. App Structure

```
Streamlit/
├── app.py                  ← Main Streamlit entrypoint
├── requirements.txt        ← Python dependencies
src/
├── conflict_detection.py   ← Baseline conflict detection engine
├── visualization.py        ← Plotly animated intersection renderer
├── data_generation.py      ← Synthetic scenario generator
data/
├── intersection_layout.json ← Lane/direction/destination mapping
├── generated_dataset.csv   ← Generated training data (auto-created)
```

---

## 2.3. Features

### 🚗 Vehicle Scenario Builder
An interactive form replaces raw JSON input. Users build scenarios vehicle-by-vehicle using validated fields:

| Field | Input Type | Details |
|---|---|---|
| `vehicle_id` | Text | Auto-suggested (V001, V002…), must be unique |
| `lane` | Dropdown | Lanes 1–8, each showing its direction automatically |
| `direction` | Auto (read-only) | Derived from selected lane — no manual entry |
| `destination` | Dropdown | Populated dynamically from `intersection_layout.json` for the chosen lane |
| `speed` | Number input | km/h, 1–200 |
| `distance_to_intersection` | Number input | Meters, 1–5000 |

### 📋 Live JSON Sync
The vehicle list and a raw JSON panel are shown **side by side** and stay in sync in real time:
- Adding or removing a vehicle instantly updates the JSON panel
- Editing the JSON directly updates the vehicle list (with inline validation)
- The **Detect Conflicts** button always reads from the live JSON — direct JSON edits are always included

### 🚦 Intersection Visualization (Problem View)
An animated Plotly figure showing the scenario as entered, with no resolution applied:
- Vehicles approach from their correct lane positions at **physics-accurate speeds** — a vehicle doing 100 km/h over 100 m reaches the stop-line much sooner than one doing 30 km/h
- Lane numbers **1–8** are shown as badges on each road arm with directional arrows
- Conflicting vehicle pairs are connected by a **red dashed line** through the intersection centre
- Hover tooltips show vehicle ID, direction, lane, speed, TTA (time-to-intersection), and movement type

### ✅ Conflict Resolution Visualization (Solution View)
The same animation replayed with the wait times from `detect_conflicts()` applied:
- Lower-priority vehicles **pause at the stop-line** for exactly their assigned wait duration before crossing
- Priority-1 vehicles proceed immediately; the separation is clearly visible in the animation
- A summary table below shows each conflict pair, their priority order (🥇 / 🔴 yield), and wait times in seconds

### 📊 Conflict Detection Results
Raw output of `detect_conflicts()` displayed inline, showing for each conflict:
- Vehicle pair IDs
- Decision message (who yields to whom)
- Priority order dictionary
- Waiting times dictionary

### 📈 Dataset Generation
A sidebar button generates a 1,000-record synthetic dataset using `data_generation.py` and saves it to `data/generated_dataset.csv`, with a preview table in the sidebar.

---

## 2.4. Intersection Layout

The app uses a fixed 4-way intersection with 8 lanes and 8 exit destinations (A–H):

<img width="612" height="628" alt="image" src="https://github.com/user-attachments/assets/daf6baf3-376b-4a4e-b78d-c762fb4c01f6" />

| Direction | Lane | Type | Valid Destinations |
|-----------|------|------|--------------------|
| North | 1 | Entry (right/straight) | F, H |
| North | 2 | Exit ← | E, D, C |
| East | 3 | Exit ← | H, B |
| East | 4 | Entry (left) | G, E, F |
| South | 5 | Entry (right/straight) | B, D |
| South | 6 | Exit ← | A, G, H |
| West | 7 | Exit ← | D, F |
| West | 8 | Entry (left) | B, C, A |

---

## 2.5. Baseline Conflict Detection Logic

The baseline engine (`src/conflict_detection.py`) implements the rule-based system from Masri et al. (2025). For each pair of vehicles it:

1. **Checks if paths cross** — using movement types (straight / left / right) and direction rules (e.g. opposite straights don't conflict, adjacent left turns do)
2. **Checks arrival time proximity** — conflicts only trigger if both vehicles arrive within a 4-second threshold
3. **Applies priority rules** in order:
   - Straight over turn
   - Right turn over left turn
   - Right-hand rule (vehicle on the right has priority)
   - Later-arriving vehicle yields if time difference > 1s
4. **Computes waiting times** — the yielding vehicle waits until the priority vehicle has cleared the box (TTA + 2s crossing time)

Returns a `list[dict]` with keys: `vehicle1_id`, `vehicle2_id`, `decision`, `place`, `priority_order`, `waiting_times`.

---

## 2.6. Speed-Aware Animation Model

The animation timeline is computed from real vehicle physics, not fixed fractions:

```
total_sim_time = max over all vehicles of:
    time_to_intersection + wait_seconds + 4s (box crossing)

For each vehicle:
    approach_end = TTA / total_sim_time     ← reaches stop-line here
    wait_end     = (TTA + wait) / total_sim_time  ← starts crossing here
```

This means vehicles with different speeds and distances reach the stop-line at different points in the animation, accurately reflecting when conflicts would actually occur.

---

## 2.7. Running Locally

```bash
# Clone the repo
git clone https://github.com/NiemaAM/LLM-Driven-Agents-for-Traffic-Signal-Optimization.git
cd LLM-Driven-Agents-for-Traffic-Signal-Optimization

# Install dependencies
pip install -r Streamlit/requirements.txt

# Run the app
streamlit run Streamlit/app.py
```

The app will open at `http://localhost:8501`.

---

## 2.8. Deployment

The app is deployed on **Streamlit Community Cloud** and rebuilds automatically on every push to `main`.

To redeploy manually or set up a new deployment:
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Set **Main file path** to `Streamlit/app.py`
4. Set **Branch** to `main`

Dependencies are managed via `Streamlit/requirements.txt`.

---

## 2.9. Example Scenario

The app loads with this default two-vehicle conflict scenario:

https://github.com/user-attachments/assets/f610a451-dfbc-4764-b468-f01ae383ecfe

```json
{
  "vehicles_scenario": [
    {
      "vehicle_id": "V001",
      "lane": 1,
      "speed": 50,
      "distance_to_intersection": 100,
      "direction": "north",
      "destination": "F"
    },
    {
      "vehicle_id": "V002",
      "lane": 3,
      "speed": 50,
      "distance_to_intersection": 100,
      "direction": "east",
      "destination": "B"
    }
  ]
}
```

**Expected output:** V001 (north → straight) and V002 (east → straight) approach from perpendicular directions at the same speed and distance — a classic crossing conflict. The right-hand rule assigns priority to V002 (on the right of V001), so V001 must yield with a computed wait time.

https://github.com/user-attachments/assets/02d05a07-a455-4d5f-b525-beb84a9ee4b7

---

---

## 3. Milestone 3: Data ingestion & validation pipeline  

---

## 4. Milestone 4: Model training & experiment tracking  

---

## 5. Milestone 5: Deployment & API serving  

---

## 6. Milestone 6: Monitoring & continual learning  

---
