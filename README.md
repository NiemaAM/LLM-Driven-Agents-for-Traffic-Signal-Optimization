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

[3. Milestone 3: Data Ingestion & Validation Pipeline](#3-milestone-3-data-ingestion--validation-pipeline)

[4. Milestone 4: Model Training & Experiment Tracking](#4-milestone-4-model-training--experiment-tracking)

[5. Milestone 5: Deployment & API Serving](#5-milestone-5-deployment--api-serving)

[6. Milestone 6: Monitoring & Continual Learning](#6-milestone-6-monitoring--continual-learning)

---

# 1. Milestone 1: Project Inception
**Report:** [HERE](docs/milestone1_report.pdf)

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

- [data](data/generated_dataset.csv)
- [intersection layout](data/intersection_layout.json)

**Data description:**

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

Full references are provided [Here](references/references.bib).

| Approach | Direct Agent Decision-Making | Chain-of-Thought (CoT) | Fine-Tuned Model | Actionable Driver Guidance | Emergency Vehicle Priority | Closed-Loop Feedback | RL Integration | Conflict Detection & Resolution | Natural Language Rationales |
|-----------|-----------------------------|-------------------------|------------------|----------------------------|---------------------------|---------------------|---------------|-------------------------------|----------------------------|
| Masri et al. (2025) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| LLM-TrafficBrain (2025) | ✓ | ◐ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ |
| LLMLight (2025) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ◐ | ✗ | ✓ |
| LLM-DCTSC (2025) | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ◐ | ✗ | ✓ |
 
<div align="center"> ✓ = Fully Supported  ◐ = Partially Supported  ✗ = Not Supported </div>

### 🤖 Model choice/ Specification of a baseline

- **Model Name:** Intersection Conflict Detection LLM
- **Repository:** https://github.com/sarimasri3/Intersection-Conflict-Detection/ 
- **Reference Paper:** Masri et al. (2025) 
- Full description available in [Model Card](model/model_card.md).

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

## 2. Milestone 2: Baseline Proof-of-Concept (PoC)  

---

## 3. Milestone 3: Data ingestion & validation pipeline  

---

## 4. Milestone 4: Model training & experiment tracking  

---

## 5. Milestone 5: Deployment & API serving  

---

## 6. Milestone 6: Monitoring & continual learning  

---
