# LLM-Driven Agents for Traffic Signal Optimization  
CSC5382 – AI for Digital Transformation  
Milestone 1 – Project Inception  

---

## 1. Project Overview  

Urban congestion remains a critical challenge in modern cities, causing increased travel times, fuel consumption, environmental pollution, and economic losses. Traditional traffic signal control systems rely on fixed-time schedules or manually engineered adaptive logic, which struggle to respond to dynamic and unpredictable traffic conditions.

This project proposes an **LLM-driven traffic signal optimization system**, where Large Language Models (LLMs) act as intelligent agents that:

- Interpret traffic state descriptions  
- Detect potential vehicle conflicts  
- Propose optimized signal phase transitions  
- Iteratively refine policies based on simulation feedback  

The system is designed as a **Human-in-the-Loop AI Decision Support System** for urban planners and smart city operators.

---

## 2. Business Case  

### Problem  
Urban intersections suffer from congestion and safety risks due to static or rule-based traffic signal systems.

### Target Stakeholders  
- Municipal traffic authorities  
- Smart city operators  
- Urban infrastructure planners  

### Proposed Solution  
Deploy an LLM-driven intelligent agent capable of generating and refining traffic signal control strategies using structured traffic data and simulation environments.

---

## 3. Business Value of Using ML  

### Operational Value  
- Reduced congestion  
- Shorter vehicle waiting times  
- Increased intersection throughput  

### Environmental Value  
- Reduced CO₂ emissions  
- Reduced idle engine time  
- Improved air quality  

### Economic Value  
- Lower fuel consumption  
- Reduced operational costs  
- Improved transport efficiency  

### Strategic Value  
- Scalable across cities  
- Adaptable to evolving traffic patterns  
- Reduced dependency on manual rule engineering  

---

## 4. Dataset  

**Source:** https://doi.org/10.5281/zenodo.14171745  

The dataset contains simulated multi-lane intersection traffic scenarios with annotated conflict events.

### Input Features  

- vehicle_id  
- lane  
- speed  
- distance_to_intersection  
- direction  
- destination  

### Labels  

- is_conflict  
- conflict_vehicles  
- priority_order  
- waiting_times  
- decisions  

The dataset supports safety-aware traffic signal modeling and conflict detection tasks.

---

## 5. Project Archetype  

This project follows the **Human-in-the-Loop AI System** archetype:

1. LLM generates signal control strategies  
2. Simulation engine evaluates performance  
3. Engineers validate results  
4. Iterative refinement improves system performance  

The system does **not directly control physical infrastructure** without validation.

---

## 6. Literature Review Summary  

Recent 2025 research demonstrates successful use of LLMs for traffic signal control:

- **Masri et al. (2025)** – 4D LLM traffic control architecture  
- **LLM-TrafficBrain** – Semantic prompt-based dynamic signal control  
- **LLMLight** – LLM as direct traffic signal control agent  
- **LLM-DCTSC** – Coordinated multi-intersection optimization with DPO  

These works demonstrate:

- Chain-of-Thought reasoning improves interpretability  
- Fine-tuned LLMs outperform heuristic methods  
- Closed-loop systems improve congestion metrics  

---

## 7. Baseline Model  

**Model Name:** Intersection Conflict Detection LLM  
**Repository:** https://github.com/sarimasri3/Intersection-Conflict-Detection/  

### Baseline Characteristics  

- Base Model: GPT-family lightweight variant  
- Training: Supervised fine-tuning  
- Task: Conflict detection + reasoning generation  
- Output: Conflict status + recommended action  

### Intended Use  

- Intersection-level safety detection  
- Simulation-based optimization experiments  

### Limitations  

- Trained on simulated data  
- Limited network-level coordination  
- May degrade under unseen distributions  

---

## 8. Evaluation Metrics  

### Safety Metrics  
- Precision  
- Recall  
- F1-Score  
- False Negative Rate (FNR)  

### Operational Metrics  
- Average Waiting Time (AWT)  
- Average Travel Time (ATT)  
- Average Queue Length (AQL)  
- Intersection Throughput  

### Cost-Sensitive Evaluation  
Higher penalty assigned to safety-critical false negatives.

These metrics directly map to business KPIs such as accident reduction, congestion mitigation, and emission reduction.

---

## 9. Repository Structure  
LLM-Traffic-Signal-Optimization/
│
├── README.md
│
├── docs/
│ ├── milestone1_report.pdf
│ ├── presentation_slides.pdf
│
├── data/
│ ├── raw/
│ ├── processed/
│
├── baseline/
│ ├── baseline_description.md
│
├── notebooks/
│ ├── exploratory_analysis.ipynb
│
├── references/
│ ├── bibliography.bib
│
└── video_presentation_link.txt


---

## 10. Next Milestones  

- Milestone 2: Baseline Proof-of-Concept (PoC)  
- Milestone 3: Data ingestion & validation pipeline  
- Milestone 4: Model training & experiment tracking  
- Milestone 5: Deployment & API serving  
- Milestone 6: Monitoring & continual learning  

---

## 11. References  

Full reference list available in `/references` and in milestone report.

