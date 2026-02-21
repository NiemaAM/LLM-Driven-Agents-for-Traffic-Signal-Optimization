# Model Card for Intersection Conflict Detection LLM

The Intersection Conflict Detection LLM is a fine-tuned Large Language Model designed to detect vehicle conflicts at urban intersections and generate structured traffic control recommendations for simulation-based intelligent transportation systems.

## Model Details

### Model Description

This model is a supervised fine-tuned GPT-family language model specialized for traffic conflict detection and structured reasoning at urban intersections. It processes structured vehicle state descriptions and outputs conflict classification, reasoning, and recommended mitigation actions. The model is intended for research, simulation experiments, and human-in-the-loop traffic signal optimization workflows.

- **Developed by:** Masri et al., 2025
- **Model type:** Fine-tuned Large Language Model (Transformer-based)
- **Language(s) (NLP):** English
- **License:** MIT
- **Finetuned from model:** GPT-family lightweight variant

### Model Sources

- **Repository:** https://github.com/sarimasri3/Intersection-Conflict-Detection/
- **Paper:** https://doi.org/10.3390/vehicles7010011

## Uses

### Direct Use

The model can be directly used to:
- Detect conflicts between vehicles at urban intersections
- Generate structured safety-aware recommendations
- Support simulation-based traffic control experiments
- Assist researchers in human-in-the-loop AI workflows

### Downstream Use

The model may be integrated into:
- Traffic simulators (e.g., SUMO-based systems)
- Smart city traffic analytics dashboards
- AI-driven decision-support systems for transportation research
- Larger LLM-driven signal optimization pipelines

### Out-of-Scope Use

- Autonomous real-world traffic signal control without human validation
- Safety-critical infrastructure deployment without testing
- Real-time infrastructure control without supervision
- Network-wide traffic optimization without coordination modeling

## Bias, Risks, and Limitations

- Trained exclusively on simulated traffic scenarios
- May not generalize to real-world noisy sensor data
- Limited to intersection-level reasoning
- False negatives (missed conflicts) pose safety risks
- Model outputs may appear confident even under uncertainty

### Recommendations

Users (both direct and downstream) should:
- Employ the model strictly within a human-in-the-loop framework
- Validate outputs in controlled simulation environments
- Closely monitor false negative rates
- Avoid autonomous deployment in live infrastructure

## Model Details

**Model Name:** Intersection Conflict Detection LLM  
**Model Type:** Supervised Fine-Tuned Large Language Model  
**Base Architecture:** GPT-family lightweight variant  
**Task:** Traffic Conflict Detection + Structured Reasoning  
**Domain:** Intelligent Transportation Systems (ITS)  
**Reference Paper:** Masri et al., 2025  

This model is based on the framework proposed in:

Masri, S., Ashqar, H. I., & Elhenawy, M. (2025).  
*Large Language Models (LLMs) as Traffic Control Systems at Urban Intersections: A New Paradigm.*  
Published in *Vehicles*.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "your-username/intersection-conflict-detection-llm"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

scenario = """
Vehicles:
- id: V1, lane: north, speed: 20, distance: 15m
- id: V2, lane: east, speed: 15, distance: 10m
"""

inputs = tokenizer(scenario, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
## Training Details

### Training Data

The model was fine-tuned on simulated intersection traffic scenarios containing:

- Structured vehicle state variables
- Annotated conflict events
- Priority rankings
- Recommended control decisions

Dataset source: https://doi.org/10.5281/zenodo.14171745

The dataset represents synthetic multi-lane urban intersection scenarios and may not fully generalize to real-world noisy sensor data.

### Training Procedure

- Training Method: Supervised fine-tuning
- Objective: Binary classification + structured reasoning generation
- Input Format: Natural language structured vehicle scenario description
- Output Format: Conflict status + explanation + control recommendation

Hardware requirements:
- GPU-enabled environment (e.g., cloud VM or Google Colab)

#### Preprocessing

- Structured vehicle state data converted into natural language prompts
- Conflict annotations formatted into structured JSON-like outputs
- Basic filtering of incomplete or inconsistent scenarios

#### Training Hyperparameters

- **Training regime:** Training regime: fp16 mixed precision

#### Speeds, Sizes, Times

- GPU-based training (cloud VM or Google Colab)
- Lightweight model variant for efficient experimentation
- Estimated training duration: 5–10 GPU hours

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Held-out simulated intersection scenarios from the same dataset distribution.

#### Factors

- Traffic density variations
- Vehicle speed differences
- Conflict vs non-conflict distributions

#### Metrics

- Precision
- Recall
- F1-Score
- False Negative Rate (FNR)

FNR is emphasized due to safety-critical implications.

### Results

The model demonstrates strong conflict detection performance in simulated environments with balanced precision and recall while maintaining controlled false negative rates.

#### Summary

The model performs reliably within synthetic traffic scenarios but requires further validation for real-world deployment.

## Model Examination

Interpretability is supported through structured natural language reasoning outputs that explain conflict decisions and recommended actions.

## Environmental Impact

Carbon emissions can be estimated using the Machine Learning Impact calculator.

- Hardware Type: GPU (e.g., NVIDIA T4 or similar)
- Hours used: Approximately 5–10 hours
- Cloud Provider: Google Colab / Cloud VM
- Compute Region: Not specified
- Carbon Emitted: Not formally measured

## Technical Specifications

### Model Architecture and Objective

- Transformer-based GPT-family architecture
- Decoder-style language model
- Supervised fine-tuning objective minimizing cross-entropy loss
- Outputs structured reasoning and classification tokens

### Compute Infrastructure

GPU-enabled environment required for training; CPU sufficient for inference.

#### Hardware

- NVIDIA T4 (or equivalent GPU)
- Standard CPU for inference

#### Software

- Python 3.9+
- PyTorch
- Hugging Face Transformers library

## Citation

**BibTeX:**

```bibtex
@article{masri2025llmtraffic,
  author = {Masri, S. and Ashqar, H. I. and Elhenawy, M.},
  title = {Large Language Models (LLMs) as Traffic Control Systems at Urban Intersections: A New Paradigm},
  journal = {Vehicles},
  volume = {7},
  number = {1},
  pages = {11},
  year = {2025},
  doi = {10.3390/vehicles7010011}
}
```

**APA:**

Masri, S., Ashqar, H. I., & Elhenawy, M. (2025). Large language models (LLMs) as traffic control systems at urban intersections: A new paradigm. Vehicles, 7(1), 11. https://doi.org/10.3390/vehicles7010011

## Glossary

- **Conflict Detection:** Identification of potential vehicle path collisions.
- **FNR (False Negative Rate):** Percentage of actual conflicts not detected.
- **Human-in-the-Loop:** AI system design requiring human validation.
