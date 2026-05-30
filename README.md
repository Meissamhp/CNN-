# Cyber-Resilient Urban Traffic Forecasting with Physics-Informed Deep Learning

> **MSc Thesis — Transportation Engineering and Mobility**  
> Università degli Studi di Napoli Federico II  
> Supervisor: Prof. Marcello Cinque | Defense: June 30, 2026  
> Author: Meisam Hosseinpour

---

## Overview

This repository implements a **Physics-Informed Neural Network (PINN)** framework for urban traffic forecasting on the **Naples bus network**, designed to be robust against **cyber-adversarial attacks**.

The core idea: instead of a standard "black-box" AI, the model embeds the **Lighthill-Whitham-Richards (LWR)** conservation law directly into the neural network's loss function — so the model cannot be "deceived" by False Data Injection (FDI) attacks that violate physical traffic laws.

---

## Problem

Modern Intelligent Transportation Systems (ITS) face two critical challenges:

- **Black-box AI:** Standard deep learning (LSTM, CNN, GRU) ignores physical laws and produces physically impossible predictions under noisy or corrupted data
- **Cyber-attacks:** False Data Injection (FDI) attacks inject small, stealthy perturbations into GPS/AVL sensor streams — standard AI cannot detect them

---

## Solution: Physics as a Defense

The PINN framework adds a **physics residual** to the loss function:

```
Total Loss = Data Loss (MSE) + λ × Physics Residual (LWR)
```

Where the LWR conservation law acts as a **mathematical firewall**:

```
∂ρ/∂t + ∂q/∂x = 0
```

If incoming data violates this law (e.g., impossible bus speeds), the model rejects it — not through a firewall, but through physics.

---

## Dataset

- **Source:** Naples Metropolitan Transit Network (ANM)
- **Size:** 15,726 trips across 2,356 bus stations
- **Features:** arrival_sec, travel_time, station sequences
- **Period:** Real-world operational data

---

## Models Compared

| Model | Type | Physical Constraints |
|-------|------|---------------------|
| RNN | Baseline | ❌ None |
| CNN | Baseline | ❌ None |
| LSTM | Baseline | ❌ None |
| GRU | Baseline | ❌ None |
| **PINN-LWR** | **Proposed** | ✅ **LWR Conservation Law** |

---

## Project Structure

```
traffic_thesis_unified/
├── models/          # PINN + baseline model architectures
├── data/
│   ├── raw/         # Original Naples bus dataset
│   └── processed/   # Cleaned and normalized data
├── evaluation/      # Benchmarking & adversarial stress tests
└── results/         # Output figures and comparison tables
```

---

## Requirements

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

---

## How to Run

```bash
cd traffic_thesis_unified

# Train and evaluate all models
python evaluation/evaluate_clean.py

# Run adversarial (FDI attack) stress tests
python evaluation/evaluate_attacks.py

# Sensitivity analysis of physics weight λ
python evaluation/evaluate_lambda.py
```

---

## Key Innovation

> **"Physics-as-a-Defense"** — By embedding traffic conservation laws into the learning process, the PINN framework recognizes data that violates physical consistency as either sensor noise or a cyber-attack, maintaining reliable predictions where standard AI fails.

---

## Citation

```
Hosseinpour, M. (2026). Enhancing Cyber-Resilience in Urban Traffic Forecasting 
using Physics-Informed Deep Learning: A Case Study of the Naples Bus Network. 
MSc Thesis, Università degli Studi di Napoli Federico II.
```

---

## Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Meisam_Hosseinpour-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/meisam-hosseinpour-ab45bb2bb)
[![Email](https://img.shields.io/badge/Email-meisamhosseinpour1990@gmail.com-D14836?style=flat&logo=gmail)](mailto:meisamhosseinpour1990@gmail.com)
