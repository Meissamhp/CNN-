# Physics-Informed Neural Networks for Urban Traffic & CO Dispersion

> MSc Thesis Project — Università degli Studi / Apple Developer Academy  
> Author: Meisam Hosseinpour

---

## Overview

This repository contains the full implementation of a multiphysics framework using **Physics-Informed Neural Networks (PINNs)** to simultaneously model:

- 🚗 **Urban traffic flow** — based on the Lighthill-Whitham-Richards (LWR) model
- 🌫️ **Carbon monoxide (CO) dispersion** — via advection-diffusion-reaction equations

The model solves coupled PDEs **without mesh discretization**, achieving 59–65% lower error compared to classical Finite Volume Methods (FVM).

---

## Key Results

| Metric | PINN (This work) | FVM (Baseline) | Improvement |
|--------|-----------------|----------------|-------------|
| Traffic Density MAE | 0.011 vehicles/m² | 0.027 | 59.3% |
| CO Concentration MAE | 0.005 ppm | 0.016 | 65.2% |
| R² Score | 0.94 – 0.95 | 0.85 – 0.86 | +10.5% |
| Computational Time | 350s | 670s | 47.8% faster |

---

## Project Structure

```
traffic_thesis_unified/
├── models/          # PINN architecture and training
├── data/            # Raw and processed datasets
├── evaluation/      # Evaluation scripts
├── results/         # Output figures and tables
```

---

## Methods

- **Framework:** DeepXDE + PyTorch
- **Architecture:** 12 hidden layers, 150 neurons/layer, tanh activation
- **Optimizer:** AdamW (lr=0.0004)
- **Physics:** LWR traffic model + advection-diffusion-reaction PDE
- **Validation:** Compared against FVM, tested on real urban data from Naples

---

## Requirements

```bash
pip install deepxde torch numpy pandas matplotlib scikit-learn
```

---

## How to Run

```bash
cd traffic_thesis_unified
python models/train_pinn.py
python evaluation/evaluate_clean.py
```

---

## Citation

If you use this work, please cite:

```
Hosseinpour, M. (2026). A Pioneering Multiphysics Framework for Integrated Modeling 
of Urban Traffic Dynamics and Carbon Monoxide Dispersion Using Physics-Informed 
Neural Networks. MSc Thesis.
```

---

## Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Meisam_Hosseinpour-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/meisam-hosseinpour-ab45bb2bb)
[![Email](https://img.shields.io/badge/Email-meisamhosseinpour1990@gmail.com-D14836?style=flat&logo=gmail)](mailto:meisamhosseinpour1990@gmail.com)
