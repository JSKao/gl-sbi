

# Inferring Microscopic Couplings in Superconductors

[![arXiv]]([YOUR_ARXIV_LINK_HERE])
[![GitHub Stars](https://img.shields.io/github/stars/JSKao/ML-Phys.svg?style=social)](https://github.com/JSKao/ML-Phys)

**Keywords:** Simulation-Based Inference (SBI), Ginzburg-Landau (GL) Theory, Differentiable Physics, Condensed Matter Physics.

---

## 💡 Abstract & Challenge

This project addresses the challenging inverse problem of inferring microscopic couplings (Josephson coupling $\eta$, Drag $\nu$) from highly complex, metastable vortex patterns in Type-1.5 superconductors. We explicitly demonstrate that the underlying energy landscape is "glassy" (see Hessian Spectrum below), necessitating a likelihood-free approach.

We leverage a fully differentiable JAX-based TDGL solver coupled with Neural Ratio Estimation (NRE).

---


# Parameter Recovery and Quantitative Accuracy

[posterior_recovery_multipanel](assets/posterior_recovery_multipanel.png)


---

## 🛠️ Usage and Reproducibility

### 1. Setup
```bash
# Clone the repository
git clone [https://github.com/JSKao/ML-Phys/tree/main]
cd [ML-Phys]

# Install dependencies (using the environment requirements.txt)
pip install -r requirements.txt
````

### 2. Reproduce Comprehensive Results

To regenerate the data, train the model, and run the comprehensive evaluation:

Bash

```
# 1. Generate the dataset (Warning: This step is CPU/GPU intensive)
python -m src.generate_data

# 2. Train the NRE model (Takes ~X hours on Y GPU)
python -m src.train_offline

# 3. Run Quantitative Evaluation and generate all figures (Table I, Diagnostic Plots)
python -m src.run_comprehensive_tests 
```

---

> **Code Availability:** The core TDGL solver and NRE architecture are fully differentiable and implemented in JAX/Flax.