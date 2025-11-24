
---

# GL-SBI: Ginzburg-Landau Parameter Inference via JAX & NRE

**Inverse problem solver for Type-1.5 superconductors using differentiable physics and probabilistic programming.**

This repository implements a pipeline to infer microscopic parameters (specifically the Josephson coupling $\eta$) of a two-component superconductor directly from macroscopic vortex density images.

Because the likelihood function $p(x|\theta)$ for the chaotic Ginzburg-Landau dynamics is intractable, we employ **Neural Ratio Estimation (NRE)** to approximate the posterior distribution without explicit likelihood evaluation. The entire pipeline—from the finite-difference solver to the neural network—is built within the **JAX** ecosystem to leverage JIT compilation and massive vectorization.

---

## Inference Demo

Below is a test run on synthetic data where the ground truth coupling is $\eta=0.8$.

The model outputs a probability distribution rather than a single point estimate, capturing the uncertainty inherent in the chaotic vortex formation process.

---

## Core Features

- **Stateless Physics Engine**: A custom Time-Dependent Ginzburg-Landau (TDGL) solver written in pure JAX. It uses `jax.lax.scan` to unroll time evolution loops on the GPU/CPU, avoiding Python overhead.
    
- **Amortized Inference**: Unlike MCMC or ABC which require expensive simulations for every new observation, the NRE network is trained once and performs inference in milliseconds.
    
- **Physics-Informed Architecture**: The CNN encoder utilizes **Global Average Pooling** to enforce translational invariance, reflecting the uniform nature of the physical laws across the lattice.
    

---

## Structure

|**Module**|**File**|**Description**|
|---|---|---|
|**Solver**|[`src/gl_jax.py`](https://www.google.com/search?q=src/gl_jax.py&authuser=1)|Implements TDGL equations with Peierls substitution for gauge invariance.|
|**Training**|[`src/train_nre.py`](https://www.google.com/search?q=src/train_nre.py&authuser=1)|Main training loop implementing the contrastive loss for NRE.|
|**Model**|[`src/model.py`](https://www.google.com/search?q=src/model.py&authuser=1)|Flax definitions for the CNN Encoder and MLP Classifier.|
|**Generator**|[`src/simulator.py`](https://www.google.com/search?q=src/simulator.py&authuser=1)|Unified interface for generating data (used by both training and offline scripts).|
|**Demo**|[`demo.py`](https://www.google.com/search?q=main.py&authuser=1)|An interactive `matplotlib` tool to visualize vortex dynamics in real-time.|

---

## ⚡ Quick Start

### 1. Prerequisites

Requires Python 3.8+ and JAX.

Bash

```
# Install JAX (Default CPU version)
# For GPU support, please refer to JAX documentation
pip install --upgrade "jax[cpu]"

# Install the rest
pip install -r requirements.txt
```

### 2. Run the Physics Solver

Verify that the JAX solver compiles and runs correctly.

Bash

```
python -m src.gl_jax
# Expected: Done. Final shape: (64, 64)
```

### 3. Interactive Visualization

Launch the virtual lab to explore how $\eta$ and $B$ affect vortex clustering.

Bash

```
python demo.py
```

### 4. Train the Model

Run the training pipeline. This script will:

1. Generate data on-the-fly using `vmap`.
    
2. Train the NRE classifier.
    
3. Output an inference plot (`inference_result.png`).
    

Bash

```
python -m src.train_nre
```

---

## 🔬 Theory Reference

For details on the Multi-component Ginzburg-Landau Hamiltonian and the derivation of the NRE loss function, please refer to **[theory.md](https://www.google.com/search?q=theory.md&authuser=1)**.
