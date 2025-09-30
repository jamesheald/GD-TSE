# GD-TSE

**GD-TSE** is a JAX-based implementation of gamma-discounted task-space empowerment for trajectory-based learning and control. It includes contextual latent variable models (CLVMs), empowerment-based precoding, and ensemble dynamics modeling. The package leverages Flax, TensorFlow Probability, and JAX for efficient training and evaluation.

---

## Features

- **Contextual Latent Variable Model (CLVM):** Learn context-conditional latent actions for controlled variables.  
- **Empowerment Precoders:** Information-theoretic objective for action coordination.  
- **Ensemble Dynamics Models:** Sample next states with uncertainty estimates.  
- **Flexible Training:** Hydra-configured YAML for hyperparameters, datasets, and logging.  
- **JAX/Flax Based:** Efficient GPU/TPU training with automatic vectorization and batching.

---

## Installation

### Conda Environment

Create the environment from the `environment.yml`:

```bash
conda env create -f environment.yml

# Activate environment
conda activate jax_dt_py312

# Install package in editable mode
pip install -e .