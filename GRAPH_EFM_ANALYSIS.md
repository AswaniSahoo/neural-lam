# Graph-EFM Analysis: `prob_model_lam` Branch Deep-Dive

> **Author:** Aswani Sahoo  
> **Date:** March 1, 2026  
> **Context:** GSoC 2026 — Understanding the CVAE model for refactoring (Issue #49)

---

## Table of Contents

1. [Branch Overview](#1-branch-overview)
2. [graph_efm.py — The CVAE Model](#2-graph_efmpy--the-cvae-model)
3. [What GraphEFM Overrides from ARModel](#3-what-graphefm-overrides-from-armodel)
4. [What GraphEFM Adds New](#4-what-graphefm-adds-new)
5. [predict_step vs predict_step_vi (VI ≠ Prior Sampling)](#5-predict_step-vs-unroll_prediction_vi)
6. [grid_prev_embedder vs grid_current_embedder](#6-grid_prev_embedder-vs-grid_current_embedder)
7. [Coupling & Conflicts with ARModel's Assumptions](#7-coupling--conflicts-with-armodels-assumptions)
8. [Changes in Supporting Files](#8-changes-in-supporting-files)
9. [New File Architecture (Latent Enc/Dec)](#9-new-file-architecture)
10. [Key Refactoring Implications](#10-key-refactoring-implications)

---

## 1. Branch Overview

The `prob_model_lam` branch adds **probabilistic (ensemble) forecasting** via a CVAE (Conditional Variational Autoencoder) architecture. It introduces a fundamentally different prediction paradigm.

### Key Difference from `main`

| Aspect | `main` branch | `prob_model_lam` branch |
|---|---|---|
| Architecture | Deterministic encode-process-decode | CVAE with latent variable sampling |
| Prediction | Single forecast trajectory | Ensemble of S sampled trajectories |
| Training | Direct MSE/MAE loss | ELBO = Likelihood + β·KL divergence |
| `predict_step` | Abstract (one method) | Two modes: prior sampling (inference) + VI (training) |
| Outputs | `(new_state, pred_std)` | Ensemble trajectories `(B, S, T, N, d_f)` |
| New modules | None | Latent encoder/decoder hierarchy (6 new files) |
| Metrics | MSE, MAE only | + CRPS_ens, spread_squared, spsk_ratio |

### File Changes Summary

| File | Main Lines | Prob Lines | Status |
|---|---|---|---|
| `ar_model.py` | 772 | 599 | **Older version** (pre-datastore refactor) |
| `graph_efm.py` | N/A | **1129** | **NEW** — The CVAE model |
| `metrics.py` | 238 | 370 | **Extended** (+crps_ens, +spread_squared) |
| `vis.py` | 185 | 351 | **Extended** (+plot_ensemble_prediction, +plot_latent_samples) |
| `interaction_net.py` | 164 | 232 | **Extended** (+PropagationNet) |
| Latent encoders (3 files) | N/A | ~260 total | **NEW** |
| Latent decoders (3 files) | N/A | ~340 total | **NEW** |

---

## 2. graph_efm.py — The CVAE Model

**Class:** `GraphEFM(ARModel)` — 1129 lines  
**Inherits from:** `ARModel` directly (bypasses `BaseGraphModel`)

### Full Method Inventory

| # | Method | Lines | Purpose |
|---|---|---|---|
| 1 | `__init__` | 22-236 | Sets up embedders, prior model, encoder, decoder, ensemble metrics |
| 2 | `sample_next_state` | 238-257 | Optionally sample from Gaussian or just return mean |
| 3 | `embedd_current` | 259-293 | Embed grid with current target state (for encoder path) |
| 4 | `embedd_all` | 295-372 | Embed grid without current state + all graph features (for prior path) |
| 5 | `compute_step_loss` | 374-427 | Full ELBO loss for one timestep: embed → encode → likelihood + KL |
| 6 | `estimate_likelihood` | 429-475 | Sample latent, decode, compute neg-loss as likelihood |
| 7 | `training_step` | 477-581 | **Overrides ARModel** — ELBO training with optional CRPS |
| 8 | `predict_step` | 583-614 | **Overrides ARModel** — Prior sampling for inference |
| 9 | `sample_trajectories` | 616-661 | Generate S trajectory samples (prior or encoder) |
| 10 | `unroll_prediction_vi` | 663-738 | **NEW** — AR rollout using encoder (variational) distribution |
| 11 | `plot_examples` | 740-839 | **Overrides ARModel** — Ensemble version of plotting |
| 12 | `ensemble_common_step` | 841-892 | Compute ensemble forecast + spread/skill metrics |
| 13 | `validation_step` | 894-1033 | **Overrides ARModel** — Adds ensemble metrics + latent space plots |
| 14 | `log_spsk_ratio` | 1035-1070 | **NEW** — Logs spread-skill ratio |
| 15 | `on_validation_epoch_end` | 1072-1078 | **Overrides ARModel** — Adds spsk_ratio logging |
| 16 | `test_step` | 1080-1120 | **Overrides ARModel** — Adds ensemble CRPS/MAE/spread |
| 17 | `on_test_epoch_end` | 1122-1129 | **Overrides ARModel** — Adds spsk_ratio logging |

---

## 3. What GraphEFM Overrides from ARModel

| ARModel Method | GraphEFM Override | What Changes |
|---|---|---|
| `__init__` | ✅ **Extends** | Adds embedders, CVAE components (prior, encoder, decoder), ensemble metrics |
| `training_step` | ✅ **Complete override** | Replaces MSE loss with ELBO (likelihood + KL) + optional CRPS |
| `predict_step` | ✅ **Complete override** | Samples from prior distribution instead of deterministic prediction |
| `plot_examples` | ✅ **Complete override** | Plots ensemble predictions with members, mean, std |
| `validation_step` | ✅ **Extends** (calls `super()`) | Adds ensemble metrics (spread, skill) and latent-space plots |
| `test_step` | ✅ **Extends** (calls `super()`) | Adds ensemble CRPS, MAE, spread |
| `on_validation_epoch_end` | ✅ **Extends** (calls `super()`) | Logs spread-skill ratio before parent cleanup |
| `on_test_epoch_end` | ✅ **Extends** (calls `super()`) | Logs spread-skill ratio after parent cleanup |
| `unroll_prediction` | ❌ **Inherits as-is** | Used for prior-based trajectory sampling |

> ⚠️ **Critical:** GraphEFM does NOT override `unroll_prediction`. It inherits the AR rollout from `ARModel` and uses it inside `sample_trajectories()` for prior-based inference. But it also has its own `unroll_prediction_vi()` for encoder-based rollout.

---

## 4. What GraphEFM Adds New

| New Method | Purpose |
|---|---|
| `sample_next_state` | Sample from predicted Gaussian or just take mean |
| `embedd_current` | Embed grid features **including the current target** (encoder path) |
| `embedd_all` | Embed grid features **without current target** + all graph embeddings (prior path) |
| `compute_step_loss` | One-timestep ELBO computation (embed → encode → likelihood + KL) |
| `estimate_likelihood` | Sample from latent dist, decode, compute negative loss |
| `sample_trajectories` | Generate S parallel trajectory rollouts |
| `unroll_prediction_vi` | AR rollout using encoder (variational inference) distribution |
| `ensemble_common_step` | Compute ensemble metrics (spread, skill MSE) |
| `log_spsk_ratio` | Compute and log spread-skill ratio |

### New Components (separate files)

| Component | File | Purpose |
|---|---|---|
| `BaseLatentEncoder` | `base_latent_encoder.py` | Abstract encoder → distribution over latent (isotropic/diagonal Gaussian) |
| `GraphLatentEncoder` | `graph_latent_encoder.py` | Grid→mesh encoder with `PropagationNet` + m2m processor |
| `HiGraphLatentEncoder` | `hi_graph_latent_encoder.py` | Hierarchical version (bottom-up through mesh levels) |
| `ConstantLatentEncoder` | `constant_latent_encoder.py` | Fixed N(0,I) prior (no learned prior) |
| `BaseGraphLatentDecoder` | `base_graph_latent_decoder.py` | Abstract decoder: latent→grid prediction |
| `GraphLatentDecoder` | `graph_latent_decoder.py` | Non-hierarchical decoder |
| `HiGraphLatentDecoder` | `hi_graph_latent_decoder.py` | Hierarchical decoder (up then down through mesh) |

---

## 5. predict_step vs unroll_prediction_vi

This is the **core architectural distinction** of the CVAE:

### `predict_step` (Prior Sampling — Inference Mode)

```
Input: prev_state, prev_prev_state, forcing
                   │
     ┌─────────────▼───────────────┐
     │ grid_prev_embedder          │  ← Embeds WITHOUT current target
     │ (prev_prev, prev, forcing,  │
     │  static_features)           │
     └─────────────┬───────────────┘
                   │ grid_prev_emb
     ┌─────────────▼───────────────┐
     │ prior_model                 │  ← Prior distribution p(z|X_{<t})
     │ (GraphLatentEncoder or      │
     │  ConstantLatentEncoder)     │
     └─────────────┬───────────────┘
                   │ z ~ prior_dist
     ┌─────────────▼───────────────┐
     │ decoder                     │  ← Decode: z + grid_emb → prediction
     │ (GraphLatentDecoder)        │
     └─────────────┬───────────────┘
                   │
              pred_mean, pred_std
                   │
     sample_next_state(mean, std)
                   │
              new_state
```

**Key:** No access to target. Latent `z` sampled from prior. Used at inference time.

### `unroll_prediction_vi` (Variational Inference — Training Mode)

```
Input: prev_state, prev_prev_state, forcing, current_state (TARGET!)
                   │
     ┌─────────────▼───────────────┐
     │ grid_current_embedder       │  ← Embeds WITH current target
     │ (prev_prev, prev, forcing,  │
     │  static_features,           │
     │  current_state ← TARGET)    │
     └─────────────┬───────────────┘
                   │ grid_current_emb
     ┌─────────────▼───────────────┐
     │ encoder                     │  ← Variational dist q(z|X_{<t}, X_t)
     │ (GraphLatentEncoder with    │
     │  output_dist="diagonal")    │
     └─────────────┬───────────────┘
                   │ z ~ var_dist (has grad via rsample)
     ┌─────────────▼───────────────┐
     │ decoder (same decoder)      │  ← Decode: z + grid_emb → prediction
     └─────────────┬───────────────┘
                   │
              pred_mean, pred_std
```

**Key:** Has access to target. Latent `z` sampled from variational distribution (reparameterized). Used during training to provide a tighter bound.

### Summary of Differences

| Aspect | `predict_step` (prior) | `unroll_prediction_vi` (encoder) |
|---|---|---|
| **Embedder** | `grid_prev_embedder` | `grid_current_embedder` |
| **Latent source** | `prior_model` (no target access) | `encoder` (sees target) |
| **Distribution** | Typically isotropic | Always diagonal (learned σ) |
| **When used** | Inference / prior trajectories | Training ELBO / encoder trajectories |
| **Gradient flow** | Through `rsample()` from prior | Through `rsample()` from encoder |

---

## 6. grid_prev_embedder vs grid_current_embedder

These are the two separate grid embedding MLPs — the **most important architectural distinction**:

### `grid_prev_embedder` (Prior Path)

```python
grid_features = cat(prev_prev_state, prev_state, forcing, static_features)
# Input dim: grid_dim = 2 * d_state + d_static + d_forcing * (past+future+1)
# Output: (B, N_grid, d_h)
```

- Used for prior computation and decoding
- **Does NOT see the current target** X_t
- Input dimension = `self.grid_dim`

### `grid_current_embedder` (Encoder Path)

```python
grid_features = cat(prev_prev_state, prev_state, forcing, static_features, current_state)
# Input dim: grid_current_dim = grid_dim + d_state  ← EXTRA d_state for target
# Output: (B, N_grid, d_h)
```

- Used only for the encoder (variational approximation)
- **SEES the current target** X_t — this is what makes it a CVAE
- Input dimension = `grid_dim + GRID_STATE_DIM` (extra state features)

### Why Two Embedders?

The CVAE requires an **amortized inference gap**:
- The **encoder** sees the target to compute q(z|X_{≤t}) — a posterior approximation
- The **prior** only sees past states to compute p(z|X_{<t})
- At inference, only the prior is available (no future data), so the model must learn to make the prior close enough to the encoder

This is the standard CVAE setup: the encoder is the "recognition model" and the prior is the "generative model".

---

## 7. Coupling & Conflicts with ARModel's Assumptions

### 7.1. training_step Completely Replaced

`ARModel.training_step` assumes:
```
prediction, target, pred_std, _ = self.common_step(batch)
loss = self.loss(prediction, target, pred_std, mask=...)
```

`GraphEFM.training_step` does:
```
# ELBO with per-step encode-decode
for each step:
    compute_step_loss → ELBO likelihood + KL
loss = -likelihood + β * KL + crps_weight * CRPS
```

> ⚠️ **Conflict:** The whole `common_step → unroll_prediction → loss` pipeline is **bypassed** in favor of a per-step ELBO computation with KL divergence. The refactored architecture must support both pathways.

### 7.2. GraphEFM Bypasses BaseGraphModel

```
ARModel → BaseGraphModel → GraphLAM    (main branch)
ARModel → GraphEFM                      (prob_model_lam branch)
```

GraphEFM directly extends `ARModel`, NOT `BaseGraphModel`. This means:
- It re-implements graph loading (the entire `__init__` setup)
- It does NOT use `BaseGraphModel.predict_step` encode-process-decode
- The encoder/decoder architecture is completely separate from the deterministic models

> ⚠️ **Conflict:** There's massive code duplication between `BaseGraphModel.__init__` and `GraphEFM.__init__` (graph loading, embedder creation, mesh setup). Refactoring should extract this into shared infrastructure.

### 7.3. Batch Format Assumption

**Main branch:** `batch = (init_states, target_states, forcing_features, batch_times)` — 4 elements  
**Prob branch:** `batch = (init_states, target_states, forcing_features)` — 3 elements

> ⚠️ **Conflict:** The prob branch is on an older version of `ar_model.py` that doesn't have `batch_times`. Merging will require updating GraphEFM to handle the 4-element batch.

### 7.4. Hardcoded Wandb Dependency

The prob branch uses `wandb.log()` directly in `plot_examples`, `validation_step`, and `log_spsk_ratio`. Main branch has been refactored to use `self.logger` generically.

> ⚠️ **Conflict:** GraphEFM must be updated to use Lightning's logger abstraction.

### 7.5. Constants Module vs Datastore

The prob branch uses `constants.PARAM_NAMES_SHORT`, `constants.GRID_SHAPE`, `constants.LAMBERT_PROJ`, etc. Main branch has been refactored to use `datastore.get_vars_names()`, `datastore.grid_shape`, etc.

> ⚠️ **Conflict:** All constants references in GraphEFM and vis.py must be replaced with datastore API calls.

### 7.6. `border_mask` vs `boundary_mask`

GraphEFM uses `self.border_mask` (prob branch naming), while main uses `self.boundary_mask`. Same variable, different name.

### 7.7. Metric Storage Differences

GraphEFM adds ensemble-specific metrics:
- `val_metrics`: `spread_squared`, `ens_mse`
- `test_metrics`: `ens_mae`, `ens_mse`, `crps_ens`, `spread_squared`

These must be stored alongside the deterministic metrics, requiring the metric system to handle both per-sample metrics and ensemble metrics.

---

## 8. Changes in Supporting Files

### metrics.py — New Functions

| New Function | Lines | Purpose |
|---|---|---|
| `crps_ens` | 235-325 | **Ensemble CRPS** — unbiased estimator from samples (handles M=1,2,<10,≥10 cases) |
| `spread_squared` | 328-357 | **Ensemble variance** — `torch.var(pred, dim=ens_dim)` for spread-skill ratio |

**Key design:** Both have an `ens_dim` parameter (default=1) specifying which dimension holds ensemble members. The CRPS handles three complexity regimes depending on ensemble size.

### vis.py — New Functions

| New Function | Lines | Purpose |
|---|---|---|
| `plot_ensemble_prediction` | 116-202 | 3×3 grid: ground truth + ens mean + ens std + up to 6 members |
| `plot_on_axis` | 205-223 | Helper to plot one weather state on a given axis |
| `plot_latent_samples` | 275-351 | Side-by-side prior vs variational latent samples as images |

**Note:** vis.py on prob branch still uses `constants.GRID_SHAPE` and `constants.LAMBERT_PROJ` — will need updating.

### interaction_net.py — PropagationNet

```python
class PropagationNet(InteractionNet):
```

| Aspect | InteractionNet | PropagationNet |
|---|---|---|
| Aggregation | `sum` | `mean` (forced) |
| Node residual | `rec_rep = rec_rep + rec_diff` | `rec_rep = edge_rep_aggr + rec_diff` |
| Message residual | None | `x_j + edge_mlp(...)` (sender residual) |
| Purpose | Standard message passing | Incentivizes information propagation from sender → receiver |

**Key difference:** PropagationNet uses mean aggregation and adds residual connections from the sender node directly into the message. The receiver update residual is to the aggregated messages, not to the receiver's own representation. Used exclusively in encoders (where information needs to propagate from grid to mesh).

---

## 9. New File Architecture

```
Inheritance Hierarchy (prob_model_lam branch):

   pl.LightningModule
         │
      ARModel (599 lines)
      ╱       ╲
GraphEFM   BaseGraphModel
(CVAE)     (deterministic)
              ╱        ╲
         GraphFM    GraphCast


Latent Encoder Hierarchy:
   BaseLatentEncoder (nn.Module)
      ╱        │          ╲
GraphLatent  HiGraphLatent  ConstantLatent
Encoder      Encoder        Encoder


Latent Decoder Hierarchy:
   BaseGraphLatentDecoder (nn.Module)
      ╱              ╲
GraphLatent     HiGraphLatent
Decoder          Decoder
```

---

## 10. Key Refactoring Implications

### 10.1. The Dual Training Path Problem

The biggest challenge is that deterministic models and probabilistic models have **fundamentally different training loops**:

```
Deterministic:  batch → unroll_prediction → loss(pred, target)
Probabilistic:  batch → per-step {encode → sample z → decode → ELBO}
```

The refactored `ForecasterModule` must support both paths without forcing probabilistic models to go through the deterministic pipeline.

### 10.2. What Must Be Shared

| Component | Current | Should Be |
|---|---|---|
| Graph loading & embedding | Duplicated in `BaseGraphModel.__init__` and `GraphEFM.__init__` | Shared graph infrastructure module |
| AR rollout loop | `ARModel.unroll_prediction` | Generic `Forecaster` that probabilistic models can override |
| Metric storage & aggregation | Mixed into `ARModel` lifecycle hooks | Extracted `MetricsManager` callback |
| Plotting | Mixed into `ARModel` lifecycle hooks | Extracted `PlottingCallback` |
| Loss computation | Inline in `training_step` | Pluggable loss strategy |

### 10.3. What Must Stay Separate

| Component | Deterministic | Probabilistic |
|---|---|---|
| `predict_step` | Single encode-process-decode | Prior sampling + decoder |
| Training loss | MSE/MAE on predictions | ELBO (likelihood + KL) |
| Ensemble generation | N/A | `sample_trajectories` |
| Encoder | N/A | Variational encoder |
| Latent variable | N/A | Full CVAE latent path |

### 10.4. The Clean Refactoring Target

```
ForecasterModule(pl.LightningModule)
    ├── configure_optimizers, on_load_checkpoint (Lightning lifecycle)
    ├── MetricsCallback (pluggable)
    └── PlottingCallback (pluggable)

Forecaster(ABC)
    ├── training_step, validation_step, test_step
    ├── unroll_prediction
    └── common_step

StepPredictor(ABC)
    ├── predict_step (deterministic)
    └── predict_step_vi + sample_trajectories (probabilistic)
```

Both `BaseGraphModel` and `GraphEFM` would implement `StepPredictor`, with `GraphEFM` also implementing the variational interface.

---
