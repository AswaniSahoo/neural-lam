# Paper → Code Mapping: Graph-EFM & Diffusion-LAM

> **Author:** Aswani Sahoo  
> **Date:** March 1, 2026  
> **Papers:** Graph-EFM (arXiv:2406.04759), Diffusion-LAM (arXiv:2502.07532)  
> **Context:** GSoC 2026 — Mapping paper architecture to `neural-lam` code for Issue #49

---

## 1. Graph-EFM: Paper → Code Mapping

### 1.1 Generative Model (Paper §3.1 → Code)

The paper defines the generative model as:

```
p_θ(x_t | x_{<t}) = ∫ p_θ(x_t | z_t, x_{<t}) · p_θ(z_t | x_{<t}) dz_t
```

| Paper Notation | Math | Code Location | Code Implementation |
|---|---|---|---|
| **x_{<t}** | Past states | `prev_state`, `prev_prev_state`, `forcing` | Inputs to `embedd_all()` |
| **z_t** | Latent variable | `latent_samples` | `prior_dist.rsample()` in `predict_step` L605 |
| **p_θ(z_t \| x_{<t})** | Prior distribution | `self.prior_model(grid_prev_emb, ...)` | `predict_step` L600-602 |
| **p_θ(x_t \| z_t, x_{<t})** | Likelihood/Decoder | `self.decoder(grid_prev_emb, latent_samples, ...)` | `predict_step` L610-612 |
| **x_t (output)** | Predicted state | `self.sample_next_state(pred_mean, pred_std)` | `predict_step` L614 |

#### Code: `predict_step` (Generative Model = Prior Sampling)

```python
# graph_efm.py L583-614
def predict_step(self, prev_state, prev_prev_state, forcing):
    # 1. Embed past states: x_{<t} → h
    grid_prev_emb, graph_emb = self.embedd_all(prev_state, prev_prev_state, forcing)
    
    # 2. Prior: p_θ(z_t | x_{<t})
    prior_dist = self.prior_model(grid_prev_emb, graph_emb=graph_emb)
    
    # 3. Sample: z_t ~ p_θ(z_t | x_{<t})
    latent_samples = prior_dist.rsample()
    
    # 4. Decode: p_θ(x_t | z_t, x_{<t})
    pred_mean, pred_std = self.decoder(grid_prev_emb, latent_samples, prev_state, graph_emb)
    
    # 5. Optionally sample observation noise
    return self.sample_next_state(pred_mean, pred_std), pred_std
```

---

### 1.2 Variational Inference (Paper §3.1 → Code)

The paper introduces the encoder (recognition model):

```
q_φ(z_t | x_{≤t}) ≈ p_θ(z_t | x_{≤t})
```

| Paper Notation | Math | Code Location | Code Implementation |
|---|---|---|---|
| **x_{≤t}** | Past + current target | `prev_state` + `current_state` | Input to `embedd_current()` |
| **q_φ(z_t \| x_{≤t})** | Encoder distribution | `self.encoder(grid_current_emb, ...)` | `compute_step_loss` L403-404 |
| **z_t ~ q_φ** | Encoder sample | `var_dist.rsample()` | `estimate_likelihood` L450 |
| **Recognition model** | Amortized inference | `grid_current_embedder` | Sees target X_t (extra input dim) |

#### Code: `compute_step_loss` (Variational Inference at Training)

```python
# graph_efm.py L374-427
def compute_step_loss(self, prev_states, current_state, forcing_features):
    # 1. Embed WITHOUT target (for prior + decoder)
    grid_prev_emb, graph_emb = self.embedd_all(...)
    
    # 2. Embed WITH target (for encoder) — THE KEY CVAE DIFFERENCE
    grid_current_emb = self.embedd_current(..., current_state)  # ← sees X_t!
    
    # 3. Encoder: q_φ(z_t | x_{≤t})
    var_dist = self.encoder(grid_current_emb, graph_emb=graph_emb)
    
    # 4. Likelihood estimation (sample from encoder, decode)
    likelihood_term, pred_mean, pred_std = self.estimate_likelihood(
        var_dist, current_state, last_state, grid_prev_emb, graph_emb)
    
    # 5. Prior: p_θ(z_t | x_{<t})
    prior_dist = self.prior_model(grid_prev_emb, graph_emb=graph_emb)
    
    # 6. KL divergence: KL(q_φ || p_θ) — regularization
    kl_term = torch.sum(kl_divergence(var_dist, prior_dist), dim=(1,2))
    
    return likelihood_term, kl_term, pred_mean, pred_std
```

---

### 1.3 ELBO Training Objective (Paper §3.2 → Code)

The paper's ELBO:

```
ELBO = E_q[log p_θ(x_t | z_t, x_{<t})] - β · KL(q_φ(z_t|x_{≤t}) || p_θ(z_t|x_{<t}))
Loss = -ELBO = -Likelihood + β · KL
```

| Paper Term | Math | Code Variable | Code Location |
|---|---|---|---|
| **E_q[log p(x\|z)]** | Likelihood | `mean_likelihood` | `training_step` L533 |
| **KL(q\|\|p)** | KL divergence | `mean_kl` | `training_step` L543 |
| **β** | KL weight | `self.kl_beta` | `training_step` L545 |
| **ELBO** | Full bound | `elbo` | `training_step` L544 |
| **Loss** | Minimized objective | `loss` | `training_step` L545 |
| **CRPS term** | Optional ensemble loss | `crps_loss` | `training_step` L571-574 |

#### Code: `training_step` (ELBO Optimization)

```python
# graph_efm.py L477-581
def training_step(self, batch):
    # AR loop: compute per-step ELBO
    for i in range(pred_steps):
        loss_like_term, loss_kl_term, pred_mean, pred_std = self.compute_step_loss(...)
        loss_like_list.append(loss_like_term)
        loss_kl_list.append(loss_kl_term)
        
        # Update state for next step (sample or mean)
        predicted_state = self.sample_next_state(pred_mean, pred_std)
        new_state = border_mask * target + interior_mask * predicted_state
    
    # Aggregate ELBO over time
    mean_likelihood = mean(sum(loss_like_list))     # Paper: E_q[log p(x|z)]
    mean_kl = mean(sum(loss_kl_list))               # Paper: KL(q||p)
    elbo = mean_likelihood - mean_kl                # Paper: ELBO
    loss = -mean_likelihood + kl_beta * mean_kl     # Paper: -ELBO with β
    
    # Optional: add CRPS loss on prior-sampled trajectories
    if crps_weight > 0:
        loss = loss + crps_weight * crps_loss
```

#### β-Annealing (Paper §3.2)

The paper describes β-annealing: start with β=0 (pure autoencoder mode), linearly increase to β=1 over training. In code, this is controlled by `args.kl_beta`:

```python
# When kl_beta = 0: pure autoencoder, no KL constraint
if self.kl_beta > 0:
    loss = -mean_likelihood + self.kl_beta * mean_kl
else:
    loss = -mean_likelihood  # Pure reconstruction
```

> **Note:** The actual annealing schedule is handled outside the model (in the training script config), not inside `graph_efm.py`.

---

### 1.4 Architecture Components (Paper Fig. 1 → Code)

```
Paper Figure 1 → Code Mapping:

┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (ELBO)                          │
│                                                             │
│  x_{<t} ──→ grid_prev_embedder ──→ grid_prev_emb            │
│              (prior path)            │                      │
│                                      ├──→ prior_model       │
│                                      │    p(z|x_{<t})       │
│                                      │         │            │
│  x_{≤t} ──→ grid_current_embedder    │    KL(q||p)          │
│              (encoder path)          │         ↑            │
│                    │                 │    ┌────┘            │
│                    └──→ encoder ─────┼──→ q(z|x_{≤t})       │
│                         q_φ              │                  │
│                                     z ~ q│                  │
│                                          ↓                  │
│                              decoder(grid_prev_emb, z)      │
│                                          │                  │
│                                     pred_mean, pred_std     │
│                                          │                  │
│                              -loss(pred, target) = Lhood    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE (Prior Sampling)                │
│                                                             │
│  x_{<t} ──→ grid_prev_embedder ──→ grid_prev_emb            │
│                                      │                      │
│                                      └──→ prior_model       │
│                                           p(z|x_{<t})       │
│                                                │            │
│                                           z ~ p│ (× S)      │
│                                                ↓            │
│                                    decoder(grid_prev_emb,z) │
│                                                │            │
│                                    S trajectories (ensemble)│
└─────────────────────────────────────────────────────────────┘
```

---

### 1.5 Graph Architecture (Paper §3 → Code)

| Paper Component | Description | Code Class | File |
|---|---|---|---|
| **Grid nodes** | Weather state at each grid point | Handled by `ARModel` | `ar_model.py` |
| **Mesh nodes** | Multi-resolution mesh (icosahedral) | Loaded in `GraphEFM.__init__` | `graph_efm.py` L34 |
| **Grid→Mesh edges** | Connect grid to bottom mesh level | `self.g2m_edge_index` | Loaded from graph dir |
| **Mesh→Grid edges** | Connect bottom mesh to grid | `self.m2g_edge_index` | Loaded from graph dir |
| **Mesh→Mesh edges** | Intra-level connections | `self.m2m_edge_index` (list) | Per hierarchy level |
| **Up edges** | Connect level l → l+1 | `self.mesh_up_edge_index` | Hierarchical only |
| **Down edges** | Connect level l+1 → l | `self.mesh_down_edge_index` | Hierarchical only |
| **PropagationNet** | GNN for encoder (info propagation) | `PropagationNet` | `interaction_net.py` L134 |
| **InteractionNet** | GNN for decoder (standard MP) | `InteractionNet` | `interaction_net.py` L10 |

---

### 1.6 Encoder Architecture (Paper §3.1 → Code)

| Paper | Code Class | Key Design |
|---|---|---|
| Flat encoder | `GraphLatentEncoder` | g2m PropagationNet → m2m processor → latent_param_map |
| Hierarchical encoder | `HiGraphLatentEncoder` | g2m PropagationNet → up GNNs through levels → intra-level processing → param map |
| Constant prior | `ConstantLatentEncoder` | Fixed N(0, I), no learned parameters |

**Output distributions:**
- **Isotropic** (`output_dist="isotropic"`): output = mean only, σ = I → `N(μ, I)`
- **Diagonal** (`output_dist="diagonal"`): output = mean + log-std → `N(μ, softplus(σ_raw) + ε)`

The encoder always outputs `diagonal` (learned σ), while the prior can be either `isotropic` or `diagonal`.

---

### 1.7 Decoder Architecture (Paper §3.1 → Code)

| Paper | Code Class | Key Design |
|---|---|---|
| Flat decoder | `GraphLatentDecoder` | g2m InteractionNet → m2m processor → m2g PropagationNet |
| Hierarchical decoder | `HiGraphLatentDecoder` | g2m → up hierarchy (InteractionNet) → intra processing → down hierarchy (PropagationNet) → m2g |

**Decoder call signature:**
```python
pred_mean, pred_std = self.decoder(
    grid_prev_emb,     # Grid representation (from prior path)
    latent_samples,    # z ~ q(z) or z ~ p(z)
    last_state,        # For residual connection
    graph_emb          # All graph embeddings
)
```

The decoder combines grid_prev_emb with the latent variable via `combine_with_latent()`, then maps to output via `output_map` MLP.

---

### 1.8 Ensemble Generation (Paper §3.3 → Code)

```
Paper: Draw S samples z_t^(s) ~ p(z_t | x_{<t}), s = 1,...,S
       Each z_t^(s) → x_t^(s) via decoder
       Repeat autoregressively for T steps
```

| Paper Concept | Code Method | Implementation |
|---|---|---|
| S trajectory samples | `sample_trajectories()` L616-661 | Calls `unroll_prediction` S times |
| Prior-based ensemble | `unroll_prediction()` (inherited from ARModel) | Each call samples different z from prior |
| Encoder-based ensemble | `unroll_prediction_vi()` L663-738 | Each call samples different z from encoder |
| Ensemble mean | `torch.mean(trajectories, dim=1)` | Computed in `ensemble_common_step` L875 |
| Ensemble spread | `torch.var(trajectories, dim=1)` | Via `metrics.spread_squared` |

---

### 1.9 Evaluation Metrics (Paper §4.1 → Code)

| Paper Metric | Formula | Code Function | Code Location |
|---|---|---|---|
| **RMSE** | √(mean(MSE)) | `metrics.mse` → `torch.sqrt` | `ar_model.py` aggregate_and_plot_metrics |
| **MAE** | mean(\|pred - target\|) | `metrics.mae` | `metrics.py` L142 |
| **CRPS** (ensemble) | Unbiased estimator from samples | `metrics.crps_ens` | `metrics.py` L235-325 |
| **Spread** | √(Var(ensemble)) | `metrics.spread_squared` → `sqrt` | `metrics.py` L328-357 |
| **Spread-Skill Ratio** | √((M+1)/M) · spread/RMSE | `log_spsk_ratio()` | `graph_efm.py` L1035-1070 |
| **CRPS** (Gaussian) | Closed-form for Gaussian | `metrics.crps_gauss` | `metrics.py` L193-227 |
| **NLL** | -log p(target \| pred) | `metrics.nll` | `metrics.py` L166-190 |

#### CRPS Implementation Detail

The `crps_ens` function handles four ensemble size regimes for efficiency:

```python
if num_ens == 1:     → reduces to MAE
elif num_ens == 2:   → simple pairwise diff estimator
elif num_ens < 10:   → rank-based O(M·log M) with argsort
else:                → batched rank-based (loop over variables to save memory)
```

---

## 2. Diffusion-LAM: Architecture & Integration Implications

### 2.1 Key Architectural Difference

| Aspect | Graph-EFM (CVAE) | Diffusion-LAM |
|---|---|---|
| Latent space | Single sample z → decode | Iterative denoising over T diffusion steps |
| Sampling | z ~ N(μ, σ) → one forward pass | x_T ~ N(0,I) → T denoising passes |
| Training | ELBO (likelihood + KL) | Score matching / denoising loss |
| Diversity source | Different z samples | Different noise realizations |
| AR integration | Each AR step: sample z, decode | Each AR step: denoise T times |

### 2.2 Shared AR Forecaster Logic

Despite different step predictors, both models share the same AR loop:

```python
# This is IDENTICAL for both CVAE and Diffusion:
for t in range(pred_steps):
    pred_state = step_predictor(prev_state, prev_prev_state, forcing)
    new_state = boundary_mask * true_state + interior_mask * pred_state
    prev_prev_state = prev_state
    prev_state = new_state
```

This is the **core modularity argument for Issue #49**:

```
┌────────────────┐     ┌──────────────────────────┐
│  AR Forecaster │────→│     StepPredictor        │
│  (shared loop) │     │                          │
│                │     │  ┌────────────────────┐  │
│  unroll_pred() │     │  │ CVAEStepPredictor  │  │
│  boundary_mask │     │  │ (z → decode)       │  │
│  state_updates │     │  └────────────────────┘  │
│                │     │  ┌────────────────────┐  │
│                │     │  │ DiffusionPredictor │  │
│                │     │  │ (denoise T steps)  │  │
│                │     │  └────────────────────┘  │
│                │     │  ┌────────────────────┐  │
│                │     │  │ DeterministicPred  │  │
│                │     │  │ (encode-proc-dec)  │  │
│                │     │  └────────────────────┘  │
└────────────────┘     └──────────────────────────┘
```

### 2.3 What the Refactored Architecture Must Support

| Capability | Deterministic | CVAE (Graph-EFM) | Diffusion |
|---|---|---|---|
| `predict_step(x_{<t})` | Single forward pass | Prior sample + decode | T denoising passes |
| `predict_step_vi(x_{≤t})` | N/A | Encoder sample + decode | Conditioned denoising |
| Training loss | MSE/MAE | ELBO + optional CRPS | Score matching |
| Ensemble generation | N/A | S × prior sampling | S × noise realizations |
| Per-step overhead | 1 forward pass | 1 encode + 1 decode | T denoising passes |

---

## 3. Summary: Paper Equations → Code Functions

```
Paper Equation                     Code Function
─────────────────────────────────  ─────────────────────────────────
p(z|x_{<t})                        prior_model(grid_prev_emb, ...)
q(z|x_{≤t})                        encoder(grid_current_emb, ...)
p(x_t|z, x_{<t})                   decoder(grid_prev_emb, z, ...)
z ~ p(z|x_{<t})                    prior_dist.rsample()
z ~ q(z|x_{≤t})                    var_dist.rsample()
E_q[log p(x|z)]                    estimate_likelihood()
KL(q||p)                           kl_divergence(var_dist, prior_dist)
ELBO                               mean_likelihood - mean_kl
Loss = -ELBO                       -mean_likelihood + β·mean_kl
x̂_t = f(z, x_{<t})                sample_next_state(pred_mean, std)
{x̂^(s)}_{s=1}^S                    sample_trajectories(..., S)
CRPS                               metrics.crps_ens()
Spread/Skill                       log_spsk_ratio()
embed(x_{<t})                      grid_prev_embedder (prior path)
embed(x_{≤t})                      grid_current_embedder (encoder path)
```

---

## 4. Critical Mappings for Proposal Writing

When writing the GSoC proposal, i have to use these mappings:

| Proposal Section | Paper Reference | Code Reference | Key Insight |
|---|---|---|---|
| Problem statement | §1 of Graph-EFM | `ar_model.py` (772-line monolith) | All concerns mixed into one class |
| Why modularity matters | §3 CVAE + Diffusion-LAM | `GraphEFM` bypasses `BaseGraphModel` | Different models need different training loops but same AR rollout |
| Proposed `StepPredictor` interface | §3.1 decode step | `predict_step` in both branches | Abstract method with same signature works for both |
| Proposed `Forecaster` class | §3.3 ensemble generation | `unroll_prediction` method | Identical AR loop shared across all models |
| Metrics extraction | §4.1 evaluation | `aggregate_and_plot_metrics` | Currently mixed into Lightning hooks |
| Training strategy extraction | §3.2 ELBO | `training_step` in GraphEFM vs ARModel | Completely different loss strategies |

---
