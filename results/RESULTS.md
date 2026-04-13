# memory-bench: Results Summary

**42 runs** across 3 context lengths (2048, 4096, 8192), 5 mechanisms, 4 seeds.

## Setup

| Parameter | Value |
|-----------|-------|
| Architecture | 12-layer GPT (nanochat), 286M params |
| Attention | SSSL pattern: 3/4 layers short window (T/4), 1/4 global |
| Control | Global-attention baseline (all layers full context) |
| Context lengths | 2048, 4096, 8192 tokens |
| Seeds | 42, 67, 1337, 3141 |
| Hardware | 8×H100 SXM, 53 GPU-hours total |

## Results

### BPB by Condition and Context Length

| Context | Local (SSSL) | Global (full attn) | Persistent Memory | PM vs Local | PM vs Global |
|---------|-------------|-------------------|-------------------|-------------|-------------|
| 2048 | 0.84599 ± 0.00053 (n=5) | 0.84554 ± 0.00018 (n=3) | 0.84420 ± 0.00063 (n=4) | **-1.79 mBPB** | **-1.34 mBPB** |
| 4096 | 0.84712 ± 0.00081 (n=3) | 0.84654 ± 0.00079 (n=3) | 0.84529 ± 0.00010 (n=3) | **-1.82 mBPB** | **-1.24 mBPB** |
| 8192 | 0.84932 ± 0.00062 (n=3) | 0.84870 ± 0.00068 (n=3) | 0.84763 ± 0.00052 (n=3) | **-1.69 mBPB** | **-1.06 mBPB** |

*Persistent Memory beats both local and global baselines at every context length.*

### All Mechanisms at T=2048

| Mechanism | Mean BPB | Δ vs Baseline | p-value | Seeds |
|-----------|----------|--------------|---------|-------|
| Baseline | 0.84599 | — | — | 5 |
| Persistent Memory | 0.84420 | **-1.79 mBPB** | 0.002 | 4 |
| RMT | 0.88524 | +39.25 mBPB | <0.001 | 4 |
| TTT-Linear | 0.85198 | +6.00 mBPB | <0.001 | 4 |
| Gated DeltaNet | 0.84734 | +1.35 mBPB | 0.008 | 4 |

## Positional Context Deficit

The **positional context deficit** measures how much local SSSL attention degrades prediction at each position relative to global attention:

$$D(p, T) = \text{BPB}_{\text{local}}(p, T) - \text{BPB}_{\text{global}}(p, T)$$

**Memory gain** measures how much a mechanism recovers:

$$G(p, T) = \text{BPB}_{\text{local}}(p, T) - \text{BPB}_{\text{memory}}(p, T)$$

| Context | Mean Deficit | Mean Gain (PM) | Closure | Gain-Deficit ρ |
|---------|-------------|---------------|---------|----------------|
| 2048 | +0.00053 | +0.00300 | 0.268 | -0.834 (p=0.0000) |
| 4096 | +0.00059 | +0.00258 | 2.134 | +0.488 (p=0.0000) |
| 8192 | +0.00138 | +0.00290 | 1.588 | +0.425 (p=0.0000) |

### Falsification Checks

- **Deficit grows with context**: PASS (T=2048: 0.00053, T=4096: 0.00059, T=8192: 0.00138)

## Figures

### Local vs Global vs Persistent Memory
![Local vs Global vs Persistent Memory](results/figures/fig_three_condition.png)

### BPB vs Context Length (crossover analysis)
![BPB vs Context Length (crossover analysis)](results/figures/fig_crossover.png)

### Positional Context Deficit Map
![Positional Context Deficit Map](results/figures/fig_deficit_map.png)

### Persistent Memory: Deficit and Closure
![Persistent Memory: Deficit and Closure](results/figures/fig_deficit_closure_persistent.png)

### Closure vs Context Length
![Closure vs Context Length](results/figures/fig_closure_vs_context.png)

### BPB by Position (multi-context grid)
![BPB by Position (multi-context grid)](results/figures/fig_bpb_position_grid.png)

### Mechanism Comparison at T=2048
![Mechanism Comparison at T=2048](results/figures/fig1_bpb_comparison.png)

### Compute Efficiency
![Compute Efficiency](results/figures/fig_compute_efficiency.png)
