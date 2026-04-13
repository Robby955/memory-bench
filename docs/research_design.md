# Positional Context Deficit: Research Design Document

## Thesis Statement

We study whether explicit memory mechanisms recover performance specifically where
restricted-context attention architectures are most impaired. We define a positional
context deficit as the per-position performance gap between a local-attention model
and a matched global-attention reference, then test whether persistent memory
preferentially closes this deficit as sequence length grows.

---

## 1. Intellectual Foundation

### 1.1 The Analogy from Survey Statistics

The framework draws on the structure of small area estimation (SAE), specifically
the Fay-Herriot model (Fay & Herriot, 1979; Rao & Molina, 2015). In SAE:

- **Direct estimates** at each area have precision proportional to local sample size
- **Model-based estimates** borrow strength from auxiliary covariates across areas
- The **composite estimator** is a precision-weighted combination:
  `theta_hat_i = gamma_i * y_i + (1 - gamma_i) * x_i' * beta`
  where `gamma_i = sigma_u^2 / (sigma_u^2 + psi_i)` and `psi_i` is the local
  sampling variance

The mapping to our setting:

| SAE Concept | Our Framework | Formal Validity |
|-------------|---------------|-----------------|
| Area i | Position region p | Strong: both define subpopulations for local estimation |
| Direct estimate y_i | Local attention prediction at p | Strong: based purely on locally available data |
| Sampling variance psi_i | Prediction uncertainty from limited context window | Moderate: psi_i is known in SAE, must be estimated here |
| Synthetic estimate | Memory-augmented prediction | Strong: uses auxiliary information beyond local window |
| Shrinkage factor gamma_i | Adaptive combination weight | Strong: the formula is a theorem, not a metaphor |
| Borrowing strength | Cross-position information transfer via memory | Moderate: same statistical effect, different mechanism |

**Where the analogy is genuine (cite-worthy):**
- The composite estimator structure (Fay & Herriot 1979) literally applies
- James-Stein (1961) guarantees shrinkage dominates direct estimation when m >= 3
- Heteroscedastic empirical Bayes (Stephens 2017, Xie et al. 2012) handles
  position-varying precision

**Where the analogy requires care:**
- psi_i is not known — must be operationally defined (we use the deficit)
- Positions are not independent — adjacent buckets share context (see Section 4)
- The "synthetic estimate" in SAE is linear; memory mechanisms are not

**Recommended usage:** Use SAE as structural motivation and cite Fay-Herriot for
the composite estimator. Do NOT claim this IS small area estimation. Frame it as:
"inspired by the borrowing-strength framework in survey statistics, we define..."

### 1.2 Why the Deficit Exists: Theoretical Grounding

The positional context deficit has formal theoretical support:

1. **Information dilution in stacked sliding windows** (Xiao 2025): The effective
   influence of token j on token i decays exponentially with the number of hops
   through the window, not linearly. The theoretical receptive field of L*W vastly
   overstates the empirical receptive field.

2. **Architectural position bias** (Chowdhury 2026, "Lost in the Middle at Birth"):
   The U-shaped position bias exists at initialization before training, arising from
   causal masking (primacy) and residual connections (dead zone in the middle).

3. **Residual connection barrier** (Herasimchyk et al. 2026, Wu et al. ICML 2025):
   Residual connections create an exponential barrier to distant information flow,
   provably producing U-shaped attention at finite depth.

4. **Empirical evidence** (Liu et al. 2023, "Lost in the Middle"): The U-shaped
   performance curve is consistent across GPT-3.5, Claude, and long-context models.
   Models retrieve best from beginning and end, worst from middle.

---

## 2. Core Metrics

### 2.1 Definitions

Let BPB_A(p, T) denote the mean bits-per-byte at position bucket p for model A
trained with context length T. All models share architecture, optimizer, and token
budget; they differ only in attention pattern and/or memory mechanism.

**Positional context deficit:**
```
deficit(p, T) = BPB_local(p, T) - BPB_global(p, T)
```
where BPB_local uses window pattern SSSL and BPB_global uses window pattern L.
Positive deficit = local attention underperforms at position p.

**Memory gain:**
```
gain(p, T) = BPB_local(p, T) - BPB_memory(p, T)
```
where BPB_memory uses the same SSSL pattern plus a memory mechanism.
Positive gain = memory helps at position p.

**Deficit closure:**
```
closure(p, T) = gain(p, T) / max(deficit(p, T), epsilon)
```
with epsilon = 0.001 BPB. Closure measures the fraction of the context deficit
recovered by the memory mechanism.

- closure ~ 1.0: memory fully closes the gap
- closure ~ 0.0: memory does not help where attention is impaired
- closure > 1.0: memory exceeds global-attention performance (overshooting)
- Undefined where deficit < epsilon (local attention is already sufficient)

### 2.2 What These Metrics Are NOT

- **deficit is NOT "ground truth information access."** Global attention is a
  different inductive bias, not a pure information ceiling. It is an operational
  proxy for the upper bound on what full-context access provides.

- **closure is NOT causal proof.** High closure at high-deficit positions is
  consistent with the hypothesis that memory compensates for attention limitations,
  but correlation between deficit and gain does not establish causation. The
  synthetic probes (Section 5) provide the controlled manipulation.

### 2.3 Aggregate Metrics (per context length T)

- Mean deficit: `D(T) = mean_p(deficit(p, T))`
- Mean gain: `G(T) = mean_p(gain(p, T))`
- Mean closure: `C(T) = mean_p(closure(p, T))` over buckets where deficit > epsilon
- Deficit-gain correlation: Spearman rho between deficit(p) and gain(p)

### 2.4 Hypotheses

**H1 (deficit grows with context):** D(8192) > D(4096) > D(2048).
Rationale: at longer context, the SSSL window covers a smaller fraction of the
sequence, increasing the number of positions with indirect-only access.

**H2 (gain concentrates at high-deficit positions):** The Spearman correlation
between deficit(p, T) and gain(p, T) is positive and significant for T >= 4096.

**H3 (closure rises with context):** C(8192) > C(4096) > C(2048).
Memory becomes more useful as the deficit it addresses grows.

**Falsification conditions:**
- H1 fails: deficit does not grow systematically with context length
- H2 fails: gain is uniform across positions (memory provides constant overhead
  reduction, not position-targeted compensation)
- H3 fails: closure is flat or decreasing (memory does not selectively target
  high-deficit regions more at longer context)

---

## 3. Experimental Design

### 3.1 Run Matrix

| Context | Model | Window | Seeds | Purpose |
|---------|-------|--------|-------|---------|
| 2048 | Baseline | SSSL | 4 | DONE (v1 results) |
| 2048 | Baseline | L | 3 | Reference: information ceiling at 2048 |
| 2048 | Persistent | SSSL | 4 | DONE (v1 results) |
| 4096 | Baseline | SSSL | 3 | Local baseline at 4096 |
| 4096 | Baseline | L | 3 | Reference at 4096 |
| 4096 | Persistent | SSSL | 3 | Memory mechanism at 4096 |
| 8192 | Baseline | SSSL | 3 | Local baseline at 8192 |
| 8192 | Baseline | L | 3 | Reference at 8192 |
| 8192 | Persistent | SSSL | 3 | Memory mechanism at 8192 |
| **Total new** | | | **24** | |

### 3.2 Controls

- Same architecture family (nanochat GPT, depth 12)
- Matched total token budget (524,288 tokens/batch)
- Matched optimizer (Muon + AdamW) and hyperparameters
- Context length as the primary independent variable
- Window pattern as the secondary independent variable (SSSL vs L)
- Mechanism (none vs persistent) as the tertiary variable

**Admitted confounds:** Varying context length also changes:
- Number of optimization steps (same token budget / larger sequences = fewer steps)
- Gradient noise profile
- Data packing statistics (document boundaries, effective truncation)
- SSSL window behavior (T/4 window at T=8192 is 2048 tokens vs 512 at T=2048)

These are documented as limitations, not controlled away.

### 3.3 Why Only Persistent Memory

From v1 results (4-seed, 2048 context):

| Mechanism | vs Baseline | p-value | Decision |
|-----------|-------------|---------|----------|
| Persistent Memory | -0.0017 | 0.012 | KEEP (only winner) |
| Gated DeltaNet | +0.0014 | 0.020 | CUT (overhead > benefit) |
| TTT-Linear | +0.0061 | 0.001 | CUT (significant penalty) |
| RMT | +0.0393 | <0.001 | CUT (catastrophic penalty) |

Running all 5 mechanisms would triple compute cost for no additional insight.
Persistent memory is the only mechanism with a plausible path to positive closure.

---

## 4. Statistical Analysis Plan

### 4.1 Primary Analysis: Deficit Closure

For each context length T, compute deficit(p, T), gain(p, T), closure(p, T)
per position bucket. Report mean and standard deviation across seeds.

### 4.2 Handling Autocorrelation

Adjacent position buckets share overlapping context and underlying text continuity.
Naive significance tests on bucketed BPB overstate significance.

**Approach (ranked by defensibility):**

1. **Block bootstrap** (primary): Resample contiguous blocks of positional buckets.
   Block size chosen via autocorrelation function of the BPB series — use the
   smallest block where lag-k autocorrelation drops below 0.05.
   Precedent: Meta's blockwise bootstrap for ASR evaluation.

2. **Newey-West HAC standard errors** (for regression): When testing "does
   delta-BPB vary with position?", use OLS with Newey-West covariance.
   Available in `statsmodels.stats.sandwich_covariance`.

3. **Effective degrees of freedom** (simplest): Compute lag-1 autocorrelation r1,
   apply N_eff = N * (1 - r1) / (1 + r1) (Bretherton et al. 1999). Use N_eff
   instead of N for all t-tests.

**Report all three.** If they agree, the result is robust. If they disagree,
report the most conservative.

### 4.3 Secondary Analysis: Difference-in-Differences

Partition position buckets into two groups:
- **Near:** positions 0 to T/4 (within the SSSL window for all layers)
- **Far:** positions T/4 to T (beyond the short-window boundary)

The DID estimator:
```
DID = [BPB_memory(far) - BPB_local(far)] - [BPB_memory(near) - BPB_local(near)]
```

If memory specifically helps beyond the window boundary, DID < 0.

The parallel trends assumption is testable: check that BPB_local and BPB_memory
have similar position-dependent trends in the "near" region.

### 4.4 Tertiary Analysis: Regression Discontinuity at Window Boundary

The SSSL short window size at context T is approximately T/4 (ceiled to FA3 tile
size). This creates a quasi-discontinuity: positions before T/4 have direct
attention access for most layers; positions after T/4 rely on indirect propagation.

**Fuzzy RD design:** Treatment intensity (fraction of layers with window coverage)
changes at the boundary. Use local linear regression following Cattaneo & Titiunik
(2022) to estimate the jump in delta-BPB at the boundary.

This is a secondary analysis. The deficit/closure framework is primary.

---

## 5. Synthetic Probes (Controlled Manipulation)

### 5.1 Purpose

Aggregate BPB analysis shows *where* memory helps. Synthetic probes show *that*
the gain is specifically from cross-window information transfer, not from
statistical artifacts of training dynamics.

### 5.2 Probe Suite

1. **Token recall at distance:** Place a target token at position P, fill with
   natural-text filler, query recall at position P+D. Sweep D from D < T/4 (within
   window) to D > T/4 (beyond window). Memory should help at D > T/4.

2. **Entity attribute tracking:** "Entity X has attribute Y" at early position,
   query "What is X's attribute?" at late position. Distance between introduction
   and query crosses the window boundary.

3. **Cross-boundary associative recall:** Key-value pairs where keys are in one
   window region and queries are in another. Extension of existing AR evaluation.

### 5.3 What Probes Do NOT Prove

Synthetic probes test a specific retrieval capability. They do not prove that
the natural-text BPB gains are caused by the same mechanism. They provide
consistent evidence, not proof. Frame as: "targeted probes confirm that persistent
memory enables cross-window information transfer, consistent with the positional
deficit closure observed on natural text."

---

## 6. Related Work and Novelty

### 6.1 Direct Precedents

**LongPPL** (Fang et al., ICLR 2025): Decomposes perplexity by token type
(key tokens vs non-key tokens that require long-range context). Our approach
decomposes by position, which is complementary. LongPPL identifies *which tokens*
need long-range context; we identify *where in the sequence* models struggle.
Must cite and differentiate clearly.

**"Lost in the Middle"** (Liu et al., TACL 2024): Established position-dependent
evaluation for LLMs. However: measures task accuracy (QA retrieval), not BPB;
evaluates at document level, not token level; does not compare architectural
variants. We bring this insight down to token-level BPB with controlled
architectural comparison.

**Needle-in-a-Haystack** (Kamradt 2024): Position x context heatmaps. Uses
synthetic probes (planted facts), not natural BPB. Our deficit maps use
naturalistic text with no synthetic injection for the primary analysis.

**Forgetting Curve** (Liu et al., EMNLP 2024): Position-resolved memorization
accuracy. Demonstrates that Transformers and SSMs have qualitatively different
forgetting curves. Closest to our position-resolved methodology.

### 6.2 Theoretical Support

**"Lost in the Middle at Birth"** (Chowdhury 2026): U-shaped position bias exists
at initialization. The deficit is partially architectural, not learned.

**"Why Stacking Sliding Windows Can't See Very Far"** (Xiao 2025): Information
dilution + residual barriers cause exponential decay of influence beyond the
window. This is the theoretical mechanism behind our deficit.

**"On the Emergence of Position Bias"** (Wu et al., ICML 2025) and
**"A Residual-Aware Theory"** (Herasimchyk et al. 2026): Formal proofs of
U-shaped attention concentration at finite depth.

### 6.3 Novelty Claim

The first controlled per-position BPB analysis of memory mechanisms under
restricted-context attention, using a deficit/closure framework to measure
whether gains concentrate where the base architecture is most impaired.

No prior work combines:
1. Controlled same-architecture comparison (local vs global vs memory)
2. Per-position BPB decomposition
3. The deficit/closure framework with formal hypothesis testing

---

## 7. Key Citations

### Statistics / SAE
- Fay & Herriot (1979). Estimates of Income for Small Places. JASA.
- James & Stein (1961). Estimation with Quadratic Loss. Berkeley Symposium.
- Efron & Morris (1973). Stein's Estimation Rule. JASA.
- Efron & Morris (1975). Data Analysis Using Stein's Estimator. JASA.
- Rao & Molina (2015). Small Area Estimation, 2nd ed. Wiley.
- Stephens (2017). False Discovery Rates: A New Deal. Biostatistics.
- Xie, Kou & Brown (2012). SURE Estimates for Heteroscedastic Models. JASA.
- Bretherton et al. (1999). Effective Sample Size. J. Climate.
- Afyouni, Smith & Nichols (2019). Effective DoF under Autocorrelation. NeuroImage.
- Cattaneo & Titiunik (2022). Regression Discontinuity Designs. ARE.

### Memory Mechanisms
- Sukhbaatar et al. (2019). Persistent Memory. NeurIPS workshop.
- Bulatov et al. (2022). Recurrent Memory Transformer. NeurIPS.
- Sun et al. (2024). Test-Time Training (TTT). arXiv:2407.04620.
- Yang et al. (2024). DeltaNet. NeurIPS.
- Yang, Kautz & Hatamizadeh (2025). Gated DeltaNet. ICLR.

### Long-Context Evaluation
- Liu et al. (2023). Lost in the Middle. TACL.
- Fang et al. (2025). LongPPL. ICLR.
- Liu et al. (2024). Forgetting Curve. EMNLP.
- Kuratov et al. (2024). BABILong. NeurIPS.
- Kamradt (2024). Needle in a Haystack. GitHub.
- Chowdhury (2026). Lost in the Middle at Birth. arXiv.

### Sliding Window Attention
- Jiang et al. (2023). Mistral 7B. arXiv.
- Xiao (2025). Why Stacking Sliding Windows Can't See Very Far. Blog.
- Beltagy et al. (2020). Longformer. arXiv.
- Zaheer et al. (2020). BigBird. NeurIPS.

### Position Bias Theory
- Wu et al. (2025). On the Emergence of Position Bias. ICML.
- Herasimchyk et al. (2026). Residual-Aware Theory of Position Bias. arXiv.
- Hsieh et al. (2024). Found in the Middle. ACL Findings.

### Statistical Methodology
- Dror et al. (2018). Testing Statistical Significance in NLP. ACL.
- Meta (ASR Blockwise Bootstrap). Meta Research.
- Newey & West (1987). HAC Estimators. Econometrica.
