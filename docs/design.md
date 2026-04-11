# memory-bench: Design Document

## Motivation

Every memory-augmented LLM paper benchmarks against a different baseline, at a different scale, on different data. The three major 2025 surveys (arXiv:2508.10824, 2509.18868, 2404.13501) organize the literature taxonomically but run zero controlled experiments. Titans (Google, 2025) compared ~6 methods but omitted persistent memory tokens. No plug-and-play harness exists where you can swap memory mechanisms into the same transformer backbone and compare fairly.

memory-bench fills this gap: a controlled empirical study of memory mechanisms at GPT-2 scale, built on Karpathy's nanochat as the backbone.

## Research Questions

1. Which memory mechanisms provide the largest BPB improvement per parameter at small scale (~286M params)?
2. **Where in the context does memory help?** BPB-by-position analysis reveals whether mechanisms improve early, mid, or late positions -- the most actionable finding for practitioners.
3. What is the efficiency-quality Pareto frontier across memory types (BPB vs FLOPs, BPB vs VRAM)?
4. Which mechanisms improve long-range retrieval (NIAH) vs general language modeling (BPB)?

## The Backbone: nanochat

nanochat is Karpathy's latest GPT codebase, successor to nanoGPT. It trains a competitive language model in ~30 minutes on 8xH100.

**Model architecture:**
- 12 layers, 768 hidden dim, 6 attention heads (GQA)
- Sliding window attention (pattern "SSSL": 3 sliding + 1 long per group of 4)
- QK normalization with 1.2x sharpening (softmax temperature)
- RoPE positional embeddings
- Value embeddings (ResFormer-style residual on V)
- "Smear" predecessor mixing (gate * prev_token blended into current)
- Backout residual (subtract mid-layer activation * lambda)
- ReLU^2 MLP (4x expansion)
- Total: 286M parameters

**Training:**
- Data: ClimbMix-400B (curated multilingual web text, March 2026)
- Tokenizer: rustbpe BPE, 32,768 vocab
- Optimizer: Muon (backbone) + AdamW (embeddings, head, memory params)
- LR: linear warmup 250 steps, cosine decay
- Batch size: 524,288 tokens
- Training: 2,520 steps = 1.32B tokens
- Hardware: 8xH100 80GB, DDP
- Throughput: ~4M tok/s, ~40% MFU, ~130ms/step

**Why nanochat:**
- Open, maintained, well-engineered by Karpathy himself
- Competitive BPB (parameter golf leaderboard)
- Clean `GPTConfig` + `GPT` interface makes mechanism injection straightforward
- Community recognition ensures attention to results

## Mechanism Design

### The MemoryModule Interface

All mechanisms implement the same interface, ensuring fair comparison:

```python
class MemoryModule(ABC):
    def wrap_model(self, model: GPT, config: GPTConfig) -> GPT
    def extra_param_groups(self) -> list[dict]
    def on_segment_boundary(self, state) -> state
    def reset(self)
    @property
    def name(self) -> str
    @property
    def num_memory_params(self) -> int
    @property
    def requires_segments(self) -> bool
```

`wrap_model` modifies the model in-place (replacing layers, adding parameters) and returns it. The training harness then merges `extra_param_groups` into the optimizer. This design means mechanism code never touches the training loop -- clean separation.

### 1. Persistent Memory Tokens

**What it does:** Adds M learned KV pairs to every attention layer. These are position-agnostic (no RoPE) and act as static background knowledge.

**How it works:**
- `wrap_model` replaces each `CausalSelfAttention` with `PersistentMemoryAttention`
- At each layer: K_full = [K_mem ; K_tokens], V_full = [V_mem * scale ; V_tokens]
- Attention mask allows tokens to attend to all memory positions + causal on token positions
- V_scale starts at 0.01 (zero-init residual), learned per-layer
- Memory keys get RMSNorm + 1.2x scaling to match token keys

**Parameters:** M=32 tokens * 12 layers * 2 (K+V) * n_kv_head * head_dim + 12 scale params

**Key insight:** Simplest mechanism -- no recurrence, no inner loop. If persistent memory helps, it proves the model benefits from additional capacity in the KV space.

### 2. Recurrent Memory Transformer (RMT)

**What it does:** Splits sequences into fixed segments. M memory tokens carry hidden states between segments via a learned projection.

**How it works:**
- `wrap_model` registers M memory embeddings + a learned d->d projection
- Training loop detects `requires_segments=True` and splits sequences
- For each segment: prepend memory hidden states, run full transformer, extract new memory states
- Memory projection: `new_mem = memory_proj(raw_memory)`, initialized near 0.5*I (mild damping)
- Truncated BPTT: gradients flow back through `bptt_depth` segment boundaries (default=2)
- Memory state is detached between segments beyond BPTT depth

**Parameters:** M * d (memory embeddings) + d^2 + d (projection) = ~17K for M=16

**Key insight:** RMT is the only mechanism that changes the training loop structure (segmented forward). This makes it the hardest to implement correctly and the most likely to have subtle bugs. We found and fixed multiple (non-contiguous tensor crash, memory params not in model.parameters(), etc).

### 3. TTT-Linear (Test-Time Training)

**What it does:** Replaces one transformer layer with an inner linear model W that updates itself per-token via gradient descent on a self-supervised reconstruction loss.

**How it works:**
- `wrap_model` replaces layer at `layer_idx` (default: n_layer//2) with `TTTLinearLayer`
- Inner model W starts as 0.01 * I, updated per chunk via the dual form:
  - E = K @ W^T - V (reconstruction error)
  - eta = softplus(W_eta @ x) (learned per-token, per-head learning rate)
  - A = tril(Q @ K^T) (causal attention matrix)
  - Z = Q @ W^T - A @ (eta * E) (dual form output)
  - W -= (1/C) K^T @ (eta * E) (chunk update)
- LaCT extensions: L2 weight normalization after each chunk, output RMSNorm
- Replaces both attention AND MLP in the target block (TTTBlock)

**Parameters:** LR projection (d * n_kv_head + bias), output norm, init scale = ~590K

**Key insight:** TTT-Linear is mathematically equivalent to DeltaNet when chunk_size=1 and without LayerNorm (Yang & Katharopoulos, 2026). With chunks, it trades off exact per-token updates for GPU parallelism.

### 4. Gated DeltaNet

**What it does:** Replaces softmax attention with linear attention + delta rule. Maintains a state matrix S updated per-token.

**How it works:**
- `wrap_model` replaces attention at specified layers with `GatedDeltaNetAttention`
- Short convolution (kernel=4) on Q, K, V for local context mixing
- L2 normalization on Q, K (required for delta rule stability -- note: NOT the same as nanochat's RMSNorm + 1.2x)
- Gated delta rule recurrence:
  - alpha = exp(logsigmoid(W_alpha @ x)) (per-dimension decay)
  - beta = sigmoid(W_beta @ x) (write gate)
  - S = alpha * S - beta * k * (k^T * S) + beta * k * v^T (erase + write)
  - o = q^T * S (read)
- FLA Triton kernels for GPU-efficient chunk-parallel training
- Pure PyTorch fallback for CPU testing

**Parameters:** Beta projection, alpha projection, output gate, 3 short convolutions, GroupNorm = ~400K

**Key insight:** DeltaNet is the only mechanism that uses a fundamentally different attention computation (linear vs softmax). It also requires `--no-compile` because FLA's Triton kernels are incompatible with torch.compile.

### Design decisions on QK preprocessing

A note on fairness: nanochat's baseline attention applies RMSNorm + 1.2x scaling to Q and K. This is a softmax temperature tweak. Each mechanism uses its own architecture-specific preprocessing:

- **Persistent Memory:** RMSNorm + 1.2x (matches baseline, since it uses SDPA)
- **RMT:** RMSNorm + 1.2x in the main transformer layers (memory tokens get dummy IDs)
- **TTT-Linear:** RMSNorm only (the 1.2x is softmax-specific; TTT uses tril(QK^T) without softmax)
- **DeltaNet:** L2 normalization (required by the delta rule; ||k||=1 keeps the erase term stable)

This is intentional: forcing all mechanisms to use the same preprocessing would break DeltaNet's stability and misapply a softmax-specific tweak to linear attention.

## Evaluation

### BPB (bits per byte)

Primary metric. Computed as total_nats / (ln2 * total_bytes) where total_bytes accounts for the variable byte length of each token. This is tokenizer-invariant and directly comparable across models.

For RMT, a dedicated `evaluate_bpb_segments` function processes validation data in segments with memory carry-over, matching the training forward path.

### BPB by position

The most informative analysis. We bucket the 2048-position context into 32 bins and compute mean BPB per bin. This reveals:
- How much the model benefits from context at each position
- Whether memory mechanisms shift the curve (steeper improvement = better memory utilization)
- Position-specific failure modes (e.g., does RMT's segment boundary create a BPB spike?)

### Needle-in-a-Haystack (NIAH)

Synthetic retrieval task: insert a 6-digit passkey at a random position in filler text, ask the model to retrieve it. Tests whether memory mechanisms improve long-range recall beyond what BPB captures.

Protocol: 5 context lengths x 5 passkey positions x 100 trials per condition. Reports accuracy heatmap.

### Synthetic tasks (associative recall, copy, selective copy)

Additional probes for specific memory capabilities. Not run by default (require inference engine). Available for post-hoc analysis on saved checkpoints.

## Experimental Controls

**Fair comparison:** Same model init (per seed), same data order, same optimizer, same LR schedule, same training steps. The only variable is the memory mechanism.

**Seed averaging:** 4 seeds (42, 67, 1337, 3141) to reduce variance. Report mean +/- std with paired t-tests.

**Parameter accounting:** Each mechanism adds a small number of parameters (0.01% to 0.2% of baseline). Parameter counts, FLOPs per step, total FLOPs, and peak VRAM are logged in each result JSON.

**Checkpointing:** Model checkpoints saved after training for post-hoc analysis (NIAH, synthetic tasks, BPB-by-position at different contexts).

## Test Suite

154 tests across 13 files, covering:

| Test file | Count | What it tests |
|-----------|-------|---------------|
| test_mechanisms.py | 20 | Wrap/forward smoke tests for all mechanisms |
| test_numerical.py | 15 | Reference implementations, mathematical properties |
| test_integration.py | 13 | Seed reproducibility, RMT train/eval consistency, GQA |
| test_synthetic.py | 12 | Prompt generation, BPB-by-position correctness |
| test_seed_e2e.py | 5 | Exact train.py initialization sequence |
| test_cli_args.py | 14 | Every mechanism's CLI args plumbed to model |
| test_regression.py | 11 | Recorded loss baselines for all mechanisms (2% rtol) |
| test_niah_generation.py | 27 | NIAH prompt generation, all edge cases |
| test_optimizer_surgery.py | 22 | No stale/orphan/duplicate params, gradient flow |
| test_evaluate_bpb_segments.py | 4 | RMT BPB math matches manual computation |
| test_attention_fallback.py | 13 | FA3 vs SDPA path equivalence |
| test_engine.py | 6 | KV cache, generation engine |
| conftest.py | -- | Seed isolation autouse fixture |

Run with: `pytest tests/ -v` (~30s on CPU).

## Dependencies

- `nanochat` (git submodule, Karpathy's transformer codebase)
- `torch >= 2.9` (CUDA 12.8 for H100)
- `fla-core >= 0.4` (Gated DeltaNet Triton kernels, MIT license)
- `rustbpe` (fast BPE tokenizer, from nanochat)
- `tokenizers` (HuggingFace tokenizers, for initial setup)
- `matplotlib` (plot generation)
- `wandb` (optional, disabled by default in benchmark runs)

## Future Work

- **Parameter-matched baselines:** Adjust base model dimension so base + memory = constant total params
- **Multi-scale:** d6, d12, d24 to test whether memory benefit grows with depth
- **Multi-needle NIAH:** Multiple passkeys, paraphrased retrieval (vanilla NIAH is largely solved)
- **Forgetting curves:** Train, inject new information, measure how quickly it's forgotten
- **Hybrid mechanisms:** Combine persistent memory + DeltaNet in the same model
