# Pre-Run Memo

**Title**
**When Does Memory Pay Off? A Deficit-Based Evaluation of Memory in Context-Restricted Transformers**

**Objective**
This study tests whether restricted attention induces a measurable positional context deficit relative to a matched global-attention reference, and whether explicit memory closes enough of that deficit to justify its overhead.

**Why this matters**
Aggregate BPB alone cannot distinguish between three cases:

1. there is no meaningful access problem,
2. there is an access problem but memory does not repair it,
3. there is an access problem and memory repairs it enough to matter.

This study is designed to separate those cases.

**Research questions**
RQ1. Does the local-attention architecture incur a measurable positional deficit relative to a matched global-attention reference?
RQ2. Does that deficit grow with context length and/or concentrate in late sequence positions?
RQ3. Does persistent memory preferentially close that deficit in the regions where it is largest?
RQ4. When does deficit closure outweigh memory overhead on aggregate BPB and compute efficiency?

**Experimental design**
For each context length (T in {2048, 4096, 8192}), train:

* local-attention baseline,
* matched global-attention reference,
* persistent-memory model on the local-attention backbone.

Training settings are matched as closely as possible: model size, optimizer family, total token budget, data pipeline, and evaluation protocol. The primary manipulated factor is context access pattern.

**Core definitions**
Let B_m(T,b) be BPB for model m at context length T and position bucket b.

Positional context deficit:
  D(T,b) = B_local(T,b) - B_global(T,b)

Memory gain:
  G(T,b) = B_local(T,b) - B_memory(T,b)

Regional closure over region R:
  C(T,R) = [B_local(T,R) - B_memory(T,R)] / [B_local(T,R) - B_global(T,R)]

where B_bar(T,R) is the token-weighted average BPB in region R. Closure is interpreted only when the denominator is positive.

**Primary endpoint**
Late-region closure at T=8192, with the late region defined as the final quartile of sequence positions.

This is the main test of whether memory repairs the deficit where restricted attention should be most vulnerable.

**Secondary endpoints**
Late-minus-early closure selectivity:
  S(T) = C(T, R_late) - C(T, R_early)

Growth of late-region deficit with context length.
Per-position deficit and gain curves.
Aggregate validation BPB.
Compute efficiency, such as BPB gain or deficit closure per GPU-minute.
Synthetic long-range recall / retrieval probes with dependency length beyond the local access range.

**Hypotheses**
H1. The local model has positive positional context deficit in late positions relative to the global reference.
H2. Late-position deficit increases with context length.
H3. Persistent memory closes a positive fraction of the late-position deficit.
H4. Closure is selective: larger in late positions than early positions.
H5. On synthetic probes, memory advantage appears primarily when relevant information lies beyond the easy local-access regime.

**Falsification conditions**
If late-position deficit is negligible even at long context, then restricted attention is not the bottleneck in this regime.
If deficit exists but memory does not close it, then persistent memory is not an effective repair mechanism here.
If memory gains are uniform rather than concentrated in high-deficit regions, the mechanism story is weak.
If synthetic probes show no beyond-window advantage, the memory story is further weakened.

**Analysis plan**
Primary statistical unit: seed-level regional metrics, not individual buckets.
Per-position curves are descriptive and mechanistic, but not the sole inferential basis.
Report seed means with uncertainty intervals.
Use paired comparisons across matched seeds where possible.
Avoid bucket-wise significance testing; that is noisy and easy to overinterpret.
Treat correlation between deficit and gain as supporting evidence, not the headline claim.

**Soft kill gates**
After the first 4096 local/global pair, check whether late-region deficit is materially above zero. If not, the 8192 expansion becomes lower priority.
After the first 4096 full triplet, check whether persistent memory shows positive late-region gain or probe advantage. If neither appears, proceed to 8192 only if the deficit itself is clearly growing.
Do not continue extra exploratory branches unless one of these is alive: real deficit, real closure, or real probe advantage.

**Expected outcomes and interpretation**
If deficit is absent: local attention is already sufficient here. Useful negative result.
If deficit is present but closure is weak: memory solves little of the real problem.
If deficit is present and closure is strong but aggregate BPB still loses: memory targets the right problem but too expensively.
If deficit is present, closure is strong, and aggregate BPB improves: strongest result.

**Deliverables**
A reproducible benchmark extension.
Three core figures: deficit map, closure map, synthetic probe curves.
A short technical note / workshop-style paper.
Clean analysis scripts and result JSONs.
A practical decision rule for when memory is worth its cost.
