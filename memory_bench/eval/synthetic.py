"""
Synthetic evaluation tasks for testing memory mechanisms.

These tasks are designed to isolate specific memory capabilities:

1. **Associative Recall**: Present K key-value pairs, then query for a specific
   value. Tests: can the model store and retrieve arbitrary associations?

2. **Multi-Query Associative Recall (MQAR)**: Same as above but with multiple
   queries. Tests: can the model retrieve multiple associations from the same
   context? (From Zoology benchmark, Arora et al. 2024)

3. **Copy Task**: Model must reproduce an input sequence exactly.
   Tests: can the model maintain exact token-level information over distance?

4. **Selective Copy**: Copy only specific marked tokens from a longer sequence.
   Tests: can the model selectively attend to relevant information?

5. **BPB vs Position**: Measure per-position loss on natural text.
   Tests: how does model quality vary with position in the context?
   (Memory mechanisms should improve loss at later positions.)

Protocol:
    All tasks use the model's own tokenizer and generate prompts programmatically.
    For generative tasks (assoc recall, copy), we generate tokens and check
    exact match. For BPB vs position, we compute loss at each position.
"""

import math
import random
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Associative Recall
# ─────────────────────────────────────────────────────────────────────────────

def _generate_kv_pairs(
    tokenizer,
    num_pairs: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Generate random key-value pairs using short words.

    Keys are "key_XX" and values are random 4-digit numbers.
    This ensures unique, unambiguous associations.
    """
    pairs = []
    for i in range(num_pairs):
        key = f"key_{i:02d}"
        value = str(rng.randint(1000, 9999))
        pairs.append((key, value))
    return pairs


def generate_assoc_recall_prompt(
    tokenizer,
    num_pairs: int,
    query_idx: int,
    rng: random.Random,
    with_filler: bool = True,
    filler_tokens: int = 0,
) -> tuple[list[int], str]:
    """Generate an associative recall test prompt.

    Format:
        key_00: 1234. key_01: 5678. key_02: 9012. [filler] What is key_01? Answer:

    Args:
        tokenizer: nanochat tokenizer
        num_pairs: number of key-value pairs to present
        query_idx: which pair to query (0-indexed)
        rng: random generator
        with_filler: whether to add filler text between pairs and query
        filler_tokens: approximate number of filler tokens

    Returns:
        (token_ids, expected_answer) tuple
    """
    pairs = _generate_kv_pairs(tokenizer, num_pairs, rng)

    # Build the context: present all KV pairs
    context_parts = []
    for key, value in pairs:
        context_parts.append(f"{key}: {value}.")
    context = " ".join(context_parts)

    # Optional filler
    filler = ""
    if with_filler and filler_tokens > 0:
        filler_sentences = [
            " The weather is pleasant today.",
            " Many discoveries were made recently.",
            " The sun rises in the east.",
            " Numbers are fundamental to mathematics.",
        ]
        while len(tokenizer.encode(filler)) < filler_tokens:
            filler += rng.choice(filler_sentences)

    # Query
    query_key, expected_value = pairs[query_idx]
    query = f" What is {query_key}? Answer: "

    full_text = context + filler + query
    bos_id = tokenizer.get_bos_token_id()
    tokens = [bos_id] + tokenizer.encode(full_text)

    return tokens, expected_value


def evaluate_associative_recall(
    engine,
    tokenizer,
    num_pairs_list: list[int] = None,
    filler_tokens_list: list[int] = None,
    num_trials: int = 50,
    max_gen_tokens: int = 10,
    seed: int = 42,
) -> dict:
    """Run associative recall evaluation.

    Tests the model's ability to store and retrieve arbitrary key-value
    associations at varying numbers of pairs and filler distances.

    Args:
        engine: nanochat Engine for generation
        tokenizer: nanochat tokenizer
        num_pairs_list: numbers of KV pairs to test
        filler_tokens_list: filler token counts between pairs and query
        num_trials: trials per condition
        max_gen_tokens: max tokens to generate
        seed: random seed

    Returns:
        Dict with accuracy by (num_pairs, filler_tokens)
    """
    if num_pairs_list is None:
        num_pairs_list = [4, 8, 16, 32]
    if filler_tokens_list is None:
        filler_tokens_list = [0, 64, 256, 512]

    rng = random.Random(seed)
    results = {}
    total_correct = 0
    total_trials = 0

    for num_pairs in num_pairs_list:
        results[num_pairs] = {}
        for filler_tokens in filler_tokens_list:
            correct = 0

            for trial in range(num_trials):
                query_idx = rng.randint(0, num_pairs - 1)
                prompt_tokens, expected = generate_assoc_recall_prompt(
                    tokenizer, num_pairs, query_idx, rng,
                    with_filler=filler_tokens > 0,
                    filler_tokens=filler_tokens,
                )

                try:
                    output_tokens, _ = engine.generate_batch(
                        prompt_tokens, num_samples=1,
                        max_tokens=max_gen_tokens, temperature=0, seed=seed + trial,
                    )
                    generated = output_tokens[0][len(prompt_tokens):]
                    generated_text = tokenizer.decode(generated)
                    if expected in generated_text:
                        correct += 1
                except Exception as e:
                    warnings.warn(f"Assoc recall generation failed (pairs={num_pairs}, trial={trial}): {e}")

            accuracy = correct / max(num_trials, 1)
            results[num_pairs][filler_tokens] = accuracy
            total_correct += correct
            total_trials += num_trials

    return {
        "task": "associative_recall",
        "num_pairs": num_pairs_list,
        "filler_tokens": filler_tokens_list,
        "accuracy": {
            str(n_pairs): {str(ft): v for ft, v in inner.items()}
            for n_pairs, inner in results.items()
        },
        "overall_accuracy": total_correct / max(total_trials, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Query Associative Recall (MQAR)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mqar(
    engine,
    tokenizer,
    num_pairs: int = 16,
    num_queries: int = 4,
    num_trials: int = 50,
    max_gen_tokens: int = 10,
    seed: int = 42,
) -> dict:
    """Multi-Query Associative Recall (Zoology benchmark).

    Present N key-value pairs, then ask Q questions about different pairs.
    This tests whether the model can retrieve multiple associations from
    the same context — harder than single-query because the model must
    maintain multiple associations simultaneously.

    Args:
        engine: nanochat Engine
        tokenizer: nanochat tokenizer
        num_pairs: number of KV pairs
        num_queries: number of queries per trial
        num_trials: number of trials
        max_gen_tokens: max generation tokens per query
        seed: random seed

    Returns:
        Dict with per-query and overall accuracy
    """
    rng = random.Random(seed)
    per_query_correct = [0] * num_queries
    total_trials = 0

    for trial in range(num_trials):
        pairs = _generate_kv_pairs(tokenizer, num_pairs, rng)

        # Build context
        context = " ".join(f"{k}: {v}." for k, v in pairs)

        # Select random queries (without replacement)
        query_indices = rng.sample(range(num_pairs), min(num_queries, num_pairs))

        for q_pos, q_idx in enumerate(query_indices):
            query_key, expected = pairs[q_idx]
            full_text = context + f" What is {query_key}? Answer: "
            bos_id = tokenizer.get_bos_token_id()
            tokens = [bos_id] + tokenizer.encode(full_text)

            try:
                output_tokens, _ = engine.generate_batch(
                    tokens, num_samples=1,
                    max_tokens=max_gen_tokens, temperature=0, seed=seed + trial * 100 + q_pos,
                )
                generated = output_tokens[0][len(tokens):]
                if expected in tokenizer.decode(generated):
                    per_query_correct[q_pos] += 1
            except Exception as e:
                warnings.warn(f"MQAR generation failed (trial={trial}, query={q_pos}): {e}")

        total_trials += 1

    return {
        "task": "mqar",
        "num_pairs": num_pairs,
        "num_queries": num_queries,
        "per_query_accuracy": [c / max(total_trials, 1) for c in per_query_correct],
        "overall_accuracy": sum(per_query_correct) / max(total_trials * num_queries, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Copy Task
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_copy(
    engine,
    tokenizer,
    sequence_lengths: list[int] = None,
    num_trials: int = 50,
    seed: int = 42,
) -> dict:
    """Copy task: model must reproduce an input sequence exactly.

    Format:
        Copy the following: 1234 5678 9012 3456. Output:

    Tests exact token-level memory over varying distances.

    Args:
        engine: nanochat Engine
        tokenizer: nanochat tokenizer
        sequence_lengths: number of tokens to copy
        num_trials: trials per length
        seed: random seed

    Returns:
        Dict with accuracy by sequence length
    """
    if sequence_lengths is None:
        sequence_lengths = [4, 8, 16, 32, 64]

    rng = random.Random(seed)
    results = {}

    for seq_len in sequence_lengths:
        correct = 0

        for trial in range(num_trials):
            # Generate random digit sequence
            digits = [str(rng.randint(0, 9)) for _ in range(seq_len)]
            seq_str = " ".join(digits)

            prompt = f"Copy the following sequence exactly: {seq_str}. Output: "
            bos_id = tokenizer.get_bos_token_id()
            tokens = [bos_id] + tokenizer.encode(prompt)

            try:
                output_tokens, _ = engine.generate_batch(
                    tokens, num_samples=1,
                    max_tokens=seq_len * 3,  # generous token budget
                    temperature=0, seed=seed + trial,
                )
                generated = output_tokens[0][len(tokens):]
                generated_text = tokenizer.decode(generated).strip()

                # Check exact match (ignoring whitespace differences)
                expected_digits = digits
                generated_digits = generated_text.split()[:seq_len]
                if generated_digits == expected_digits:
                    correct += 1
            except Exception as e:
                warnings.warn(f"Copy task generation failed (len={seq_len}, trial={trial}): {e}")

        results[seq_len] = correct / max(num_trials, 1)

    return {
        "task": "copy",
        "sequence_lengths": sequence_lengths,
        "accuracy": {str(k): v for k, v in results.items()},
        "overall_accuracy": sum(results.values()) / max(len(results), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Selective Copy
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_selective_copy(
    engine,
    tokenizer,
    context_lengths: list[int] = None,
    num_marked: int = 4,
    num_trials: int = 50,
    seed: int = 42,
) -> dict:
    """Selective copy: copy only marked tokens from a longer sequence.

    Format:
        Sequence: a b [c] d e [f] g h [i] j. Copy only the bracketed items:

    Expected output: c f i

    Tests selective attention: can the model attend to and reproduce only
    the relevant tokens while ignoring distractors?

    Args:
        engine: nanochat Engine
        tokenizer: nanochat tokenizer
        context_lengths: total sequence lengths
        num_marked: number of tokens to mark for copying
        num_trials: trials per condition
        seed: random seed

    Returns:
        Dict with accuracy by context length
    """
    if context_lengths is None:
        context_lengths = [16, 32, 64, 128]

    rng = random.Random(seed)
    results = {}

    for ctx_len in context_lengths:
        correct = 0

        for trial in range(num_trials):
            # Generate random single-digit tokens
            tokens_list = [str(rng.randint(0, 9)) for _ in range(ctx_len)]

            # Mark random positions
            mark_positions = sorted(rng.sample(range(ctx_len), min(num_marked, ctx_len)))
            marked_values = [tokens_list[p] for p in mark_positions]

            # Build sequence with brackets around marked items
            parts = []
            for i, tok in enumerate(tokens_list):
                if i in mark_positions:
                    parts.append(f"[{tok}]")
                else:
                    parts.append(tok)

            prompt = f"Sequence: {' '.join(parts)}. Copy only the bracketed items: "
            bos_id = tokenizer.get_bos_token_id()
            prompt_tokens = [bos_id] + tokenizer.encode(prompt)

            try:
                output_tokens, _ = engine.generate_batch(
                    prompt_tokens, num_samples=1,
                    max_tokens=num_marked * 3,
                    temperature=0, seed=seed + trial,
                )
                generated = output_tokens[0][len(prompt_tokens):]
                generated_text = tokenizer.decode(generated).strip()
                generated_items = generated_text.split()[:num_marked]

                if generated_items == marked_values:
                    correct += 1
            except Exception as e:
                warnings.warn(f"Selective copy generation failed (ctx_len={ctx_len}, trial={trial}): {e}")

        results[ctx_len] = correct / max(num_trials, 1)

    return {
        "task": "selective_copy",
        "context_lengths": context_lengths,
        "num_marked": num_marked,
        "accuracy": {str(k): v for k, v in results.items()},
        "overall_accuracy": sum(results.values()) / max(len(results), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BPB vs Position Analysis
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_bpb_by_position(
    model,
    dataloader,
    token_bytes: torch.Tensor,
    num_steps: int = 100,
    num_buckets: int = 32,
) -> dict:
    """Compute BPB at each position in the context.

    This measures how model quality varies with position. Memory mechanisms
    should show improved BPB at later positions (where the accumulated
    context can help more).

    Instead of per-token loss (noisy), we bucket positions into num_buckets
    bins and report mean BPB per bucket.

    Args:
        model: GPT model
        dataloader: validation data loader
        token_bytes: bytes-per-token tensor for BPB conversion
        num_steps: number of evaluation steps
        num_buckets: number of position buckets

    Returns:
        Dict with per-bucket BPB values
    """
    model.eval()
    device = next(model.parameters()).device

    # Accumulators per position
    total_loss_per_pos = None
    count_per_pos = None

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        B, T = x.shape

        if total_loss_per_pos is None:
            total_loss_per_pos = torch.zeros(T, device=device)
            count_per_pos = torch.zeros(T, device=device)

        # Get logits (model returns logits when called with x only)
        logits = model(x)
        if not (isinstance(logits, torch.Tensor) and logits.dim() == 3):
            raise ValueError(f"Model must return (B, T, V) logits when called without targets, got {type(logits)}")

        # Compute per-position cross-entropy
        # logits: (B, T, V), y: (B, T)
        valid = y != -1  # (B, T)
        y_safe = y.clamp(min=0)  # avoid gather crash on -1
        log_probs = F.log_softmax(logits.float(), dim=-1)
        target_log_probs = log_probs.gather(-1, y_safe.unsqueeze(-1)).squeeze(-1)  # (B, T)
        nll = -target_log_probs * valid.float()  # (B, T) — zeroed at padding positions

        # Convert to BPB: nll / log(2) / bytes_per_token
        if token_bytes is not None:
            bytes_y = token_bytes[y.clamp(0)].float()  # (B, T)
            bpb = nll / (bytes_y.clamp(min=1) * math.log(2))
        else:
            bpb = nll / math.log(2)

        # Accumulate per position
        total_loss_per_pos[:T] += (bpb * valid.float()).sum(dim=0)
        count_per_pos[:T] += valid.float().sum(dim=0)

    # Compute mean BPB per position
    mean_bpb = (total_loss_per_pos / count_per_pos.clamp(min=1)).cpu().numpy()

    # Bucket into num_buckets bins
    T = len(mean_bpb)
    bucket_size = max(T // num_buckets, 1)
    buckets = {}
    for b in range(num_buckets):
        start = b * bucket_size
        end = min(start + bucket_size, T)
        if start >= T:
            break
        bucket_bpb = float(np.mean(mean_bpb[start:end]))
        bucket_center = (start + end) / 2
        buckets[f"{start}-{end}"] = {
            "center_position": bucket_center,
            "bpb": bucket_bpb,
        }

    return {
        "task": "bpb_by_position",
        "num_buckets": len(buckets),
        "sequence_length": T,
        "buckets": buckets,
        "overall_mean_bpb": float(np.mean(mean_bpb)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Window Probes (Direction B: Positional Context Deficit)
# ─────────────────────────────────────────────────────────────────────────────
#
# Loss-based probes for testing whether memory mechanisms enable information
# transfer beyond the SSSL window boundary (T/4). Unlike the generation-based
# probes above, these measure prediction loss at specific positions, producing
# continuous metrics compatible with the deficit/closure framework.
#
# Design principles:
# - Plant novel (random) associations that cannot be predicted from training
# - Fill remaining context with natural-looking procedural filler text
# - Measure cross-entropy loss in bits at positions where the answer appears
# - Report per-distance loss for comparison across models
#
# Reference: docs/research_design.md Section 5 (Synthetic Probes)
# ─────────────────────────────────────────────────────────────────────────────

# Semantically neutral filler sentences for padding between fact and query.
_FILLER_SENTENCES = [
    "The development of new methods remains an active area of research.",
    "Several factors contribute to the overall performance of the system.",
    "This approach has been widely adopted in various applications.",
    "Further analysis is needed to understand the underlying mechanisms.",
    "The results demonstrate the effectiveness of the proposed technique.",
    "Previous work has established several important baselines.",
    "The implementation follows standard best practices for reliability.",
    "Additional experiments were conducted to validate the findings.",
    "The framework provides a flexible foundation for future extensions.",
    "These observations are consistent with theoretical predictions.",
    "Recent advances have significantly improved the state of the art.",
    "The methodology was designed to minimize confounding variables.",
    "Data collection followed established protocols for consistency.",
    "Quality assurance measures were applied throughout the process.",
    "Computational resources were allocated based on task requirements.",
    "The evaluation protocol includes multiple independent measurements.",
]

# Fictional names and organizations (prevents training-data memorization)
_ENTITY_NAMES = [
    "Kelvin", "Mateo", "Priya", "Linnea", "Tariq", "Saoirse", "Dmitri",
    "Yuki", "Amara", "Caspian", "Zuri", "Elowen", "Ravi", "Freya",
    "Idris", "Nadia", "Orion", "Callista", "Boden", "Liora",
]

_FICTIONAL_ORGS = [
    "Nexora", "Veltrix", "Quorin", "Zephyral", "Pyndex", "Calthor",
    "Brionix", "Stelvara", "Kovanti", "Meridex", "Falcuri", "Synthos",
    "Dravion", "Elluvian", "Kromet", "Vaelith", "Tyndall", "Orvisan",
]


def _make_filler_tokens(
    tokenizer, target_length: int, rng: random.Random,
) -> list[int]:
    """Generate filler text of exactly target_length tokens."""
    if target_length <= 0:
        return []
    text = ""
    tokens: list[int] = []
    while len(tokens) < target_length:
        text += " " + rng.choice(_FILLER_SENTENCES)
        tokens = tokenizer.encode(text)
    return tokens[:target_length]


def _compute_loss_at_positions(
    model,
    sequence: list[int],
    target_positions: list[int],
    device: torch.device,
) -> list[float]:
    """Compute cross-entropy loss in bits at specific sequence positions.

    Args:
        model: GPT model returning (B, T, V) logits
        sequence: token IDs of length T
        target_positions: positions in sequence where we measure loss.
            Position p means: loss of predicting sequence[p] given
            context sequence[0:p].
        device: torch device

    Returns:
        Per-position losses in bits (nats / log(2))
    """
    T = len(sequence)
    x = torch.tensor([sequence[:-1]], device=device)
    logits = model(x)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    losses = []
    for pos in target_positions:
        if pos < 1 or pos >= T:
            continue
        nll = -log_probs[0, pos - 1, sequence[pos]].item()
        losses.append(nll / math.log(2))
    return losses


def _default_distances(max_seq_len: int) -> list[int]:
    """Distance sweep points: 3 within SSSL window, 1 at boundary, 3 beyond."""
    T = max_seq_len
    boundary = T // 4
    candidates = sorted(set([
        max(64, T // 16),
        max(128, T // 8),
        max(192, 3 * T // 16),
        boundary,
        min(3 * T // 8, T - 64),
        min(T // 2, T - 64),
        min(3 * T // 4, T - 64),
    ]))
    return [d for d in candidates if 50 < d < T - 50]


@torch.no_grad()
def evaluate_token_recall_at_distance(
    model,
    tokenizer,
    _token_bytes: Optional[torch.Tensor] = None,
    max_seq_len: int = 2048,
    distances: Optional[list[int]] = None,
    num_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Probe 1: Token recall at varying distances from a planted fact.

    Plants "Item alpha-{id} costs {value}." near the sequence start, adds
    natural-text filler, then presents "Item alpha-{id} costs" at distance D.
    Measures prediction loss on the value tokens.

    The association is random and cannot be predicted from training. Lower
    loss = better recall. Memory mechanisms should show lower loss than
    local attention at distances beyond T/4 (the SSSL window boundary).
    """
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)

    if distances is None:
        distances = _default_distances(max_seq_len)

    window_boundary = max_seq_len // 4
    results_by_distance = {}

    for distance in distances:
        trial_losses = []

        for _ in range(num_trials):
            item_id = rng.randint(100, 999)
            value = rng.randint(1000, 9999)

            fact_tokens = tokenizer.encode(f" Item alpha-{item_id} costs {value}.")
            query_tokens = tokenizer.encode(f" Item alpha-{item_id} costs")
            value_tokens = tokenizer.encode(f" {value}")

            bos = tokenizer.get_bos_token_id()

            filler_len = distance - len(fact_tokens)
            if filler_len < 0:
                continue
            filler = _make_filler_tokens(tokenizer, filler_len, rng)

            prefix = [bos] + fact_tokens + filler + query_tokens
            value_start = len(prefix)
            core = prefix + value_tokens

            remaining = max_seq_len - len(core)
            if remaining < 0:
                continue
            tail = _make_filler_tokens(tokenizer, remaining, rng) if remaining > 0 else []
            sequence = core + tail

            positions = list(range(value_start, value_start + len(value_tokens)))
            losses = _compute_loss_at_positions(model, sequence, positions, device)
            if losses:
                trial_losses.append(float(np.mean(losses)))

        if trial_losses:
            results_by_distance[distance] = {
                "mean_loss_bits": float(np.mean(trial_losses)),
                "std_loss_bits": float(np.std(trial_losses)),
                "n_valid": len(trial_losses),
            }

    return {
        "task": "token_recall_at_distance",
        "max_seq_len": max_seq_len,
        "window_boundary": window_boundary,
        "distances": distances,
        "results_by_distance": {str(k): v for k, v in results_by_distance.items()},
    }


@torch.no_grad()
def evaluate_entity_tracking(
    model,
    tokenizer,
    _token_bytes: Optional[torch.Tensor] = None,
    max_seq_len: int = 2048,
    distances: Optional[list[int]] = None,
    num_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Probe 2: Entity-attribute tracking at varying distances.

    Plants "{Name} works at {Org}." near the sequence start, fills with
    natural text, then presents "{Name} works at" at distance D. Measures
    prediction loss on the organization name tokens.

    Uses fictional names and organizations to prevent memorization from
    training data. Lower loss = better entity tracking across distance.
    """
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)

    if distances is None:
        distances = _default_distances(max_seq_len)

    window_boundary = max_seq_len // 4
    results_by_distance = {}

    for distance in distances:
        trial_losses = []

        for _ in range(num_trials):
            name = rng.choice(_ENTITY_NAMES)
            org = rng.choice(_FICTIONAL_ORGS)

            fact_tokens = tokenizer.encode(f" {name} works at {org}.")
            query_tokens = tokenizer.encode(f" {name} works at")
            value_tokens = tokenizer.encode(f" {org}")

            bos = tokenizer.get_bos_token_id()

            filler_len = distance - len(fact_tokens)
            if filler_len < 0:
                continue
            filler = _make_filler_tokens(tokenizer, filler_len, rng)

            prefix = [bos] + fact_tokens + filler + query_tokens
            value_start = len(prefix)
            core = prefix + value_tokens

            remaining = max_seq_len - len(core)
            if remaining < 0:
                continue
            tail = _make_filler_tokens(tokenizer, remaining, rng) if remaining > 0 else []
            sequence = core + tail

            positions = list(range(value_start, value_start + len(value_tokens)))
            losses = _compute_loss_at_positions(model, sequence, positions, device)
            if losses:
                trial_losses.append(float(np.mean(losses)))

        if trial_losses:
            results_by_distance[distance] = {
                "mean_loss_bits": float(np.mean(trial_losses)),
                "std_loss_bits": float(np.std(trial_losses)),
                "n_valid": len(trial_losses),
            }

    return {
        "task": "entity_tracking",
        "max_seq_len": max_seq_len,
        "window_boundary": window_boundary,
        "distances": distances,
        "results_by_distance": {str(k): v for k, v in results_by_distance.items()},
    }


@torch.no_grad()
def evaluate_cross_boundary_ar(
    model,
    tokenizer,
    _token_bytes: Optional[torch.Tensor] = None,
    max_seq_len: int = 2048,
    num_pairs: int = 8,
    num_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Probe 3: Cross-boundary associative recall with multiple KV pairs.

    Plants N key-value associations in the first T/4 of the sequence, fills
    the middle with natural text, then queries each key in the second half.
    Measures prediction loss on value tokens.

    Unlike Probes 1-2 (single fact, sweep distance), this tests multi-item
    recall across the window boundary simultaneously — probing memory
    capacity, not just reach.

    Results are reported per pair (pair 0 planted first = farthest from
    its query, pair N-1 planted last = closest to its query).
    """
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)

    window_boundary = max_seq_len // 4

    # Accumulate losses per pair, keyed by actual distance
    pair_trial_losses = defaultdict(list)   # pair_idx -> [trial losses]
    pair_distances = defaultdict(list)      # pair_idx -> [distances]

    for _ in range(num_trials):
        pairs = []
        for _ in range(num_pairs):
            key_id = rng.randint(100, 999)
            value = rng.randint(1000, 9999)
            pairs.append((key_id, value))

        bos = tokenizer.get_bos_token_id()

        # Build KV section: placed sequentially from start
        kv_section = [bos]
        kv_end_positions = []  # end position of each KV pair's value
        for key_id, value in pairs:
            kv_text = f" Key-{key_id} maps to {value}."
            kv_tokens = tokenizer.encode(kv_text)
            kv_end_positions.append(len(kv_section) + len(kv_tokens))
            kv_section.extend(kv_tokens)

        # Filler: from end of KV section to ~T/2
        filler_target = max(0, max_seq_len // 2 - len(kv_section))
        filler = _make_filler_tokens(tokenizer, filler_target, rng)
        current = kv_section + filler

        # Build queries in order (pair 0 first)
        value_map = []  # (value_start, value_end, pair_idx, kv_end_pos)
        for idx, (key_id, value) in enumerate(pairs):
            q_tokens = tokenizer.encode(f" Key-{key_id} maps to")
            v_tokens = tokenizer.encode(f" {value}")

            value_start = len(current) + len(q_tokens)
            value_end = value_start + len(v_tokens)

            if value_end > max_seq_len:
                break
            current = current + q_tokens + v_tokens
            value_map.append((value_start, value_end, idx, kv_end_positions[idx]))

        # Pad remainder
        remaining = max_seq_len - len(current)
        if remaining > 0:
            current = current + _make_filler_tokens(tokenizer, remaining, rng)
        elif remaining < 0:
            current = current[:max_seq_len]

        # Compute loss at each query's value positions
        for value_start, value_end, pair_idx, kv_end in value_map:
            if value_end > len(current):
                continue
            positions = list(range(value_start, value_end))
            losses = _compute_loss_at_positions(model, current, positions, device)
            if losses:
                pair_trial_losses[pair_idx].append(float(np.mean(losses)))
                pair_distances[pair_idx].append(value_start - kv_end)

    # Summarize per pair
    per_pair = {}
    for idx in range(num_pairs):
        if idx in pair_trial_losses and pair_trial_losses[idx]:
            per_pair[idx] = {
                "mean_loss_bits": float(np.mean(pair_trial_losses[idx])),
                "std_loss_bits": float(np.std(pair_trial_losses[idx])),
                "mean_distance_tokens": float(np.mean(pair_distances[idx])),
                "n_valid": len(pair_trial_losses[idx]),
            }

    # Overall: mean across all pairs and trials
    all_losses = [l for losses in pair_trial_losses.values() for l in losses]
    overall = float(np.mean(all_losses)) if all_losses else float("nan")

    return {
        "task": "cross_boundary_ar",
        "max_seq_len": max_seq_len,
        "window_boundary": window_boundary,
        "num_pairs": num_pairs,
        "per_pair_results": {str(k): v for k, v in per_pair.items()},
        "overall_mean_loss_bits": overall,
    }
