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
