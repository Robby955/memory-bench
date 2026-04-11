"""
Needle-in-a-Haystack (NIAH) evaluation.

Tests whether the model can retrieve a specific piece of information
(a "passkey") embedded at various positions within a long context of
filler text. This is the standard test for memory capability in LLMs.

Protocol:
    1. Generate filler text (repeated simple sentences)
    2. Insert a random 6-digit passkey at a specified position
    3. Add a retrieval prompt at the end
    4. Generate tokens and check if the passkey appears in output
    5. Repeat across multiple context lengths and positions

The key insight: models with effective memory mechanisms should maintain
high retrieval accuracy even when the passkey is far from the end of
the context, while standard transformers degrade with distance.
"""

import random
import torch


# Filler sentences used to pad the context
FILLER_SENTENCES = [
    "The weather today is quite pleasant with clear skies and mild temperatures.",
    "Many people enjoy reading books in their spare time as a relaxing activity.",
    "The quick brown fox jumps over the lazy dog near the old stone wall.",
    "Scientists continue to make important discoveries about the natural world.",
    "Music has been an important part of human culture for thousands of years.",
    "The ocean covers more than seventy percent of the surface of the earth.",
    "Education is widely considered to be one of the most important investments.",
    "Technology continues to advance at a remarkable pace in modern society.",
    "Fresh fruits and vegetables are an essential part of a healthy diet.",
    "The history of civilization spans many thousands of years across the globe.",
]


def generate_niah_prompt(
    tokenizer,
    context_length: int,
    passkey_position: float,
    passkey: str,
) -> tuple[list[int], str]:
    """Generate a NIAH test prompt.

    Args:
        tokenizer: nanochat tokenizer
        context_length: target context length in tokens
        passkey_position: where to insert passkey (0.0 = start, 1.0 = end)
        passkey: the passkey string to insert

    Returns:
        (token_ids, passkey) tuple
    """
    # Build the passkey insertion text
    passkey_text = f" The secret passkey is {passkey}. Remember this number. "
    retrieval_prompt = " What is the secret passkey mentioned above? The secret passkey is "

    # Tokenize the special parts to know their length
    passkey_tokens = tokenizer.encode(passkey_text)
    retrieval_tokens = tokenizer.encode(retrieval_prompt)
    bos_id = tokenizer.get_bos_token_id()

    # Calculate how many filler tokens we need
    reserved = 1 + len(passkey_tokens) + len(retrieval_tokens)  # BOS + passkey + retrieval
    filler_needed = context_length - reserved

    if filler_needed <= 0:
        # Context too short for this test
        return None, passkey

    # Generate filler tokens
    filler_tokens = []
    rng = random.Random(0)  # fixed seed for reproducible filler text
    while len(filler_tokens) < filler_needed:
        sentence = rng.choice(FILLER_SENTENCES)
        tokens = tokenizer.encode(" " + sentence)
        filler_tokens.extend(tokens)
    filler_tokens = filler_tokens[:filler_needed]

    # Insert passkey at the specified position
    insert_pos = int(passkey_position * len(filler_tokens))
    insert_pos = max(0, min(insert_pos, len(filler_tokens)))

    full_tokens = (
        [bos_id]
        + filler_tokens[:insert_pos]
        + passkey_tokens
        + filler_tokens[insert_pos:]
        + retrieval_tokens
    )

    # Truncate to exact context length (in case of off-by-one from tokenization)
    full_tokens = full_tokens[:context_length]

    return full_tokens, passkey


def check_passkey_in_output(tokenizer, output_tokens: list[int], passkey: str) -> bool:
    """Check if the passkey appears in the generated output."""
    output_text = tokenizer.decode(output_tokens)
    return passkey in output_text


@torch.no_grad()
def generate_naive(model, prompt_tokens: list[int], max_tokens: int = 20,
                   device: torch.device = None) -> list[int]:
    """O(T²) autoregressive generation without KVCache.

    Works with any model (baseline or memory-augmented) because it runs the
    full model forward on the entire sequence each step. Slower than
    KVCache-based generation but correct for all mechanisms.
    """
    T = len(prompt_tokens)
    total_len = T + max_tokens

    # Pre-allocate buffer to avoid creating new tensors each step
    buf = torch.zeros(1, total_len, dtype=torch.long, device=device)
    buf[0, :T] = torch.tensor(prompt_tokens, dtype=torch.long)
    cur_len = T

    for _ in range(max_tokens):
        logits = model(buf[:, :cur_len])  # (1, cur_len, vocab)
        next_token = logits[0, -1].argmax().item()
        buf[0, cur_len] = next_token
        cur_len += 1

    return buf[0, :cur_len].tolist()


@torch.no_grad()
def generate_rmt(model, mechanism, prompt_tokens: list[int], max_tokens: int = 20,
                 device: torch.device = None) -> list[int]:
    """Segment-aware generation for RMT.

    Processes the prompt in segments with memory carried forward, then
    generates tokens autoregressively using the carried memory.

    Key invariants:
    - segment_memory: the memory INPUT for the current segment. Fixed within
      a segment, updated only when crossing a segment boundary.
    - During generation, each step re-processes the growing current_seg with
      segment_memory. The forward output memory is DISCARDED (not reused as
      input) until the segment fills up and a boundary is crossed.
    - last_logits: cached logits from the most recent forward pass, used when
      current_seg is empty (right after a segment boundary).
    - on_segment_boundary matches training convention: called between segments.
    """
    seg_len = mechanism.segment_length
    n_prompt = len(prompt_tokens)
    n_full = n_prompt // seg_len

    prompt_t = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    segment_memory = None  # Memory INPUT for current segment
    last_logits = None     # Cached logits from last forward pass

    # Phase 1: Process full prompt segments
    for i in range(n_full):
        seg = prompt_t[:, i * seg_len:(i + 1) * seg_len]
        last_logits, new_mem = mechanism.forward_segment_logits(model, seg, segment_memory)
        new_mem = new_mem.detach()
        if i < n_full - 1:
            # Between segments: apply boundary to prepare for next segment
            segment_memory = mechanism.on_segment_boundary(new_mem)
        else:
            # After last full segment: save output for phase 2/3
            segment_memory = new_mem

    # Phase 2: Process remainder of prompt
    remainder_start = n_full * seg_len
    remainder = prompt_tokens[remainder_start:]

    if remainder:
        # Remainder starts a new segment → apply boundary first
        if n_full > 0:
            segment_memory = mechanism.on_segment_boundary(segment_memory)
        seg_x = torch.tensor([remainder], dtype=torch.long, device=device)
        last_logits, _ = mechanism.forward_segment_logits(model, seg_x, segment_memory)
        # segment_memory stays as the INPUT for this segment — we'll re-process
        # remainder + generated tokens with this same segment_memory each step
    elif n_full > 0:
        # Exact division: generation starts a fresh segment
        segment_memory = mechanism.on_segment_boundary(segment_memory)

    if last_logits is None:
        return prompt_tokens  # Empty or degenerate prompt

    # Phase 3: Generate tokens
    current_seg = list(remainder)  # Tokens in current segment (re-processed each step)
    all_generated = []

    for _ in range(max_tokens):
        # Get next-token logits
        if current_seg:
            seg_x = torch.tensor([current_seg], dtype=torch.long, device=device)
            logits, _ = mechanism.forward_segment_logits(model, seg_x, segment_memory)
        else:
            # Empty current_seg (just crossed a boundary or exact-division start):
            # use cached logits from the last forward pass
            logits = last_logits

        next_token = logits[0, -1].argmax().item()
        all_generated.append(next_token)
        current_seg.append(next_token)

        # If segment is now full, advance memory
        if len(current_seg) >= seg_len:
            seg_x = torch.tensor([current_seg[:seg_len]], dtype=torch.long, device=device)
            last_logits, new_mem = mechanism.forward_segment_logits(
                model, seg_x, segment_memory
            )
            segment_memory = mechanism.on_segment_boundary(new_mem.detach())
            current_seg = current_seg[seg_len:]

    return prompt_tokens + all_generated


def evaluate_niah(
    model_or_engine,
    tokenizer,
    context_lengths: list[int] = None,
    passkey_positions: list[float] = None,
    num_trials: int = 50,
    max_gen_tokens: int = 20,
    device: torch.device = None,
    seed: int = 42,
    mechanism=None,
) -> dict:
    """Run the full NIAH evaluation.

    Supports two modes:
    - Engine-based (baseline): fast KVCache generation via nanochat Engine
    - Naive (mechanisms): O(T²) full-forward generation, works with any mechanism
    - RMT: segment-aware generation with memory carry

    Args:
        model_or_engine: nanochat Engine (baseline) or GPT model (mechanisms)
        tokenizer: nanochat tokenizer
        context_lengths: list of context lengths to test
        passkey_positions: list of relative positions (0.0-1.0) to insert passkey
        num_trials: number of trials per (length, position) pair
        max_gen_tokens: max tokens to generate for retrieval
        device: target device
        seed: random seed for reproducibility
        mechanism: MemoryModule instance (None for baseline/Engine mode)

    Returns:
        Dict with accuracy per (length, position) pair and overall accuracy.
    """
    if context_lengths is None:
        context_lengths = [256, 512, 1024, 2048]
    if passkey_positions is None:
        passkey_positions = [0.1, 0.25, 0.5, 0.75, 0.9]

    use_engine = mechanism is None and hasattr(model_or_engine, 'generate_batch')
    use_rmt = mechanism is not None and getattr(mechanism, 'requires_segments', False)

    rng = random.Random(seed)
    results = {}
    total_correct = 0
    total_trials = 0

    for ctx_len in context_lengths:
        results[ctx_len] = {}
        for pos in passkey_positions:
            correct = 0

            for trial in range(num_trials):
                passkey = str(rng.randint(100000, 999999))
                prompt_tokens, passkey = generate_niah_prompt(
                    tokenizer, ctx_len, pos, passkey
                )
                if prompt_tokens is None:
                    continue

                try:
                    if use_engine:
                        output_tokens, _ = model_or_engine.generate_batch(
                            prompt_tokens,
                            num_samples=1,
                            max_tokens=max_gen_tokens,
                            temperature=0,
                            seed=seed + trial,
                        )
                        all_tokens = output_tokens[0]
                    elif use_rmt:
                        all_tokens = generate_rmt(
                            model_or_engine, mechanism, prompt_tokens,
                            max_tokens=max_gen_tokens, device=device,
                        )
                    else:
                        all_tokens = generate_naive(
                            model_or_engine, prompt_tokens,
                            max_tokens=max_gen_tokens, device=device,
                        )

                    generated = all_tokens[len(prompt_tokens):]
                    if check_passkey_in_output(tokenizer, generated, passkey):
                        correct += 1
                except Exception as e:
                    import warnings
                    warnings.warn(f"NIAH generation failed (ctx={ctx_len}, pos={pos}, trial={trial}): {e}")
                    # count as failure but don't silently swallow

            accuracy = correct / max(num_trials, 1)
            results[ctx_len][pos] = accuracy
            total_correct += correct
            total_trials += num_trials

    overall = total_correct / max(total_trials, 1)

    return {
        "context_lengths": context_lengths,
        "positions": passkey_positions,
        "accuracy": {str(k): {str(p): v for p, v in inner.items()} for k, inner in results.items()},
        "overall_accuracy": overall,
    }
