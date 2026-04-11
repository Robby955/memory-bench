"""Training harness for memory-bench. Wraps nanochat with pluggable memory mechanisms."""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.nn.functional as F

# nanochat imports (available via sys.path from memory_bench.__init__)
import memory_bench  # sets up sys.path
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    autodetect_device_type, get_peak_flops, COMPUTE_DTYPE,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb

from memory_bench.models import build_gpt_config, count_parameters
from memory_bench.mechanisms import MECHANISMS
from memory_bench.eval.niah import evaluate_niah
from memory_bench.eval.synthetic import evaluate_bpb_by_position

print_banner()

# ---------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="memory-bench training")

# Model
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--window-pattern", type=str, default="SSSL")

# Memory mechanism
parser.add_argument("--mechanism", type=str, default="none", choices=list(MECHANISMS.keys()))
parser.add_argument("--num-memory-tokens", type=int, default=32, help="for persistent / rmt")
parser.add_argument("--segment-length", type=int, default=512, help="for rmt")
parser.add_argument("--bptt-depth", type=int, default=2, help="for rmt")
parser.add_argument("--ttt-layer", type=int, default=-1, help="layer to replace with TTT (-1=middle)")
parser.add_argument("--ttt-chunk-size", type=int, default=64, help="chunk size for TTT")
parser.add_argument("--deltanet-layers", type=str, default="", help="comma-separated layer indices for deltanet")

# Training
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=-1)
parser.add_argument("--target-param-data-ratio", type=float, default=12)
parser.add_argument("--weight-decay", type=float, default=0.28)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.008)
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
parser.add_argument("--num-iterations", type=int, default=-1)

# Evaluation
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=80 * 524288)
parser.add_argument("--niah-at-end", action="store_true", help="run NIAH eval at end of training")
parser.add_argument("--niah-trials", type=int, default=50)

# Logging
parser.add_argument("--exp-tag", type=str, default="dummy")
parser.add_argument("--wandb-project", type=str, default="memory-bench")
parser.add_argument("--no-compile", action="store_true", help="skip torch.compile")

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Compute init

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# Override nanochat's hardcoded seed 42 with our seed argument.
# compute_init() sets torch.manual_seed(42) -- we override it here so that
# different --seed values produce different model initializations.
import random
import numpy as np
torch.manual_seed(args.seed)
if device_type == "cuda":
    torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Seed-dependent data ordering: permute parquet file order so different seeds
# see data in different sequences. This captures data-order variance, not just
# model-init variance. Val file (last) stays fixed to prevent data leakage.
import nanochat.dataloader as _dl
_orig_list_parquet = _dl.list_parquet_files
def _seeded_list_parquet(**kwargs):
    files = _orig_list_parquet(**kwargs)
    if len(files) > 1:
        train_files = files[:-1]
        val_file = files[-1:]
        rng = random.Random(args.seed)
        rng.shuffle(train_files)
        return train_files + val_file
    return files
_dl.list_parquet_files = _seeded_list_parquet
print0(f"Data ordering seeded with seed={args.seed}")

if device_type == "cuda":
    gpu_peak_flops = get_peak_flops(torch.cuda.get_device_name(0))
else:
    gpu_peak_flops = float("inf")

# wandb
import wandb
use_dummy = args.exp_tag == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy else wandb.init(project=args.wandb_project, name=args.exp_tag, config=vars(args))

# ---------------------------------------------------------------------------
# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# ---------------------------------------------------------------------------
# Build mechanism
mechanism = None
if args.mechanism != "none":
    MechanismClass = MECHANISMS[args.mechanism]
    if args.mechanism == "persistent":
        mechanism = MechanismClass(num_tokens=args.num_memory_tokens)
    elif args.mechanism == "rmt":
        mechanism = MechanismClass(
            num_tokens=args.num_memory_tokens,
            seg_length=args.segment_length,
            bptt_depth=args.bptt_depth,
        )
    elif args.mechanism == "ttt":
        mechanism = MechanismClass(
            layer_idx=args.ttt_layer,
            chunk_size=args.ttt_chunk_size,
        )
    elif args.mechanism == "deltanet":
        layer_indices = [int(x) for x in args.deltanet_layers.split(",")] if args.deltanet_layers else None
        mechanism = MechanismClass(layer_indices=layer_indices)

    print0(f"Memory mechanism: {mechanism.name}")

# ---------------------------------------------------------------------------
# Build model
gpt_config = build_gpt_config(args, vocab_size)

with torch.device("meta"):
    model = GPT(gpt_config)
model.to_empty(device=device)
model.init_weights()

model_config_kwargs = asdict(gpt_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")

# ---------------------------------------------------------------------------
# Setup optimizer BEFORE mechanism wrapping. DistMuonAdamW's Muon groups
# interact poorly with TTT/DeltaNet replacement params (CUDA illegal memory
# access in Newton-Schulz iteration). Building the optimizer first, then
# adding replacement params as AdamW avoids this. The tradeoff: TTT/DeltaNet's
# c_q/c_k/c_v/c_proj use AdamW instead of Muon, which if anything
# *disadvantages* them (conservative for a benchmark comparison).
orig_model = model

# Scaling laws (simplified from nanochat's base_train.py)
num_scaling_params = sum(p.numel() for name, p in orig_model.named_parameters()
                        if "transformer.h" in name or "lm_head" in name)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

B_REF = 2**19
D_REF = args.target_param_data_ratio * num_scaling_params  # approximate

total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / max(D_REF, 1)
    predicted = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(max(predicted, 1)))
    print0(f"Auto batch size: {total_batch_size:,}")

batch_lr_scale = (total_batch_size / B_REF) ** 0.5
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / max(target_tokens, 1))

optimizer = orig_model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

# Now apply memory mechanism (adds/replaces params in model).
# After wrapping, we need to fix the optimizer:
# 1. Remove stale params (from replaced layers, no longer in model)
# 2. Add new params (from mechanism's replacement layers) as AdamW
# 3. Add mechanism-specific params to a separate AdamW optimizer
mech_optimizer = None
if mechanism is not None:
    model = mechanism.wrap_model(model, gpt_config)

    # Identify mechanism-specific params (go to separate optimizer)
    mech_extra_ids = set()
    for g in mechanism.extra_param_groups():
        for p in g["params"]:
            mech_extra_ids.add(id(p))

    # Step 1: Remove stale params from optimizer groups
    current_param_ids = {id(p) for p in model.parameters()}
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) in current_param_ids]

    # Step 2: Find new params that need the main optimizer
    # (e.g. TTT/DeltaNet create new c_q/c_k/c_v/c_proj with same shapes)
    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer_param_ids.add(id(p))

    orphan_params = [p for p in model.parameters()
                     if id(p) not in optimizer_param_ids and id(p) not in mech_extra_ids]

    # Add orphan params as AdamW. Adding to Muon groups causes CUDA illegal
    # memory access in DistMuonAdamW (Newton-Schulz + TTT gradient interaction).
    # Using AdamW is conservative: it disadvantages TTT/DeltaNet if anything.
    if orphan_params:
        print0(f"Mechanism orphan params → AdamW: {len(orphan_params)} params, "
               f"{sum(p.numel() for p in orphan_params):,} elements")
        optimizer.add_param_group({
            "kind": "adamw", "params": orphan_params, "lr": 0.003 * batch_lr_scale,
            "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01,
            "initial_lr": 0.003 * batch_lr_scale,
        })

    # Step 3: Separate optimizer for mechanism-specific params
    mech_param_groups = []
    for group in mechanism.extra_param_groups():
        group.pop("kind", None)
        group["initial_lr"] = group["lr"]
        mech_param_groups.append(group)
    if mech_param_groups:
        mech_optimizer = torch.optim.AdamW(
            mech_param_groups,
            lr=mech_param_groups[0]["lr"],
            foreach=False,
        )
        for pg in mech_optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

param_counts = count_parameters(model)
print0(f"Total parameters: {param_counts['total']:,}")
if mechanism:
    print0(f"Memory parameters: {mechanism.num_memory_params:,}")

# Compile after wrapping (skip for mechanisms that use Triton kernels, e.g. DeltaNet)
if not args.no_compile:
    model = torch.compile(model, dynamic=False)

# ---------------------------------------------------------------------------
# Dataloaders
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
)
x, y, dataloader_state_dict = next(train_loader)

# ---------------------------------------------------------------------------
# Training iterations
if args.num_iterations > 0:
    num_iterations = args.num_iterations
else:
    num_iterations = target_tokens // total_batch_size

total_tokens_planned = total_batch_size * num_iterations
print0(f"Training for {num_iterations:,} iterations ({total_tokens_planned:,} tokens)")

# LR schedule
def get_lr_multiplier(it):
    warmup = args.warmup_steps
    warmdown = round(args.warmdown_ratio * num_iterations)
    if it < warmup:
        return (it + 1) / warmup
    elif it <= num_iterations - warmdown:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown
        return progress + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    warmdown = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown
    if it < 400:
        return 0.85 + (it / 400) * 0.12
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown
        return 0.97 * (1 - progress) + 0.90 * progress
    return 0.97

def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# Gradient accumulation
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens == 0
grad_accum_steps = total_batch_size // world_tokens
print0(f"Grad accum steps: {grad_accum_steps}")

# Estimate FLOPs per token
num_flops_per_token = orig_model.estimate_flops()

# ---------------------------------------------------------------------------
# Segment-aware evaluation for RMT

@torch.no_grad()
def evaluate_bpb_segments(model, mechanism, batches, steps, token_bytes, seq_len):
    """Evaluate BPB for segment-based mechanisms (RMT).

    Processes validation data in segments with memory, matching the training
    forward path. Returns BPB comparable to evaluate_bpb.
    """
    dev = model.get_device()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=dev)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=dev)
    seg_len = mechanism.segment_length
    assert seq_len % seg_len == 0, \
        f"seq_len ({seq_len}) must be divisible by segment_length ({seg_len}) for segment eval"
    n_segments = seq_len // seg_len
    batch_iter = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        memory_state = None
        for seg_idx in range(n_segments):
            seg_start = seg_idx * seg_len
            seg_end = seg_start + seg_len
            seg_x = x[:, seg_start:seg_end]
            seg_y = y[:, seg_start:seg_end]

            logits, memory_state = mechanism.forward_segment_logits(
                model, seg_x, memory_state
            )
            memory_state = memory_state.detach()
            memory_state = mechanism.on_segment_boundary(memory_state)

            # Per-token cross-entropy (matching evaluate_bpb's accounting)
            loss2d = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                seg_y.reshape(-1),
                ignore_index=-1,
                reduction="none",
            ).view(seg_y.shape)

            # BPB accounting: weight by token byte lengths
            y_flat = seg_y.reshape(-1)
            valid = y_flat >= 0
            y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
            num_bytes_flat = torch.where(
                valid, token_bytes[y_safe],
                torch.zeros_like(y_flat, dtype=token_bytes.dtype),
            )
            total_nats += (loss2d.view(-1) * (num_bytes_flat > 0)).sum()
            total_bytes += num_bytes_flat.sum()

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float("inf")
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Training loop
step = 0
val_bpb = None
min_val_bpb = float("inf")
smooth_loss = 0
total_time = 0

while True:
    last_step = step == num_iterations

    # Evaluate val BPB
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        if mechanism is not None and mechanism.requires_segments:
            # Segment-aware eval for RMT: process val data in segments with
            # memory, matching the training forward path exactly.
            val_bpb = evaluate_bpb_segments(
                orig_model, mechanism, val_loader, eval_steps,
                token_bytes, args.max_seq_len,
            )
        else:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        min_val_bpb = min(min_val_bpb, val_bpb)
        print0(f"Step {step:05d} | val_bpb: {val_bpb:.6f}")
        wandb_run.log({"step": step, "val/bpb": val_bpb, "total_time": total_time})
        model.train()

    if last_step:
        break

    # Training step
    synchronize()
    t0 = time.time()

    if mechanism is not None and mechanism.requires_segments:
        # RMT: segment-level training
        total_loss = 0
        memory_state = None
        assert args.max_seq_len % mechanism.segment_length == 0, \
            f"max_seq_len ({args.max_seq_len}) must be divisible by segment_length ({mechanism.segment_length})"
        n_segments = args.max_seq_len // mechanism.segment_length

        for micro_step in range(grad_accum_steps):
            for seg_idx in range(n_segments):
                seg_start = seg_idx * mechanism.segment_length
                seg_end = seg_start + mechanism.segment_length
                seg_x = x[:, seg_start:seg_end]
                seg_y = y[:, seg_start:seg_end]

                loss, memory_state = mechanism.forward_segment(
                    orig_model, seg_x, seg_y, memory_state
                )
                loss = loss / (grad_accum_steps * n_segments)
                total_loss += loss.detach()

                # Truncated BPTT: segments within the BPTT window share a
                # computation graph through memory_state. We must use
                # retain_graph=True for all but the last segment in the window
                # so that earlier segments' graphs remain available.
                is_last_segment = (seg_idx == n_segments - 1)
                in_bptt_window = (seg_idx >= n_segments - mechanism.bptt_depth)
                retain = in_bptt_window and not is_last_segment
                loss.backward(retain_graph=retain)

                if seg_idx < n_segments - 1:
                    if seg_idx < n_segments - mechanism.bptt_depth:
                        memory_state = memory_state.detach()
                    memory_state = mechanism.on_segment_boundary(memory_state)

            mechanism.reset()
            memory_state = None
            x, y, dataloader_state_dict = next(train_loader)

        train_loss_f = total_loss.item()
    else:
        # Standard training (baseline, persistent, ttt, deltanet)
        total_loss_acc = 0.0
        for micro_step in range(grad_accum_steps):
            loss = model(x, y)
            total_loss_acc += loss.detach().item()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, dataloader_state_dict = next(train_loader)
        train_loss_f = total_loss_acc / grad_accum_steps

    # Optimizer step
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group.get("kind") == "muon":
            group["momentum"] = get_muon_momentum(step)
            group["weight_decay"] = get_weight_decay(step)
    # Gradient clipping: Muon has implicit normalization (Newton-Schulz), but
    # mechanism orphan params (AdamW) and mech_optimizer params have none.
    # Without clipping, DeltaNet gate params can diverge by step ~50.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Guard: zero out None grads before optimizer step.
    # Mechanism wrapping can leave params in the main optimizer that don't
    # participate in the (now-wrapped) forward, causing _reduce_adamw to crash.
    none_grad_count = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
                none_grad_count += 1
    if none_grad_count > 0 and step == 0:
        print0(f"WARNING: {none_grad_count} params had None grads (zeroed). Check mechanism wrapping.")
    optimizer.step()
    if mech_optimizer is not None:
        # Sync mechanism param gradients across ranks (not in DistMuonAdamW)
        if ddp_world_size > 1:
            for group in mech_optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    if not p.grad.is_contiguous():
                        p.grad = p.grad.contiguous()
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        for group in mech_optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        mech_optimizer.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    dt = time.time() - t0

    # Logging
    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * train_loss_f
    debiased = smooth_loss / (1 - ema ** (step + 1))
    if step > 10:
        total_time += dt
    tok_sec = int(total_batch_size / dt)
    mfu = 100 * num_flops_per_token * total_batch_size / dt / (gpu_peak_flops * ddp_world_size)

    if step % 50 == 0:
        mech_name = mechanism.name if mechanism else "baseline"
        print0(f"[{mech_name}] step {step:05d}/{num_iterations} | loss: {debiased:.4f} | dt: {dt*1000:.0f}ms | tok/s: {tok_sec:,} | mfu: {mfu:.1f}%")
        wandb_run.log({"step": step, "train/loss": debiased, "train/tok_sec": tok_sec, "train/mfu": mfu})

    # GC management (from nanochat)
    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()
    elif step % 5000 == 0:
        gc.collect()

    step += 1

# ---------------------------------------------------------------------------
# Final stats
print0(f"Training complete. Total time: {total_time/60:.1f}m")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Peak VRAM: {get_max_memory() / 1024**2:.0f} MiB")

# ---------------------------------------------------------------------------
# BPB-by-position analysis (always run — cheap and highly informative)
bpb_by_pos = None
if master_process:
    gc.enable()
    gc.collect()
    print0("Running BPB-by-position analysis...")
    model.eval()
    bpb_val_loader = build_val_loader()
    bpb_by_pos = evaluate_bpb_by_position(
        orig_model if mechanism and mechanism.requires_segments else model,
        bpb_val_loader, token_bytes, num_steps=50, num_buckets=32,
    )
    bucket_bpbs = [v["bpb"] for v in bpb_by_pos["buckets"].values()]
    print0(f"BPB by position (first/mid/last bucket): "
           f"{bucket_bpbs[0]:.4f} / "
           f"{bucket_bpbs[len(bucket_bpbs)//2]:.4f} / "
           f"{bucket_bpbs[-1]:.4f}")

# Save checkpoint (enables post-hoc NIAH, analysis, reproducibility)
if master_process:
    os.makedirs("results/checkpoints", exist_ok=True)
    mech_label = args.mechanism if args.mechanism != "none" else "baseline"
    ckpt_file = f"results/checkpoints/{mech_label}_d{args.depth}_s{args.seed}.pt"
    ckpt = {"model_state_dict": orig_model.state_dict(), "step": step, "val_bpb": val_bpb}
    if mechanism:
        ckpt["mechanism_params"] = {k: v for k, v in zip(
            [f"param_{i}" for i in range(len(mechanism.extra_param_groups()))],
            [{n: p.data for n, p in enumerate(g["params"])} for g in mechanism.extra_param_groups()],
        )} if mechanism.extra_param_groups() else {}
    torch.save(ckpt, ckpt_file)
    print0(f"Checkpoint saved to {ckpt_file}")

# NIAH evaluation
niah_results = None
if args.niah_at_end and master_process:
    print0("Running NIAH evaluation...")
    model.eval()
    if mechanism is None:
        # Baseline: use Engine (fast KVCache generation)
        from nanochat.engine import Engine
        engine = Engine(orig_model, tokenizer)
        niah_results = evaluate_niah(engine, tokenizer, num_trials=args.niah_trials,
                                     device=device, seed=args.seed)
    else:
        # Mechanisms: use naive or RMT-aware generation (O(T²) but correct)
        print0(f"Using {'segment-aware' if mechanism.requires_segments else 'naive'} generation for {mechanism.name}")
        niah_results = evaluate_niah(orig_model, tokenizer, num_trials=args.niah_trials,
                                     device=device, seed=args.seed, mechanism=mechanism)
    print0(f"NIAH results: {json.dumps(niah_results, indent=2)}")
    wandb_run.log({"niah": niah_results})

# Save results
if master_process:
    mem_params = mechanism.num_memory_params if mechanism else 0
    param_overhead_pct = 100 * mem_params / max(param_counts["total"] - mem_params, 1) if mem_params > 0 else 0.0
    results = {
        "mechanism": mechanism.name if mechanism else "baseline",
        "depth": args.depth,
        "seed": args.seed,
        "val_bpb": val_bpb,
        "min_val_bpb": min_val_bpb,
        "total_params": param_counts["total"],
        "base_params": param_counts["total"] - mem_params,
        "memory_params": mem_params,
        "param_overhead_pct": round(param_overhead_pct, 3),
        "total_time_min": total_time / 60,
        "peak_vram_mib": get_max_memory() / 1024**2,
        "flops_per_step": num_flops_per_token * total_batch_size,
        "total_flops": num_flops_per_token * total_batch_size * num_iterations,
        "bpb_by_position": bpb_by_pos,
    }
    if niah_results:
        results["niah"] = niah_results
    os.makedirs("results", exist_ok=True)
    mech_label = args.mechanism if args.mechanism != "none" else "baseline"
    result_file = f"results/{mech_label}_d{args.depth}_s{args.seed}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print0(f"Results saved to {result_file}")

wandb_run.finish()
compute_cleanup()
