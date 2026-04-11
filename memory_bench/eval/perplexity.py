"""
Perplexity evaluation wrapper.

Wraps nanochat's evaluate_bpb to report bits-per-byte on the validation set.
BPB is vocab-size-invariant, making it the correct metric for comparing
models with potentially different tokenizers.
"""

import torch
from nanochat.loss_eval import evaluate_bpb
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer, get_token_bytes


def evaluate_perplexity(
    model,
    device_batch_size: int = 32,
    max_seq_len: int = 2048,
    eval_tokens: int = 80 * 524288,
    device: torch.device = None,
) -> float:
    """Evaluate model perplexity as bits-per-byte.

    Args:
        model: GPT model (possibly with memory augmentation)
        device_batch_size: batch size per device
        max_seq_len: context length
        eval_tokens: total tokens to evaluate on
        device: target device

    Returns:
        Bits per byte (BPB) on validation set
    """
    if device is None:
        device = next(model.parameters()).device

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, device_batch_size, max_seq_len, split="val", device=device,
    )

    ddp_world_size = 1
    if torch.distributed.is_initialized():
        ddp_world_size = torch.distributed.get_world_size()

    eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)

    model.eval()
    with torch.no_grad():
        bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    model.train()

    return bpb
