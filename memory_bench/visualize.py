"""
Architecture visualization for memory-bench.

Generates a multi-page PDF showing:
1. Overall architecture: how memory mechanisms plug into the transformer
2. Per-mechanism detailed diagrams with data flow
3. Mathematical formulations
4. Comparison table

Usage:
    python -m memory_bench.visualize [--output architecture.pdf]
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import argparse
import os


# ─── Colors ──────────────────────────────────────────────────────────────────

COLORS = {
    "baseline": "#4A90D9",
    "persistent": "#E8A838",
    "rmt": "#50C878",
    "ttt": "#E85050",
    "deltanet": "#9B59B6",
    "bg": "#F8F9FA",
    "border": "#2C3E50",
    "text": "#1A1A1A",
    "light": "#ECF0F1",
    "arrow": "#34495E",
    "math_bg": "#FFF8E1",
    "code_bg": "#F5F5F5",
}


def _draw_box(ax, x, y, w, h, text, color="#4A90D9", text_color="white", fontsize=9, alpha=0.9):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=COLORS["border"],
        linewidth=1.2, alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color=text_color, fontweight="bold",
        wrap=True,
    )
    return box


def _draw_arrow(ax, x1, y1, x2, y2, color=None, style="-|>", lw=1.5):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS["arrow"]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
    )


# ─── Page 1: Title + Overview ────────────────────────────────────────────────

def _draw_title_page(fig):
    """Title page with project overview."""
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 8.5, "memory-bench", fontsize=36, ha="center", fontweight="bold",
            color=COLORS["border"], fontfamily="monospace")
    ax.text(5, 7.8, "Controlled Comparison of Memory Mechanisms\nfor Small-Scale LLMs",
            fontsize=16, ha="center", color=COLORS["text"])

    # Mechanisms overview
    mechanisms = [
        ("Persistent Memory", COLORS["persistent"],
         "K learned KV vectors per layer\nPosition-agnostic background knowledge"),
        ("RMT", COLORS["rmt"],
         "Memory tokens carried between segments\nRecurrent information bottleneck"),
        ("TTT-Linear", COLORS["ttt"],
         "Inner linear model W updated per-chunk\nMini-batch dual form (LaCT)"),
        ("Gated DeltaNet", COLORS["deltanet"],
         "Linear attention with delta rule\nD×D state matrix, O(1) per layer"),
    ]

    y_start = 6.0
    for i, (name, color, desc) in enumerate(mechanisms):
        y = y_start - i * 1.3
        _draw_box(ax, 1, y, 2.5, 0.9, name, color=color, fontsize=11)
        ax.text(4, y + 0.45, desc, fontsize=9, va="center", color=COLORS["text"])

    # Built on nanochat
    ax.text(5, 0.8, "Built on nanochat (Karpathy 2025) · GPT-2 scale · 8×H100",
            fontsize=10, ha="center", color="#7F8C8D", style="italic")


# ─── Page 2: Transformer Architecture with Injection Points ──────────────────

def _draw_transformer_architecture(fig):
    """Show the standard transformer block with memory mechanism injection points."""
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Transformer Block Architecture - Memory Injection Points",
                 fontsize=14, fontweight="bold", pad=20)

    # Standard transformer block (center)
    cx = 4.5
    block_w = 3.0

    # Input
    _draw_box(ax, cx, 0.2, block_w, 0.5, "Input Embeddings + RoPE", color="#7FB3D8", fontsize=8)

    # RMSNorm
    _draw_box(ax, cx, 1.0, block_w, 0.4, "RMSNorm", color="#BDC3C7", text_color="#2C3E50", fontsize=8)
    _draw_arrow(ax, cx + block_w / 2, 0.7, cx + block_w / 2, 1.0)

    # Attention
    _draw_box(ax, cx, 1.6, block_w, 0.8, "CausalSelfAttention\n(FA3 / SDPA)", color=COLORS["baseline"], fontsize=9)
    _draw_arrow(ax, cx + block_w / 2, 1.4, cx + block_w / 2, 1.6)

    # Residual add
    ax.text(cx + block_w + 0.3, 2.0, "+", fontsize=16, fontweight="bold", color=COLORS["border"])
    _draw_arrow(ax, cx + block_w / 2, 2.4, cx + block_w / 2, 2.6)

    # RMSNorm 2
    _draw_box(ax, cx, 2.6, block_w, 0.4, "RMSNorm", color="#BDC3C7", text_color="#2C3E50", fontsize=8)

    # MLP
    _draw_box(ax, cx, 3.2, block_w, 0.7, "MLP (ReLU²)\n4× expansion", color="#85929E", fontsize=9)
    _draw_arrow(ax, cx + block_w / 2, 3.0, cx + block_w / 2, 3.2)

    # Output
    ax.text(cx + block_w + 0.3, 3.55, "+", fontsize=16, fontweight="bold", color=COLORS["border"])
    _draw_box(ax, cx, 4.1, block_w, 0.5, "× n_layer", color="#D5DBDB", text_color="#2C3E50", fontsize=9)
    _draw_arrow(ax, cx + block_w / 2, 3.9, cx + block_w / 2, 4.1)

    # Injection points (right side)
    rx = 8.5
    annotations = [
        (2.0, COLORS["persistent"], "Persistent Memory\nPrepend K learned\nKV pairs to K,V"),
        (2.0, COLORS["rmt"], "RMT\nPrepend M memory\ntokens at embedding"),
        (2.0, COLORS["ttt"], "TTT-Linear\nReplace attention with\ndual form W update"),
        (2.0, COLORS["deltanet"], "Gated DeltaNet\nReplace attention with\nδ-rule recurrence"),
    ]

    for i, (y_ref, color, text) in enumerate(annotations):
        y = 5.5 + i * 1.1
        _draw_box(ax, rx, y, 3.0, 0.9, text, color=color, fontsize=7)
        # Arrow from injection point to the attention block
        _draw_arrow(ax, rx, y + 0.45, cx + block_w + 0.1, y_ref,
                    color=color, style="-|>", lw=1.0)

    # Legend
    ax.text(0.5, 9.5, "Each mechanism modifies the attention layer differently:",
            fontsize=10, color=COLORS["text"])
    legend_items = [
        (COLORS["persistent"], "Persistent: adds KV vectors (non-destructive)"),
        (COLORS["rmt"], "RMT: prepends memory tokens (recurrent between segments)"),
        (COLORS["ttt"], "TTT: replaces attention with inner model + gradient descent"),
        (COLORS["deltanet"], "DeltaNet: replaces attention with linear recurrence"),
    ]
    for i, (color, text) in enumerate(legend_items):
        y = 9.0 - i * 0.35
        ax.plot([0.5, 0.9], [y, y], color=color, linewidth=3)
        ax.text(1.1, y, text, fontsize=8, va="center", color=COLORS["text"])


# ─── Page 3: TTT-Linear Dual Form ────────────────────────────────────────────

def _draw_ttt_diagram(fig):
    """Detailed TTT-Linear data flow and dual form computation."""
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("TTT-Linear: Mini-Batch Dual Form (LaCT)", fontsize=14, fontweight="bold", pad=20)

    # Input
    _draw_box(ax, 0.5, 8.5, 2, 0.6, "Input x", color="#7FB3D8", fontsize=10)

    # Projections (4 parallel)
    projs = [
        (0.2, "Q = W_q·x", "#E74C3C"),
        (2.7, "K = W_k·x", "#3498DB"),
        (5.2, "V = W_v·x", "#2ECC71"),
        (7.7, "η = softplus(W_η·x)", "#F39C12"),
    ]
    for x, label, color in projs:
        _draw_box(ax, x, 7.2, 2.2, 0.6, label, color=color, fontsize=8)
        _draw_arrow(ax, 1.5, 8.5, x + 1.1, 7.8)

    # RoPE + Norm
    ax.text(1.5, 6.7, "RoPE + L2 Norm →", fontsize=8, color="#7F8C8D")

    # Chunk processing
    _draw_box(ax, 0.5, 5.5, 9, 0.7, "Process chunks of C tokens (chunk_size = 64)", color="#D5DBDB", text_color="#2C3E50", fontsize=10)

    # Dual form steps
    steps = [
        ("Step 1: E = K @ W^T - V", "Reconstruction error at current W"),
        ("Step 2: A = tril(Q @ K^T)", "Causal attention matrix (like linear attention)"),
        ("Step 3: Z = Q @ W^T - A @ (η·E)", "Output with virtual SGD updates (DUAL FORM)"),
        ("Step 4: W ← W - (1/C)·K^T @ (η·E)", "Update inner model for next chunk"),
        ("Step 5: W ← normalize_cols(W)", "L2 weight normalization (LaCT)"),
    ]
    for i, (eq, desc) in enumerate(steps):
        y = 4.5 - i * 0.85
        # Math box
        box = FancyBboxPatch(
            (0.5, y), 5.5, 0.65,
            boxstyle="round,pad=0.05",
            facecolor=COLORS["math_bg"], edgecolor="#F39C12",
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(0.8, y + 0.4, eq, fontsize=9, fontfamily="monospace",
                fontweight="bold", color=COLORS["ttt"])
        ax.text(6.3, y + 0.33, desc, fontsize=7.5, color="#7F8C8D")

    # Output
    _draw_box(ax, 3, 0.2, 4, 0.6, "Z → RMSNorm → Output Projection",
              color=COLORS["ttt"], fontsize=9)

    # Key insight box
    box = FancyBboxPatch(
        (8, 2.5), 3.5, 2.0,
        boxstyle="round,pad=0.1",
        facecolor="#FDEDEC", edgecolor=COLORS["ttt"],
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(9.75, 4.1, "Key Insight", fontsize=10, ha="center",
            fontweight="bold", color=COLORS["ttt"])
    ax.text(8.2, 3.6,
            "When η=const, W₀=0:\n"
            "Z = η·tril(QK^T)·V\n"
            "= causal linear attention!\n\n"
            "TTT adds bias (QW₀^T)\n"
            "and corrective errors.",
            fontsize=7.5, fontfamily="monospace", color=COLORS["text"])


# ─── Page 4: Gated DeltaNet ──────────────────────────────────────────────────

def _draw_deltanet_diagram(fig):
    """Detailed Gated DeltaNet data flow."""
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Gated DeltaNet: Linear Attention with Delta Rule",
                 fontsize=14, fontweight="bold", pad=20)

    # Input
    _draw_box(ax, 4, 9.0, 3, 0.5, "Input x ∈ R^D", color="#7FB3D8", fontsize=10)

    # Projections
    projs = [
        (0, "Q proj", "#9B59B6"),
        (2.2, "K proj", "#9B59B6"),
        (4.4, "V proj", "#9B59B6"),
        (6.6, "β proj", "#E67E22"),
        (8.8, "α proj", "#E67E22"),
    ]
    for x, label, color in projs:
        _draw_box(ax, x, 8.0, 2, 0.5, label, color=color, fontsize=8)
        _draw_arrow(ax, 5.5, 9.0, x + 1, 8.5)

    # Short Conv
    ax.text(0.5, 7.3, "↓ ShortConv1d (k=4)", fontsize=8, color="#7F8C8D")
    _draw_box(ax, 0, 6.6, 6.5, 0.5, "Causal depthwise conv on Q, K, V (local context)",
              color="#D2B4DE", fontsize=8, text_color="#2C3E50")

    # Norms and gates
    ax.text(0.5, 6.0, "↓ L2 normalize Q, K        ↓ sigmoid(β)        ↓ logsigmoid(α) = gk",
            fontsize=8, color="#7F8C8D")

    # Delta rule recurrence
    _draw_box(ax, 0.5, 4.5, 10, 1.2,
              "Delta Rule Recurrence (per timestep t)",
              color=COLORS["deltanet"], fontsize=11, alpha=0.3)

    # Steps inside
    equations = [
        "S_t = diag(exp(gk_t)) · S_{t-1}              ← DECAY (forget old)",
        "S_t = S_t - β_t · k_t · (k_t^T · S_t)       ← ERASE (clear slot for k_t)",
        "S_t = S_t + β_t · k_t · v_t^T                ← WRITE (store new association)",
        "o_t = q_t^T · S_t                              ← READ  (query the memory)",
    ]
    for i, eq in enumerate(equations):
        y = 5.4 - i * 0.28
        ax.text(1, y, eq, fontsize=7.5, fontfamily="monospace", color=COLORS["text"])

    # Output processing
    _draw_box(ax, 2, 3.0, 7, 0.5, "GroupNorm → SiLU gate → Output Projection",
              color=COLORS["deltanet"], fontsize=9)
    _draw_arrow(ax, 5.5, 4.5, 5.5, 3.5)

    # State diagram (right side)
    _draw_box(ax, 8, 7.5, 3.5, 1.5,
              "State: S ∈ R^{D×D}\n\nFixed size per layer\nO(1) memory\nO(T) computation",
              color="#FADBD8", text_color="#2C3E50", fontsize=8)

    # Production deployments
    box = FancyBboxPatch(
        (0.5, 0.3), 10, 2.2,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["light"], edgecolor=COLORS["deltanet"],
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(5.5, 2.2, "Production Deployments & Connections",
            fontsize=10, ha="center", fontweight="bold", color=COLORS["deltanet"])
    ax.text(0.8, 1.7,
            "• Qwen3.5 (Alibaba): 3:1 hybrid (3 DeltaNet : 1 softmax attention)\n"
            "• Kimi Linear (Moonshot AI): DeltaNet backbone\n"
            "• Equivalence: TTT-Linear without LayerNorm + batch=1 ≡ DeltaNet (Yang 2026)\n"
            "• FLA library: Triton kernels for O(T) chunk-parallel training",
            fontsize=8, color=COLORS["text"], linespacing=1.6)


# ─── Page 5: Persistent Memory & RMT ─────────────────────────────────────────

def _draw_persistent_rmt(fig):
    """Persistent Memory and RMT diagrams side by side."""
    # Top: Persistent Memory
    ax1 = fig.add_axes([0.05, 0.52, 0.9, 0.43])
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 5)
    ax1.axis("off")
    ax1.set_title("Persistent Memory Tokens", fontsize=13, fontweight="bold")

    # Attention matrix visualization
    # Show the asymmetric mask: memory always visible, regular causal
    ax1.text(0.5, 4.3, "Attention mask: memory tokens always visible, regular tokens causal",
             fontsize=9, color=COLORS["text"])

    # Draw attention matrix
    n_mem = 3
    n_reg = 6
    n_total = n_mem + n_reg
    cell_size = 0.35
    x_start = 1
    y_start = 3.8

    for i in range(n_total):
        for j in range(n_total):
            x = x_start + j * cell_size
            y = y_start - i * cell_size

            # Memory columns always visible
            if j < n_mem:
                color = COLORS["persistent"]
                alpha = 0.8
            # Regular positions: causal
            elif i >= n_mem and j >= n_mem and j - n_mem <= i - n_mem:
                color = COLORS["baseline"]
                alpha = 0.6
            else:
                color = "white"
                alpha = 0.3

            rect = patches.Rectangle(
                (x, y), cell_size, cell_size,
                facecolor=color, edgecolor="#BDC3C7",
                linewidth=0.5, alpha=alpha,
            )
            ax1.add_patch(rect)

    # Labels
    ax1.text(x_start + n_mem * cell_size / 2, y_start + 0.15, "mem", fontsize=7,
             ha="center", color=COLORS["persistent"], fontweight="bold")
    ax1.text(x_start + (n_mem + n_reg / 2) * cell_size, y_start + 0.15, "regular", fontsize=7,
             ha="center", color=COLORS["baseline"], fontweight="bold")

    # Math
    ax1.text(5, 3.5,
             "K_aug = [K_mem ; K]    V_aug = [V_mem ; V]\n"
             "Attn(Q, K_aug, V_aug)\n\n"
             "Zero-init safety:\n"
             "V_mem = scale · V_raw,  scale ≈ 0 at init\n"
             "→ Mechanism is identity at start",
             fontsize=8.5, fontfamily="monospace", color=COLORS["text"])

    # Params
    ax1.text(5, 1.1, "Parameters: K × n_layer × 2 × (n_kv_head × head_dim)\n"
             "K=32, d12: 32 × 12 × 2 × 768 = 589,824 (~0.6M)",
             fontsize=8, color="#7F8C8D")

    # Bottom: RMT
    ax2 = fig.add_axes([0.05, 0.02, 0.9, 0.45])
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 5)
    ax2.axis("off")
    ax2.set_title("Recurrent Memory Transformer (RMT)", fontsize=13, fontweight="bold")

    # Segment diagram
    segments = [
        ("Segment 1", 0.5, COLORS["rmt"]),
        ("Segment 2", 4.0, COLORS["rmt"]),
        ("Segment 3", 7.5, COLORS["rmt"]),
    ]

    for label, x, color in segments:
        # Memory tokens
        _draw_box(ax2, x, 3.0, 0.8, 0.6, "M", color="#F4D03F", text_color="#2C3E50", fontsize=9)
        # Segment tokens
        _draw_box(ax2, x + 1.0, 3.0, 2.0, 0.6, "x₁..x_S", color=color, fontsize=9)
        ax2.text(x + 1.5, 3.8, label, fontsize=8, ha="center", color=COLORS["text"])

    # Arrows between segments (memory carries forward)
    for i in range(2):
        x1 = segments[i][1] + 3.2
        x2 = segments[i + 1][1]
        _draw_arrow(ax2, x1, 3.3, x2, 3.3, color="#F4D03F", lw=2.5)
        ax2.text((x1 + x2) / 2, 3.55, "m'→", fontsize=8, ha="center",
                 color="#F39C12", fontweight="bold")

    # BPTT diagram
    ax2.text(0.5, 2.2,
             "BPTT: gradients flow back through N segments (default N=2)\n"
             "Memory projection: m' = W_proj · m + b  (learned write gate)\n"
             "Parameters: M × D (memory init) + D² (projection)\n"
             "M=16, d12: 16 × 768 + 768² = 602,112 (~0.6M)",
             fontsize=8.5, fontfamily="monospace", color=COLORS["text"], linespacing=1.5)

    # Key properties
    ax2.text(0.5, 0.6,
             "Key: Information bottleneck forces compression. Memory tokens must learn\n"
             "to compress and carry the most useful information between segments.",
             fontsize=8.5, style="italic", color="#7F8C8D")


# ─── Page 6: Comparison Table ─────────────────────────────────────────────────

def _draw_comparison_table(fig):
    """Side-by-side comparison of all mechanisms."""
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Memory Mechanism Comparison", fontsize=14, fontweight="bold", pad=20)

    # Table data
    headers = ["Property", "Persistent", "RMT", "TTT-Linear", "DeltaNet"]
    rows = [
        ["State type", "K learned KV\nvectors/layer", "M hidden state\nvectors", "D×D weight\nmatrix W", "D×D state\nmatrix S"],
        ["State size", "O(K·D)", "O(M·D)", "O(D²)", "O(D²)"],
        ["Per-step cost", "O(1) extra", "O(M·T)", "O(C²·D + D²)", "O(D²)"],
        ["Training", "Standard", "Segment-level\n+ BPTT", "Chunk-parallel\n(dual form)", "Chunk-parallel\n(WY repr.)"],
        ["Recurrent?", "No", "Yes (segments)", "Yes (chunks)", "Yes (tokens)"],
        ["Modifications", "Add KV pairs\nto attention", "Prepend tokens\nat embedding", "Replace attention\nlayer entirely", "Replace attention\nlayer entirely"],
        ["Key reference", "Burtsev 2020", "Bulatov 2022", "Sun 2024\nLaCT 2026", "Yang 2025\n(ICLR)"],
        ["Production", "—", "—", "—", "Qwen3.5\nKimi Linear"],
    ]

    # Draw table
    col_widths = [2.0, 2.2, 2.2, 2.5, 2.5]
    row_height = 0.85
    x_start = 0.3
    y_start = 9.0

    header_colors = [COLORS["border"], COLORS["persistent"], COLORS["rmt"],
                     COLORS["ttt"], COLORS["deltanet"]]

    # Headers
    x = x_start
    for j, (header, color, w) in enumerate(zip(headers, header_colors, col_widths)):
        rect = patches.Rectangle(
            (x, y_start), w, row_height,
            facecolor=color, edgecolor="white", linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y_start + row_height / 2, header,
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        x += w

    # Data rows
    for i, row in enumerate(rows):
        y = y_start - (i + 1) * row_height
        x = x_start
        bg = COLORS["bg"] if i % 2 == 0 else "white"

        for j, (cell, w) in enumerate(zip(row, col_widths)):
            rect = patches.Rectangle(
                (x, y), w, row_height,
                facecolor=bg, edgecolor="#BDC3C7", linewidth=0.5,
            )
            ax.add_patch(rect)
            ax.text(x + w / 2, y + row_height / 2, cell,
                    ha="center", va="center", fontsize=7.5, color=COLORS["text"])
            x += w

    # Footer note
    ax.text(0.5, 0.5,
            "All mechanisms tested at same scale (d12 ≈ 85M params) on FineWeb with nanochat backbone.\n"
            "Extra parameters budgeted from baseline model capacity for fair comparison.",
            fontsize=9, color="#7F8C8D", style="italic")


# ─── Page 7: Evaluation Protocol ─────────────────────────────────────────────

def _draw_eval_protocol(fig):
    """Evaluation protocol and tasks."""
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Evaluation Protocol", fontsize=14, fontweight="bold", pad=20)

    tasks = [
        ("Perplexity (BPB)", COLORS["baseline"],
         "Bits per byte on FineWeb validation set\n"
         "Vocab-size-invariant, directly comparable\n"
         "Primary metric for language modeling quality"),
        ("NIAH (Passkey Retrieval)", COLORS["rmt"],
         "Insert 6-digit passkey at varying positions\n"
         "Context: 256, 512, 1024, 2048 tokens\n"
         "Tests: information retention across context"),
        ("Associative Recall", COLORS["ttt"],
         "Present K key-value pairs, query for specific value\n"
         "Vary: num pairs (4-32), filler distance (0-512)\n"
         "Tests: arbitrary association storage/retrieval"),
        ("MQAR (Multi-Query)", COLORS["deltanet"],
         "Multiple queries about same KV pairs\n"
         "Tests: simultaneous multi-association maintenance\n"
         "From Zoology benchmark (Arora et al. 2024)"),
        ("Copy & Selective Copy", COLORS["persistent"],
         "Reproduce input sequence (exact match)\n"
         "Selective: copy only marked [bracketed] tokens\n"
         "Tests: exact token-level memory"),
        ("BPB vs Position", "#7F8C8D",
         "Per-position loss analysis on natural text\n"
         "Memory mechanisms should improve later positions\n"
         "Tests: accumulated context benefit"),
    ]

    for i, (name, color, desc) in enumerate(tasks):
        y = 8.5 - i * 1.4
        _draw_box(ax, 0.5, y, 3.5, 1.0, name, color=color, fontsize=9)
        ax.text(4.3, y + 0.5, desc, fontsize=8, va="center", color=COLORS["text"],
                linespacing=1.4)

    # Controls
    ax.text(0.5, 0.5,
            "Controls: 3 seeds (42, 1337, 3141) · d12 primary · Parameter-matched",
            fontsize=9, color="#7F8C8D", fontweight="bold")


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_architecture_pdf(output_path: str = "architecture.pdf"):
    """Generate the full multi-page architecture PDF."""
    with PdfPages(output_path) as pdf:
        # Page 1: Title
        fig = plt.figure(figsize=(11, 8.5))
        _draw_title_page(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Transformer architecture
        fig = plt.figure(figsize=(11, 8.5))
        _draw_transformer_architecture(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: TTT-Linear
        fig = plt.figure(figsize=(11, 8.5))
        _draw_ttt_diagram(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Gated DeltaNet
        fig = plt.figure(figsize=(11, 8.5))
        _draw_deltanet_diagram(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5: Persistent + RMT
        fig = plt.figure(figsize=(11, 8.5))
        _draw_persistent_rmt(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 6: Comparison table
        fig = plt.figure(figsize=(11, 8.5))
        _draw_comparison_table(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 7: Evaluation protocol
        fig = plt.figure(figsize=(11, 8.5))
        _draw_eval_protocol(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Architecture PDF saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate memory-bench architecture PDF")
    parser.add_argument("--output", type=str, default="architecture.pdf")
    args = parser.parse_args()
    generate_architecture_pdf(args.output)
