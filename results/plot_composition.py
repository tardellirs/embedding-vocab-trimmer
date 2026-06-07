"""Render results/composition.png — stacked parameter composition per vocab size + MTEB(por) score on top.
Run:  python3 results/plot_composition.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HID = 768
TRANSFORMER = 101.54   # transformer encoder (M) — identical across all trims
DENSE = 4.72           # 2x Dense projection heads (M) — identical across all trims

# (label, vocab_size, mteb_por_mean16) — ordered largest -> smallest to show progressive trimming
MODELS = [
    ("Original", 262144, 0.7257),
    ("128k",     131072, 0.7192),
    ("64k",       65536, 0.7172),
    ("48k",       49152, 0.7098),
    ("32k",       32768, 0.6881),
    ("24k",       24576, 0.6895),
    ("16k",       16384, 0.6520),
]

# professional palette
C_ENC = "#6366f1"   # indigo  - transformer encoder (constant)
C_EMB = "#f59e0b"   # amber   - embedding matrix (the part that shrinks)
C_DEN = "#14b8a6"   # teal    - Dense heads (constant)
SLATE = "#1e293b"
ACCENT = "#b45309"  # deep amber for the MTEB score text

labels = [m[0] for m in MODELS]
emb = [v * HID / 1e6 for _, v, _ in MODELS]
totals = [TRANSFORMER + e + DENSE for e in emb]
scores = [s for *_, s in MODELS]
x = range(len(MODELS))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fbfcfe")
bw = 0.6

ax.bar(x, [TRANSFORMER] * len(MODELS), bw, color=C_ENC, label="Transformer encoder", zorder=3)
ax.bar(x, emb, bw, bottom=[TRANSFORMER] * len(MODELS), color=C_EMB, label="Embedding matrix", zorder=3)
ax.bar(x, [DENSE] * len(MODELS), bw, bottom=[TRANSFORMER + e for e in emb], color=C_DEN,
       label="Dense layers (×2)", zorder=3)

# in-segment value labels (white) where the segment is tall enough
for xi, e in zip(x, emb):
    ax.text(xi, TRANSFORMER / 2, f"{TRANSFORMER:.0f}M", ha="center", va="center",
            color="white", fontsize=9, zorder=4)
    if e > 14:
        ax.text(xi, TRANSFORMER + e / 2, f"{e:.0f}M", ha="center", va="center",
                color="white", fontsize=9, zorder=4)

# top labels: total params (bold) + MTEB(por) score & % of full in a pill
FULL_SCORE = 0.7257
for xi, t, sc in zip(x, totals, scores):
    ax.text(xi, t + 6, f"{t:.0f}M", ha="center", va="bottom", fontweight="bold",
            color=SLATE, fontsize=10.5)
    ax.annotate(f"{sc:.3f}  ·  {sc/FULL_SCORE*100:.1f}% of full", (xi, t + 24), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=ACCENT,
                bbox=dict(boxstyle="round,pad=0.32", fc="#fff7ed", ec="#fdba74", lw=1))

ax.set_ylim(0, 360)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("parameters (millions)")
ax.set_title("EmbeddingGemma-300M — vocabulary trimming: only the embedding matrix shrinks",
             fontsize=13, fontweight="bold", color=SLATE, pad=14)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e2e8f0", lw=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#cbd5e1")
ax.tick_params(length=0)

ax.legend(handles=[Patch(color=C_DEN, label="Dense layers (×2)"),
                   Patch(color=C_EMB, label="Embedding matrix  (trimmed)"),
                   Patch(color=C_ENC, label="Transformer encoder")],
          loc="upper right", frameon=False, fontsize=9.5)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "composition.png")
fig.savefig(out)
print("wrote", out)
