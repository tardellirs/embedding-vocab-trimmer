"""Render results/pareto.png from mteb_por_pareto.csv.  Run:  python3 results/plot_pareto.py"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(here, "mteb_por_pareto.csv"))))
trim = [r for r in rows if r["vocab_size"] != "full"]
full = next(r for r in rows if r["vocab_size"] == "full")

xs = [float(r["params_millions"]) for r in trim]
ys = [float(r["mteb_por_mean16"]) for r in trim]
labels = [f'{int(r["vocab_size"])//1024}k' for r in trim]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
ax.plot(xs, ys, "-o", color="#2563eb", lw=2, ms=7, label="vocab-trimmed (training-free)")
fx, fy = float(full["params_millions"]), float(full["mteb_por_mean16"])
ax.plot([fx], [fy], "*", color="#dc2626", ms=18, label="full EmbeddingGemma-300M")

for x, y, l in zip(xs, ys, labels):
    ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, -11), fontsize=9, color="#334155")
ax.annotate("full", (fx, fy), textcoords="offset points", xytext=(-8, 8), fontsize=9, color="#dc2626")
# highlight the 64k sweet spot
sx, sy = xs[labels.index("64k")], ys[labels.index("64k")]
ax.annotate("sweet spot\n≈ full @ ½ params", (sx, sy), textcoords="offset points",
            xytext=(-130, -2), fontsize=9, color="#2563eb",
            arrowprops=dict(arrowstyle="->", color="#2563eb"))

ax.set_xlabel("parameters (millions)")
ax.set_ylabel("MTEB(por)  mean_16")
ax.set_title("Vocabulary trimming: quality vs. size — EmbeddingGemma-300M (PT)")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(here, "pareto.png"))
print("wrote", os.path.join(here, "pareto.png"))
