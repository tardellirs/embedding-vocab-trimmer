# MTEB(por) vocabulary-size Pareto — EmbeddingGemma-300M

Base model: `google/embeddinggemma-300m`. Each row is the same model with its token vocabulary
trimmed to the target size — **the transformer encoder and the pooling/Dense heads are identical
across all rows**; only the embedding matrix changes. Trimming is training-free.

## MTEB(por, v2) — 22 tasks (current)

Metric: **MTEB(por, v2)** `mean_22` (22 native PT-BR tasks — classification, multilabel-classification,
pair-classification, STS, clustering, retrieval, reranking). Higher is better.

| vocab | params | `mean_22` | % of full | notes |
|------:|-------:|:---------:|:---------:|-------|
| 16k   | ~119M  | 0.5950 | 91.7% | over-trimmed; fine retrieval degrades |
| 24k   | ~125M  | 0.6263 | 96.5% | smallest practical point |
| 32k   | ~131M  | 0.6201 | 95.5% | ≈ 24k (within noise) |
| 48k   | ~144M  | 0.6418 | 98.9% | |
| **64k** | **~157M** | **0.6453** | **99.4%** | **sweet spot — ≈ full quality, ½ params** |
| 128k  | ~207M  | 0.6491 | ≈100% | ties full model (within noise) |
| full  | ~308M  | 0.6490 | 100% | untrimmed EmbeddingGemma-300M |

**Reading the curve:** quality recovers monotonically above 32k. At 64k the trim keeps **99.4%** of
the full model's score with **51%** of the parameters. Below ~24k the model loses the fine-grained
distinctions that retrieval/reranking depend on, so the score drops faster than the parameter count.
The 128k model ties the full model on v2 (0.6491 vs 0.6490 — within noise).

## MTEB(por, v1) — 16 tasks (reference)

Metric: `mean_16` (16 tasks). Superseded by v2; kept for reference.

| vocab | params | `mean_16` | % of full |
|------:|-------:|:---------:|:---------:|
| 16k   | ~119M  | 0.6520 | 89.8% |
| 24k   | ~125M  | 0.6895 | 95.0% |
| 32k   | ~131M  | 0.6881 | 94.8% |
| 48k   | ~144M  | 0.7098 | 97.8% |
| **64k** | **~157M** | **0.7172** | **98.8%** |
| 128k  | ~207M  | 0.7192 | 99.1% |
| full  | ~308M  | 0.7257 | 100% |

---

Reproduce any row with:

```bash
python trim_vocab.py --model google/embeddinggemma-300m --corpus-config por --vocab-size 64000 --output ./out
```

Evaluation harness: the public MTEB(por) benchmark ([leaderboard](https://huggingface.co/spaces/mteb-pt/leaderboard)).
