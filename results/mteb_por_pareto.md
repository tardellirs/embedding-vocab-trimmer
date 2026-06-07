# MTEB(por) vocabulary-size Pareto — EmbeddingGemma-300M

Base model: `google/embeddinggemma-300m`. Each row is the same model with its token vocabulary
trimmed to the target size — **the transformer encoder and the pooling/Dense heads are identical
across all rows**; only the embedding matrix changes. Trimming is training-free.

Metric: **MTEB(por)** headline `mean_16` (16 tasks — classification, pair-classification, STS,
clustering, retrieval, reranking). Higher is better.

| vocab | params | `mean_16` | Δ vs full | notes |
|------:|-------:|:---------:|:---------:|-------|
| 16k   | ~119M  | 0.6520 | −0.0737 | over-trimmed; fine retrieval degrades |
| 24k   | ~125M  | 0.6895 | −0.0362 | smallest practical point |
| 32k   | ~131M  | 0.6881 | −0.0376 | ≈ 24k (within noise) |
| 48k   | ~144M  | 0.7098 | −0.0159 | |
| **64k** | **~157M** | **0.7172** | **−0.0085** | **sweet spot — ≈ full quality, ½ params** |
| full  | ~308M  | 0.7257 | — | untrimmed EmbeddingGemma-300M |

**Reading the curve:** quality recovers monotonically above 32k. At 64k the trim keeps **98.8%** of
the full model's score with **51%** of the parameters. Below ~24k the model loses the fine-grained
distinctions that retrieval/reranking depend on, so the score drops faster than the parameter count.

Reproduce any row with:

```bash
python trim_vocab.py --model google/embeddinggemma-300m --corpus-config por --vocab-size 64000 --output ./out
```

Evaluation harness: the public MTEB(por) benchmark *(release coming soon)*. Numbers above are full-suite
`mean_16`; a fast in-loop proxy that drops the largest retrieval pool inflates absolute scores and should
not be used for headline numbers.
