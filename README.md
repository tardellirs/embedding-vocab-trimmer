# How to trim an embedding model's vocabulary for a target language

> **`embedding-vocab-trimmer`** — reduce a multilingual text-embedding model to a single target language with **no training and no GPU**, by surgically trimming its token vocabulary.

In a multilingual embedding model, the token-embedding matrix dominates the parameter count
(EmbeddingGemma-300M: $262{,}144 \times 768 \approx 201\text{M}$ of ${\sim}308\text{M}$ total params).
When the model is deployed for a single language, all other language embeddings are unused.
`trim_vocab.py` identifies the tokens that language actually uses, retains the top-$K$ by corpus
frequency, re-indexes the embedding matrix, and rewrites the BPE merge table —
leaving the transformer encoder and the SentenceTransformers pooling/Dense heads **bit-for-bit unchanged**.

> **Result on Portuguese (EmbeddingGemma-300M → 157M):** the 64k-token trim retains **99.4%** of
> the full model's MTEB(por) score at **half the parameters** — with zero training.

📦 Example model: **[`tardellirs/embeddinggemma-pt-br`](https://huggingface.co/tardellirs/embeddinggemma-pt-br)** · 🛠️ Tool: [github.com/tardellirs/embedding-vocab-trimmer](https://github.com/tardellirs/embedding-vocab-trimmer)

![Parameter composition and MTEB(por) score by vocabulary size — EmbeddingGemma-300M (PT)](results/composition.png)

---

## Method

Let $\mathcal{V}$ be the full vocabulary of the multilingual model, $|\mathcal{V}| = V$, and $d$ the embedding dimension. The embedding matrix is $E \in \mathbb{R}^{V \times d}$, where row $E_v$ is the embedding of token $v \in \mathcal{V}$.

### 1. Corpus-based frequency estimation

Given a target-language corpus $\mathcal{C}$, tokenize it with the original tokenizer $\tau$ and count token occurrences:

$$f(v) = \sum_{x \in \mathcal{C}} \sum_{t \in \tau(x)} \mathbf{1}[t = v], \qquad v \in \mathcal{V}$$

### 2. Vocabulary selection

Let $\mathcal{S} \subset \mathcal{V}$ be the set of mandatory special tokens (pad, bos, eos, unk, and high-frequency byte-fallback tokens). The trimmed vocabulary of size $K$ is:

$$\mathcal{V}_K = \underset{v \,\in\, \mathcal{V} \setminus \mathcal{S}}{\operatorname{Top-}K}\{f(v)\} \;\cup\; \mathcal{S}$$

A contiguous re-indexing bijection $\sigma: \mathcal{V}_K \to \{0, \ldots, |\mathcal{V}_K|-1\}$ is then constructed, preserving the original relative order of token ids.

### 3. BPE merge consistency

A BPE vocabulary is defined by an ordered list of merge rules $\mathcal{M} = \{(a_i, b_i) \to c_i\}$. A merge is valid only when all three tokens involved survive the trim:

$$\mathcal{M}_K = \{(a, b) \to c \;\in\; \mathcal{M} \;\mid\; a \in \mathcal{V}_K \;\land\; b \in \mathcal{V}_K \;\land\; c \in \mathcal{V}_K\}$$

This is the critical step most implementations overlook: retaining a merge whose product $c \notin \mathcal{V}_K$ causes the tokenizer to emit a token id that no longer exists in the embedding matrix, silently producing garbage embeddings or an index error at inference time.

### 4. Embedding submatrix extraction

The trimmed embedding matrix $E_K \in \mathbb{R}^{|\mathcal{V}_K| \times d}$ is obtained by selecting the rows corresponding to surviving tokens in the new index order:

$$E_K = E\bigl[\sigma^{-1}(0),\; \sigma^{-1}(1),\; \ldots,\; \sigma^{-1}(|\mathcal{V}_K|-1)\bigr]$$

The encoder weights $\theta_\text{enc}$, pooling layer, and Dense projection heads are copied unchanged. The full trimmed model is:

$$\theta_K = \bigl(E_K,\; \theta_\text{enc},\; \theta_\text{pool},\; \theta_\text{dense}\bigr)$$

For every surviving token $v \in \mathcal{V}_K$, the embedding $E_K[\sigma(v)]$ is bit-for-bit identical to $E[v]$. No weight is modified, fine-tuned, or distilled.

### 5. Parameter reduction

$$P = V \cdot d + P_\text{enc}, \qquad P_K = |\mathcal{V}_K| \cdot d + P_\text{enc}, \qquad \Delta P = (V - |\mathcal{V}_K|) \cdot d$$

For EmbeddingGemma-300M ($V = 262{,}144$, $d = 768$, $P_\text{enc} \approx 107\text{M}$) trimmed to $K = 64{,}000$:

$$\Delta P = (262{,}144 - 64{,}000) \times 768 \approx 152\text{M parameters} \quad (-49\%)$$

### Quality preservation

Because the encoder is identical across all trim sizes, quality loss arises solely from tokenization changes for out-of-vocabulary tokens. As $K$ grows, the coverage of the language's actual token distribution approaches unity and the score converges to the untrimmed baseline:

$$\lim_{K \to V} \operatorname{MTEB}(f_{\theta_K}) = \operatorname{MTEB}(f_\theta)$$

Empirically, convergence is fast: at $K = 64{,}000$ on Portuguese, $\operatorname{MTEB}(f_{\theta_K}) / \operatorname{MTEB}(f_\theta) = 99.4\%$.

---

## Architecture

```
multilingual model                         language-trimmed model
┌───────────────────────────┐              ┌────────────────────────┐
│ embed_tokens  262144×768  │  ── trim ──▶ │ embed_tokens  64000×768 │  ← only this shrinks
├───────────────────────────┤              ├────────────────────────┤
│ transformer encoder       │  (unchanged) │ transformer encoder     │
│ pooling + Dense heads     │  (unchanged) │ pooling + Dense heads   │
└───────────────────────────┘              └────────────────────────┘
        ~308M params                               ~157M params
```

---

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# trim EmbeddingGemma-300M to a 64k Portuguese vocabulary
python trim_vocab.py \
    --model google/embeddinggemma-300m \
    --corpus-config por \
    --vocab-size 64000 \
    --output ./embeddinggemma-pt-br

# push to the Hub (requires HF_TOKEN)
python trim_vocab.py --model google/embeddinggemma-300m --corpus-config por \
    --vocab-size 64000 --output ./out --push <user>/embeddinggemma-pt-br
```

`--corpus-config` is the language code for the mining corpus (defaults to
[`lbourdois/fineweb-2-trimming`](https://huggingface.co/datasets/lbourdois/fineweb-2-trimming),
e.g. `por`, `fra`, `deu`, `spa`). Pass `--corpus-dataset` to mine from any other HuggingFace text dataset.

## Results — Portuguese (EmbeddingGemma-300M)

Evaluated on **MTEB(por)** — 22 native PT-BR tasks spanning classification, pair-classification, STS,
clustering, retrieval, and reranking (`mean_22`). The transformer encoder and Dense heads are **identical
at every vocab size**; the only variable is the embedding matrix.

| vocab | params | MTEB(por) `mean_22` | % of full |
|------:|-------:|:-------------------:|:---------:|
| 16k   | ~119M  | 0.5950 | 91.7% |
| 24k   | ~125M  | 0.6263 | 96.5% |
| 32k   | ~131M  | 0.6201 | 95.5% |
| 48k   | ~144M  | 0.6418 | 98.9% |
| **64k** | **~157M** | **0.6453** | **99.4%** |
| 128k  | ~207M  | 0.6491 | ≈100% |
| *full EG-300M* | *~308M* | *0.6490* | *100%* |

Quality recovers monotonically above 32k. At 64k the trim reaches 99.4% of the full model's score
at 51% of the parameters. The 128k model ties the full model within measurement noise.
See [`results/`](results/) and [`examples/embeddinggemma_pt.md`](examples/embeddinggemma_pt.md).

![MTEB(por) vs. params](results/pareto.png)

## Limitations

- **Compression, not enhancement.** Vocabulary trimming removes unused parameters; it does not improve
  the model. Fine-tuning, layer pruning, and distillation from a larger teacher were all evaluated and
  each reduced MTEB(por) by 0.02–0.04 points. The base model is at its representational ceiling for the
  target language; trimming recovers deployment efficiency at no quality cost, but cannot exceed it.
- **Tokenizer family.** Validated on BPE with `byte_fallback` (Gemma/EmbeddingGemma). The method
  generalises to other BPE/SentencePiece embedders; merge-filtering logic may require minor adaptation
  per family.
- **Architecture.** Targets SentenceTransformers models with a `transformers` encoder and an
  `embed_tokens` weight matrix. The encoder, pooling, and Dense heads pass through untouched.

## License

Tool: **Apache-2.0** (see [`LICENSE`](LICENSE)). The example model is derived from Google's
EmbeddingGemma and is released under the **[Gemma Terms of Use](https://ai.google.dev/gemma/terms)**
(see [`NOTICE`](NOTICE)). Trimmed models inherit the license of their base model.

## Citation

If this work is useful, a link or star is appreciated.
Benchmark: [MTEB(por) leaderboard](https://huggingface.co/spaces/mteb-pt/leaderboard).
