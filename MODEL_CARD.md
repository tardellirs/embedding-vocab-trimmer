---
license: gemma
language:
- pt
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: google/embeddinggemma-300m
tags:
- sentence-transformers
- embeddings
- portuguese
- vocabulary-trimming
- mteb
---

# embeddinggemma-pt-br

A **Portuguese-only** text-embedding model, **vocabulary-trimmed** from
[`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m).
It keeps the 64k most frequent Portuguese tokens and drops the rest of the multilingual
vocabulary — shrinking the model from **~308M to ~157M parameters (≈ half)** while keeping
**98.8%** of the full model's MTEB(por) score. **No training was involved** — only the token
embedding matrix was sliced; the transformer encoder and the pooling/Dense heads are identical
to the base model.

Produced with the open-source tool 🛠️ **[embedding-vocab-trimmer](https://github.com/tardellirs/embedding-vocab-trimmer)**.

## Results — MTEB(por)

| model | params | MTEB(por) `mean_16` |
|---|---:|:---:|
| google/embeddinggemma-300m | ~308M | 0.7257 |
| **embeddinggemma-pt-br (this)** | **~157M** | **0.7172** |

`mean_16` = the 16 headline MTEB(por) tasks (classification, pair-classification, STS, clustering,
retrieval, reranking). Full size sweep (16k/24k/32k/48k/64k) is in the
[tool's results](https://github.com/tardellirs/embedding-vocab-trimmer/blob/main/results/mteb_por_pareto.md).

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("tardellirs/embeddinggemma-pt-br")
emb = model.encode(
    ["O Brasil é um país tropical da América do Sul.",
     "Operações matemáticas envolvem soma e multiplicação."],
    normalize_embeddings=True,
)
print(emb.shape)  # (2, 768)
```

This model inherits EmbeddingGemma's **task-specific prompts**. For retrieval, prepend the prompts
from the base model card, e.g. `task: search result | query: ` for queries and
`title: none | text: ` for documents. Supports Matryoshka output dims (768/512/256/128) via the base
model's Dense heads.

## How it was made

Mine Portuguese token frequencies → keep top-64k + functional specials → re-index the vocabulary →
filter the BPE merges (keep `A B → AB` only if A, B and AB all survive) → slice `embed_tokens.weight`
→ reattach the original encoder + pooling/Dense. Reproduce:

```bash
python trim_vocab.py --model google/embeddinggemma-300m --corpus-config por \
    --vocab-size 64000 --output ./embeddinggemma-pt-br
```

## Scope, transparency & limitations

- **This is a compression of Google's EmbeddingGemma.** Its quality, training data and behaviour come
  entirely from the base model — this is a **deployment / efficiency artifact**. Data provenance is
  exactly that of EmbeddingGemma.
- Vocabulary trimming **compresses**; it does not enhance. Fine-tuning, pruning and distillation from a
  larger teacher were all tried and **reduced** MTEB(por) — the base model is at its representational
  ceiling.
- Portuguese only — other languages fall back to byte-level tokenization and will be poor.

## License

Derived from EmbeddingGemma and distributed under the
[**Gemma Terms of Use**](https://ai.google.dev/gemma/terms). The trimming tool is Apache-2.0.

*Benchmark: MTEB(por) — public release coming soon (citation/link to be added).*
