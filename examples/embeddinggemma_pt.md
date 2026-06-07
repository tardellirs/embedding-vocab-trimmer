# Worked example — EmbeddingGemma-300M → Portuguese

This reproduces the flagship model
[`tardellirs/embeddinggemma-pt-br`](https://huggingface.co/tardellirs/embeddinggemma-pt-br):
a 64k-vocabulary, ~157M-param Portuguese embedder trimmed from the 308M multilingual EmbeddingGemma-300M.

## Run it

```bash
python trim_vocab.py \
    --model google/embeddinggemma-300m \
    --corpus-config por \
    --vocab-size 64000 \
    --n-texts 200000 \
    --output ./embeddinggemma-pt-br
```

`google/embeddinggemma-300m` is a **gated** model — accept the license on its Hugging Face page and set
`HF_TOKEN` first. Mining 200k texts + slicing runs on CPU in a few minutes.

## Expected output (abridged)

```
base vocab=262144, merges=...
mining token frequencies on 'por' ...
  mined corpus: 200000 texts (column 'text')
  forced special tokens kept: ...
  merges <N> -> <M> (dropped ...)
trimmed embedding (262144, 768) -> (64000, 768) | total params ~157,000,000
  smoke: related=0.7xx  unrelated=0.1xx  -> OK
```

## Use the trimmed model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("tardellirs/embeddinggemma-pt-br")  # or your local ./output dir
docs = ["O Brasil é um país tropical da América do Sul.",
        "Operações matemáticas envolvem soma e multiplicação."]
emb = model.encode(docs, normalize_embeddings=True)
print(emb.shape)  # (2, 768)
```

EmbeddingGemma uses task-specific prompts; for retrieval prepend the query/document prompts exactly as
the base model card documents (`task: search result | query: ` / `title: none | text: `).

## What we observed

| vocab | params | MTEB(por) `mean_16` |
|------:|-------:|:-------------------:|
| 64k   | ~157M  | 0.7172 (≈ full EG-300M's 0.7257, at half the params) |

See [`../results/mteb_por_pareto.md`](../results/mteb_por_pareto.md) for the full size sweep, and the
**Scope & limitations** section of the main README for why fine-tuning / pruning / distillation do **not**
push this number higher (the base model is at its representational ceiling).
