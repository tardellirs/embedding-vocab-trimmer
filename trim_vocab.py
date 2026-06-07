#!/usr/bin/env python3
"""
trim_vocab.py — shrink a (BPE) text-embedding model to a single target language by
**vocabulary trimming**: keep only the tokens that language actually uses, re-index the
embedding matrix, and rewrite the merge table. This is pure embedding-matrix surgery —
**no training, no GPU required** — and it leaves the transformer encoder and the
SentenceTransformers pooling/Dense heads untouched.

Why it works: a multilingual embedding model spends most of its parameters on the token
embedding matrix (e.g. EmbeddingGemma-300M: 262k tokens x 768 = ~201M of its ~308M
params). A model used for one language only needs that language's tokens, so dropping
the rest removes a large chunk of parameters at (near) zero quality cost.

The one non-obvious step is **merge filtering**: a BPE merge rule "A B" -> "AB" must be
dropped unless A, B *and* "AB" all survive the trim, otherwise the tokenizer can produce
an id you deleted. We handle that.

Usage (CPU is fine):
    python trim_vocab.py \
        --model google/embeddinggemma-300m \
        --corpus-config por \
        --vocab-size 64000 \
        --output ./embeddinggemma-por-trim64k

    # push to the Hugging Face Hub (needs HF_TOKEN in the environment):
    python trim_vocab.py --model google/embeddinggemma-300m --corpus-config por \
        --vocab-size 64000 --output ./out --push <user>/embeddinggemma-por-trim64k

See README.md for the method, results, and limitations.
"""
import argparse
import collections
import copy
import json
import os
import shutil

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


def _merge_parts(mg):
    """A merge rule may be stored as ["A", "B"] or "A B"."""
    return mg if isinstance(mg, (list, tuple)) else mg.split(" ")


def mine_frequencies(tokenizer, corpus_dataset, corpus_config, n_texts, batch=1000):
    """Stream `n_texts` texts of the target language and count how often each token id appears."""
    from datasets import load_dataset

    ds = load_dataset(corpus_dataset, corpus_config, split="train", streaming=True)
    texts, text_col = [], None
    for ex in ds:
        if text_col is None:
            text_col = "text" if "text" in ex else next(
                k for k, v in ex.items() if isinstance(v, str) and len(v) > 20)
        t = ex.get(text_col)
        if t:
            texts.append(t)
        if len(texts) >= n_texts:
            break
    print(f"  mined corpus: {len(texts)} texts (column '{text_col}')")

    freq = collections.Counter()
    for i in range(0, len(texts), batch):
        for ids in tokenizer(texts[i:i + batch], add_special_tokens=False)["input_ids"]:
            freq.update(ids)
    print(f"  distinct token ids seen: {len(freq)}")
    return freq


def select_kept_ids(tokenizer_json, freq, vocab_size, inv_vocab):
    """Return the list of original ids to keep: forced specials first, then top-frequency, padded."""
    forced = []
    for at in tokenizer_json.get("added_tokens", []):
        c = at["content"]
        if ("unused" in c.lower()) or c.startswith("[") or "image" in c.lower():
            continue  # drop reserved / multimodal / unused slots
        forced.append(at["id"])
    forced = sorted(set(forced))
    print(f"  forced special tokens kept: {len(forced)}")

    kept, seen = list(forced), set(forced)
    for tid, _ in freq.most_common():
        if len(kept) >= vocab_size:
            break
        if tid not in seen and tid in inv_vocab:
            kept.append(tid)
            seen.add(tid)
    # if the language used fewer distinct tokens than the target, pad with low ids
    for tid in range(len(inv_vocab)):
        if len(kept) >= vocab_size:
            break
        if tid not in seen:
            kept.append(tid)
            seen.add(tid)
    return kept[:vocab_size]


def trim(model_id, corpus_dataset, corpus_config, vocab_size, n_texts, output,
         device="cpu", smoke=True):
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    print(f"downloading {model_id} ...")
    src = snapshot_download(model_id, token=token)

    tokenizer = AutoTokenizer.from_pretrained(src, token=token)
    tj = json.load(open(os.path.join(src, "tokenizer.json")))
    assert tj["model"]["type"] == "BPE", f"only BPE tokenizers supported (got {tj['model']['type']})"
    old_vocab = tj["model"]["vocab"]            # {token_str: id}
    old_merges = tj["model"]["merges"]
    inv = {i: t for t, i in old_vocab.items()}
    print(f"base vocab={len(old_vocab)}, merges={len(old_merges)}")

    print(f"mining token frequencies on '{corpus_config}' ...")
    freq = mine_frequencies(tokenizer, corpus_dataset, corpus_config, n_texts)

    kept = select_kept_ids(tj, freq, vocab_size, inv)
    assert len(kept) == vocab_size
    kept_tokens = set(inv[i] for i in kept)
    old2new = {old: new for new, old in enumerate(kept)}

    # --- rewrite tokenizer: re-index vocab, filter merges, re-index added_tokens ---
    new_vocab = {inv[old]: new for old, new in old2new.items()}
    new_merges = []
    for mg in old_merges:
        a, b = _merge_parts(mg)
        if a in kept_tokens and b in kept_tokens and (a + b) in kept_tokens:
            new_merges.append(mg)
    print(f"  merges {len(old_merges)} -> {len(new_merges)} (dropped {len(old_merges) - len(new_merges)})")

    tj2 = copy.deepcopy(tj)
    tj2["model"]["vocab"] = new_vocab
    tj2["model"]["merges"] = new_merges
    new_added = []
    for at in tj.get("added_tokens", []):
        if at["id"] in old2new:
            a = dict(at)
            a["id"] = old2new[at["id"]]
            new_added.append(a)
    new_added.sort(key=lambda a: a["id"])
    tj2["added_tokens"] = new_added

    # --- copy the model dir, rewrite tokenizer + sliced embedding + config ---
    shutil.rmtree(output, ignore_errors=True)
    shutil.copytree(src, output)
    json.dump(tj2, open(os.path.join(output, "tokenizer.json"), "w"), ensure_ascii=False)
    for fn in ("tokenizer.model", "added_tokens.json"):  # stale artifacts of the old vocab
        p = os.path.join(output, fn)
        if os.path.exists(p):
            os.remove(p)

    sd = load_file(os.path.join(src, "model.safetensors"))
    emb_key = next(k for k in sd if k.endswith("embed_tokens.weight"))
    old_emb = sd[emb_key]
    new_emb = torch.empty((vocab_size, old_emb.shape[1]), dtype=old_emb.dtype)
    for old, new in old2new.items():
        new_emb[new] = old_emb[old]
    sd[emb_key] = new_emb
    save_file(sd, os.path.join(output, "model.safetensors"), metadata={"format": "pt"})

    cfg = json.load(open(os.path.join(output, "config.json")))
    cfg["vocab_size"] = vocab_size
    json.dump(cfg, open(os.path.join(output, "config.json"), "w"))

    n_params = old_emb.shape[1] * vocab_size + sum(
        v.numel() for k, v in sd.items() if k != emb_key)
    print(f"trimmed embedding {tuple(old_emb.shape)} -> {tuple(new_emb.shape)} "
          f"| total params ~{n_params:,}")

    if smoke:
        _smoke_test(output, device)
    return output, n_params


def _smoke_test(output, device):
    """Sanity check: a related pair must score higher than an unrelated one."""
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(output, device=device, trust_remote_code=True)
    e = m.encode(
        ["O Brasil é um país tropical da América do Sul.",
         "A República Federativa do Brasil fica na América Latina.",
         "Operações matemáticas envolvem soma e multiplicação."],
        normalize_embeddings=True, convert_to_numpy=True)
    rel, unrel = float(e[0] @ e[1]), float(e[0] @ e[2])
    print(f"  smoke: related={rel:.3f}  unrelated={unrel:.3f}  -> {'OK' if rel > unrel else 'FAILED'}")
    assert rel > unrel and rel > 0.3, "smoke test failed — trim produced a degenerate model"


def main():
    ap = argparse.ArgumentParser(description="Trim a BPE text-embedding model to one language.")
    ap.add_argument("--model", default="google/embeddinggemma-300m",
                    help="base SentenceTransformers model id (BPE tokenizer)")
    ap.add_argument("--corpus-dataset", default="lbourdois/fineweb-2-trimming",
                    help="HF dataset to mine target-language token frequencies from")
    ap.add_argument("--corpus-config", default="por",
                    help="dataset config / language code (e.g. 'por' for Portuguese)")
    ap.add_argument("--vocab-size", type=int, default=64000, help="target vocabulary size")
    ap.add_argument("--n-texts", type=int, default=200000, help="texts to mine frequencies on")
    ap.add_argument("--output", default="./trimmed-model", help="output directory")
    ap.add_argument("--device", default="cpu", help="device for the smoke test (cpu/cuda)")
    ap.add_argument("--no-smoke", action="store_true", help="skip the smoke test")
    ap.add_argument("--push", default=None, help="optional HF repo id to upload to (needs HF_TOKEN)")
    ap.add_argument("--private", action="store_true", help="push as a private repo")
    args = ap.parse_args()

    out, n_params = trim(args.model, args.corpus_dataset, args.corpus_config,
                         args.vocab_size, args.n_texts, args.output,
                         device=args.device, smoke=not args.no_smoke)

    if args.push:
        from huggingface_hub import HfApi, create_repo
        token = os.environ["HF_TOKEN"]
        create_repo(args.push, private=args.private, exist_ok=True, token=token)
        HfApi().upload_folder(folder_path=out, repo_id=args.push, token=token)
        print(f"pushed -> https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
