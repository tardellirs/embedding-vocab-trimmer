#!/usr/bin/env python3
"""
gguf_bpe_patch.py — make a vocab-trimmed BPE model convert to a **faithful** GGUF via
llama.cpp's BPE path (bit-exact tokenization), instead of a reconstructed SentencePiece.

Why this is needed
------------------
`trim_vocab.py` already outputs a BPE-ready model (a re-indexed `tokenizer.json`, no
`tokenizer.model`). But when you run llama.cpp's `convert_hf_to_gguf.py` on it, convert
computes a hash (`chkhsh`) of the tokenizer's pre-tokenization of a fixed probe string to
pick the right pre-tokenizer regex — and a *trimmed* vocabulary changes that hash, so convert
bails out with:

    NotImplementedError: BPE pre-tokenizer was not recognized - update get_vocab_base_pre()

The pre-tokenizer **rules are unchanged** from the base model; only the hash differs (it hashes
a tokenization that depends on the vocab). So the fix is to tell convert that the trimmed model's
`chkhsh` uses the **same** pre-tokenizer as the base model. This tool:

  1. reads the probe string (`chktxt`) from your llama.cpp checkout,
  2. computes the `chkhsh` of both the trimmed model and the base model (the same way convert does),
  3. finds the base model's `res` (pre-tokenizer id) already registered in llama.cpp,
  4. prints — or, with `--apply`, inserts — the one-line `if chkhsh == "<trimmed>": res = "<base>"`.

This BPE path is **bit-exact** (identical tokenization to the original) and is the recommended
way to GGUF a trimmed model — for embedding models it matches the reconstructed-SPM path
(cosine ~1.0), and for LLMs used with greedy generation it is **required** (a reconstructed
unigram SPM is not bit-exact and silently degrades long-prompt generation).

Usage
-----
    python gguf_bpe_patch.py \
        --trimmed ./embeddinggemma-por-trim64k \
        --base    google/embeddinggemma-300m \
        --llama-cpp ~/llama.cpp \
        [--apply]

Then convert as usual (embedding models need the ST-dense flag; LLMs do not):

    python ~/llama.cpp/convert_hf_to_gguf.py ./embeddinggemma-por-trim64k \
        --outfile model-f16.gguf --outtype f16 --sentence-transformers-dense-modules

Note (gemma-3 LLMs only): some gemma-3 checkpoints carry a vestigial `<image_soft_token>` at
id == vocab_size that trips a separate assertion in convert's `get_vocab_base()`
(`assert max(tokenizer.vocab.values()) < vocab_size`). The loop there already caps at
vocab_size, so replacing that assert with `pass` is safe. Embedding trims produced by
`trim_vocab.py` drop multimodal/unused slots and are not affected.
"""
import argparse
import hashlib
import os
import re
import sys

from transformers import AutoTokenizer


def _find_convert_source(llama_cpp: str) -> str:
    """Locate the file that defines get_vocab_base_pre / the chktxt probe."""
    candidates = [
        os.path.join(llama_cpp, "conversion", "base.py"),   # newer layout
        os.path.join(llama_cpp, "convert_hf_to_gguf.py"),   # older single-file layout
        os.path.join(llama_cpp, "gguf-py", "gguf", "vocab.py"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            s = open(p, encoding="utf-8").read()
            if "get_vocab_base_pre" in s and "chktxt" in s:
                return p
    # last resort: scan the tree
    for root, _, files in os.walk(llama_cpp):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    s = open(p, encoding="utf-8").read()
                except Exception:
                    continue
                if "get_vocab_base_pre" in s and "chktxt" in s:
                    return p
    raise SystemExit(f"could not find get_vocab_base_pre/chktxt under {llama_cpp} "
                     "(is this a llama.cpp checkout?)")


def _extract_chktxt(src: str) -> str:
    """Pull the literal assigned to `chktxt = ...` (a multi-line string) and eval it."""
    m = re.search(r"chktxt\s*=\s*(.*?)\n\s*\n", src, re.S)
    if not m:
        raise SystemExit("could not parse the `chktxt` probe string from llama.cpp")
    return eval(m.group(1))  # llama.cpp source is trusted input


def _chkhsh(model_dir_or_id: str, chktxt: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_dir_or_id)
    return hashlib.sha256(str(tok.encode(chktxt)).encode()).hexdigest()


def _res_for_hash(src: str, h: str):
    """Return the `res` string registered for chkhsh `h` in the convert source, or None."""
    m = re.search(r'chkhsh\s*==\s*"' + re.escape(h) + r'"\s*:\s*\n\s*res\s*=\s*"([^"]+)"', src)
    return m.group(1) if m else None


def _known_res(src: str):
    """All pre-tokenizer `res` ids registered in the convert source (for the error hint)."""
    return sorted(set(re.findall(r'\n\s*res\s*=\s*"([^"]+)"', src)))


def _insert(src: str, patch: str):
    """Insert the patch into get_vocab_base_pre(), after the registered hashes and before the unrecognized-tokenizer
    branch. Newer llama.cpp raises inside `if res is None:` (after the warnings), so the patch goes before that `if`;
    put right before the `raise` it would land inside the block, after the warnings, and the raise would still run."""
    raise_re = re.search(r"\n(\s*)raise NotImplementedError\(\s*\n?\s*\"BPE pre-tokenizer was not recognized", src)
    if not raise_re:
        return None
    fn = src.rfind("def get_vocab_base_pre", 0, raise_re.start())
    guard = src.rfind("\n        if res is None:", max(fn, 0), raise_re.start())
    idx = guard + 1 if guard != -1 else raise_re.start() + 1
    return src[:idx] + patch + src[idx:]


def main():
    ap = argparse.ArgumentParser(description="Register a trimmed BPE model's pre-tokenizer with llama.cpp convert.")
    ap.add_argument("--trimmed", required=True, help="path to the trimmed model dir (output of trim_vocab.py)")
    ap.add_argument("--base", help="base model id/dir the trim came from (used to auto-detect the pre-tokenizer id)")
    ap.add_argument("--res", help="pre-tokenizer id to register (skips --base auto-detection). "
                                  "For gemma / EmbeddingGemma trims use 'granite-embed-multi-311m'.")
    ap.add_argument("--llama-cpp", required=True, help="path to your llama.cpp checkout")
    ap.add_argument("--apply", action="store_true", help="insert the patch into the convert source (otherwise just print it)")
    args = ap.parse_args()
    if not args.base and not args.res:
        raise SystemExit("pass --base <model> (auto-detect) or --res <pre-tokenizer-id> (explicit).")

    src_path = _find_convert_source(args.llama_cpp)
    src = open(src_path, encoding="utf-8").read()
    chktxt = _extract_chktxt(src)

    trimmed_h = _chkhsh(args.trimmed, chktxt)
    print(f"convert source : {src_path}")
    print(f"trimmed chkhsh : {trimmed_h}")

    if _res_for_hash(src, trimmed_h):
        print(f"\n✓ already registered (res = {_res_for_hash(src, trimmed_h)!r}) — nothing to do.")
        return

    if args.res:
        base_res = args.res
    else:
        base_h = _chkhsh(args.base, chktxt)
        print(f"base    chkhsh : {base_h}")
        base_res = _res_for_hash(src, base_h)
        if base_res is None:
            hint = ", ".join(_known_res(src)[:12])
            raise SystemExit(
                f"\nERROR: the base model's chkhsh ({base_h}) is not registered in {src_path}\n"
                f"(this happens when the base — e.g. EmbeddingGemma — is not itself a known model, even though it\n"
                f"shares a pre-tokenizer with one). Re-run with an explicit --res. For gemma/EmbeddingGemma use\n"
                f"  --res granite-embed-multi-311m\n"
                f"Registered pre-tokenizer ids include: {hint} ...")
    print(f"pre-tokenizer  : {base_res!r}")

    patch = (f'        if chkhsh == "{trimmed_h}":\n'
             f'            res = "{base_res}"  # vocab-trimmed {args.base or "model"}: same pre-tokenizer, '
             f'trimmed vocab only changes the hash\n')

    if not args.apply:
        print("\n--- add this to get_vocab_base_pre() (before the NotImplementedError) ---\n")
        print(patch)
        print("re-run with --apply to insert it automatically.")
        return

    new_src = _insert(src, patch)
    if new_src is None:
        raise SystemExit("could not find the NotImplementedError insertion point; add the patch above manually.")
    open(src_path, "w", encoding="utf-8").write(new_src)
    print(f"\n✓ patched {src_path} — the trimmed model will now convert via the faithful BPE path.")


if __name__ == "__main__":
    main()
