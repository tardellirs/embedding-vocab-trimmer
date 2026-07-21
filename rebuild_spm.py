#!/usr/bin/env python3
"""Reconstrói o tokenizer.model (SentencePiece) para um modelo com vocab trimado (colibri), filtrando o
SPM proto do modelo-base (embeddinggemma) para as peças que sobreviveram ao trim, re-indexadas.
Como é BPE e o trim só remove peças + re-indexa, manter as peças-base (com score/type originais) na ordem
nova reproduz a tokenização. Peças não achadas na base recebem default.

Uso: python3 rebuild_spm.py <base_tokenizer.model> <colibri/tokenizer.json> <out/tokenizer.model> <vocab_size>
"""
import json, sys
from sentencepiece import sentencepiece_model_pb2 as pb2


def main():
    base_model, colibri_tok, out_model, vsz = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    mp = pb2.ModelProto()
    mp.ParseFromString(open(base_model, "rb").read())
    base = {p.piece: p for p in mp.pieces}
    print(f"base SPM pieces: {len(mp.pieces)}")

    tj = json.load(open(colibri_tok))
    id2tok = {}
    for tok, i in tj["model"]["vocab"].items():
        if i < vsz:
            id2tok[i] = tok
    for at in tj.get("added_tokens", []):
        if at["id"] < vsz:
            id2tok[at["id"]] = at["content"]
    faltando_ids = [i for i in range(vsz) if i not in id2tok]
    print(f"colibri ids cobertos: {len(id2tok)}/{vsz} | buracos: {len(faltando_ids)}")

    # quantas das 256 peças de byte <0xXX> sobreviveram ao trim?
    byte_kept = sum(1 for t in id2tok.values() if t.startswith("<0x") and t.endswith(">") and len(t) == 6)
    print(f"peças de byte <0xXX> no colibri: {byte_kept}/256")

    newmp = pb2.ModelProto()
    newmp.CopyFrom(mp)
    if byte_kept < 256:  # byte_fallback exige 256 bytes; se faltam, desliga (o trim removeu bytes raros)
        newmp.trainer_spec.byte_fallback = False
        print("byte_fallback -> False (colibri nao tem 256 bytes)")
    del newmp.pieces[:]
    miss = 0
    BYTE = pb2.ModelProto.SentencePiece.BYTE
    USERDEF = pb2.ModelProto.SentencePiece.USER_DEFINED
    reclass = 0
    for i in range(vsz):
        tok = id2tok.get(i, f"<unused{i}>")
        p = newmp.pieces.add()
        if tok in base:
            p.CopyFrom(base[tok])
            if byte_kept < 256 and p.type == BYTE:  # byte_fallback off -> peças-byte viram literais
                p.type = USERDEF
                reclass += 1
        else:
            p.piece = tok
            p.score = 0.0
            p.type = pb2.ModelProto.SentencePiece.NORMAL
            miss += 1
    if reclass:
        print(f"peças BYTE reclassificadas p/ USER_DEFINED: {reclass}")
    open(out_model, "wb").write(newmp.SerializeToString())
    print(f"peças fora da base (default): {miss} | escrito {out_model} ({len(newmp.pieces)} peças)")


if __name__ == "__main__":
    main()
