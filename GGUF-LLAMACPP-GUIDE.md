# Tornar modelos trimados compatíveis com GGUF/llama.cpp — guia + como corrigir o colibri

> **Para 3 casos:** a **Parte 1** é para a sessão do `embedding-vocab-trimmer` (melhorar a ferramenta);
> a **Parte 2** é para a sessão do modelo `colibri` (corrigir o modelo publicado); a **Parte 3** é para
> **modelos LLM de GERAÇÃO** (gemma) trimados — onde a tokenização precisa ser **bit-exata** (greedy amplifica
> qualquer diferença) e o fix é **diferente/melhor** que o das Partes 1-2. Podem ser lidas juntas.
>
> **Contexto (o que descobrimos, validado):** um modelo trimado (embeddinggemma → colibri) **não** convertia
> pra GGUF corretamente. Depois de 2 correções, **funciona**: GGUF **Q8_0 = cosseno 0.991 vs HF** (near-lossless),
> **~188 MB RAM** servido por `llama-server`, rerank/retrieval iguais ao HF. As 2 correções:
> 1. Flag **`--sentence-transformers-dense-modules`** no convert (inclui as 2 camadas Dense do pipeline ST).
> 2. **Reconstruir o `tokenizer.model` (SentencePiece)** — o trim o deleta, e o llama.cpp precisa dele.
>
> ✅ **RECOMENDAÇÃO ATUAL (validada 2026-07-21): use o caminho BPE — vale pros dois casos.** A reconstrução da SPM
> (correção #2) funciona pro **embedding** mas é **insuficiente pro LLM de geração** (a SPM unigram reconstruída não é
> bit-exata → na geração greedy vira −0.33 de qualidade). O caminho **BPE** do `tokenizer.json` **é bit-exato** e
> **serve pros dois**: no embedding empata com a SPM (cosseno **0.9998 = 0.9998**), no LLM é **obrigatório**. O único
> passo extra é registrar o `chkhsh` do pre-tokenizer no convert — automatizado pelo **`gguf_bpe_patch.py`** (neste
> diretório). Fluxo completo na **§0** abaixo; detalhes e evidências nas Partes 3 (LLM) e 1.5 (embedding legado).

---

## §0 — Fluxo recomendado (caminho BPE, bit-exato — embedding E LLM)

O `trim_vocab.py` **já** produz um modelo BPE-ready (re-indexa o `tokenizer.json` e **remove** o `tokenizer.model`).
Pra virar um GGUF **fiel**, só falta registrar o `chkhsh` do pre-tokenizer no convert — o `gguf_bpe_patch.py` faz isso.

```bash
# 1) trimar (embedding-vocab-trimmer)
python trim_vocab.py --model google/embeddinggemma-300m --corpus-config por --vocab-size 64000 --output ./out

# 2) registrar o pre-tokenizer do modelo trimado no convert do llama.cpp (1x por modelo-base)
#    EmbeddingGemma/gemma NÃO são "modelos conhecidos" do convert, então passe o res explícito:
python gguf_bpe_patch.py --trimmed ./out --res granite-embed-multi-311m --llama-cpp ~/llama.cpp --apply
#    (para um base que o convert JÁ reconhece, dá pra auto-detectar com --base <model> no lugar de --res)

# 3) converter + quantizar
#    - modelo de EMBEDDING (SentenceTransformers): precisa da flag das camadas Dense
python ~/llama.cpp/convert_hf_to_gguf.py ./out --outfile m-f16.gguf --outtype f16 --sentence-transformers-dense-modules
#    - modelo LLM de geração: SEM a flag ST
# python ~/llama.cpp/convert_hf_to_gguf.py ./out --outfile m-f16.gguf --outtype f16
~/llama.cpp/build/bin/llama-quantize m-f16.gguf m-Q8_0.gguf Q8_0
```

**Por que BPE e não SPM (reconstruída):** o BPE tokeniza **idêntico** ao original (bit-exato). A SPM unigram
reconstruída (`rebuild_spm.py`, Parte 1.2) é só *quase* fiel — ok pro embedding (query curta + mean-pooling), **fatal**
pro LLM (prompt longo + greedy). Validação (Q8, VM Verda):

| caso | métrica | SPM reconstruída | **BPE (gguf_bpe_patch)** |
|---|---|---|---|
| embedding (colibri/retriever) | cosseno vs HF | 0.9998 | **0.9998** (empata) |
| LLM (gemma-3-1b-caa) | juiz de prancha | 1.58 (−0.33) | **1.92 (= original, lossless)** |
| LLM — tokenização do prompt | tokens vs original | 602 ≠ 584 | **584 = 584** |

> **Gemma-3 LLM, cuidado extra:** alguns checkpoints têm um `<image_soft_token>` órfão em id == vocab_size que estoura
> `assert max(tokenizer.vocab.values()) < vocab_size` no `get_vocab_base()` do convert. O loop já limita a
> `vocab_size`, então troque esse `assert` por `pass`. (Modelos de embedding trimados por `trim_vocab.py` não têm isso.)

As Partes 1–2 abaixo descrevem o caminho **SPM legado** (ainda válido pro embedding com llama.cpp de estoque, sem patch);
a Parte 3 traz a evidência completa do porquê o LLM exige BPE.

---

## PARTE 1 — Para a sessão do `embedding-vocab-trimmer`

### 1.1 O problema na ferramenta
`trim_vocab.py` (linha ~151) **deleta o `tokenizer.model`** como "stale artifact":
```python
for fn in ("tokenizer.model", "added_tokens.json"):  # stale artifacts of the old vocab
    ...remove...
```
Faz sentido (o SPM antigo tem 262k peças, o vocab trimado tem 64k). **Mas** o llama.cpp, para modelos gemma,
**precisa do `tokenizer.model` SPM** — sem ele o convert cai no path BPE, o pre-tokenizer não é reconhecido, usa
fallback `gpt-2` e **tokeniza diferente** (ex.: 11 tokens vs 10 do HF) → o embedding do GGUF fica **errado**
(cosseno ~0.36 vs HF). Ou seja: **todo modelo que a ferramenta produz hoje é incompatível com GGUF/llama.cpp.**

### 1.2 A melhoria: EMITIR um `tokenizer.model` SPM trimado (em vez de deletar)
Como é BPE e o trim só **remove peças + re-indexa**, dá pra reconstruir o SPM filtrando o proto do modelo-base
(que a ferramenta **já baixa** em `src`) para as peças que sobreviveram, re-indexadas. As peças mantêm score/type
originais → a tokenização SPM reproduz a do fast tokenizer. Detalhes:
- Casar cada token do vocab trimado (por string) com a peça do SPM-base → 0 peças "default" (todas batem).
- **`byte_fallback`**: exige as 256 peças `<0xXX>`. O trim geralmente remove bytes raros (ex.: só 29/256 sobram) →
  **desligar `byte_fallback`** e **reclassificar as peças-byte restantes de `BYTE` → `USER_DEFINED`** (senão o
  SentencePiece recusa carregar: "byte piece found although byte_fallback is false").

**Opção A (rápida, pós-processo):** rodar o `rebuild_spm.py` (neste diretório) depois do trim.
```bash
# base_spm.model = tokenizer.model do modelo-base (ex.: google/embeddinggemma-300m)
python rebuild_spm.py base_spm.model <out_trimado>/tokenizer.json <out_trimado>/tokenizer.model <vocab_size>
```

**Opção B (melhor, integrada no `trim_vocab.py`):** em vez de deletar o `tokenizer.model` na linha ~151, chamar a
reconstrução. O `src` já tem o `tokenizer.model` base e você já tem o `old2new`/`kept`. Esboço:
```python
from sentencepiece import sentencepiece_model_pb2 as pb2
def emit_trimmed_spm(base_model_path, new_vocab, vocab_size, out_path):
    mp = pb2.ModelProto(); mp.ParseFromString(open(base_model_path,"rb").read())
    base = {p.piece: p for p in mp.pieces}
    id2tok = {i: t for t, i in new_vocab.items() if i < vocab_size}   # new_vocab do trim
    byte_kept = sum(1 for t in id2tok.values() if t.startswith("<0x") and t.endswith(">") and len(t)==6)
    newmp = pb2.ModelProto(); newmp.CopyFrom(mp)
    if byte_kept < 256: newmp.trainer_spec.byte_fallback = False
    del newmp.pieces[:]
    BYTE=pb2.ModelProto.SentencePiece.BYTE; UD=pb2.ModelProto.SentencePiece.USER_DEFINED
    for i in range(vocab_size):
        tok=id2tok.get(i, f"<unused{i}>"); p=newmp.pieces.add()
        if tok in base:
            p.CopyFrom(base[tok])
            if byte_kept < 256 and p.type==BYTE: p.type=UD
        else:
            p.piece=tok; p.score=0.0; p.type=pb2.ModelProto.SentencePiece.NORMAL
    open(out_path,"wb").write(newmp.SerializeToString())
# ...no trim(), depois de escrever tokenizer.json:
# emit_trimmed_spm(os.path.join(src,"tokenizer.model"), new_vocab, vocab_size, os.path.join(output,"tokenizer.model"))
```
(o `rebuild_spm.py` anexo é exatamente isso, standalone. Ver `distill/rebuild_spm.py` no repo bench-small-llm também.)

### 1.3 Como CRIAR os GGUFs (Q8 / f16 / Q6 / Q5 / Q4) — pra usar depois no colibri e outros
Pré-requisitos: `llama.cpp` recente (com a flag `--sentence-transformers-dense-modules`, PR #16367 e posteriores),
compilar `llama-quantize`, `llama-embedding`, `llama-server`. O **dir de entrada** deve ser o **save do
sentence-transformers completo** (com `modules.json` + `2_Dense/` + `3_Dense/` + `tokenizer.model` SPM emitido +
`tokenizer.json` **sem added_tokens acima do vocab_size** — ver 2.1).
```bash
# 1) f16 (base p/ quantizar) — a flag das Dense é OBRIGATÓRIA
python llama.cpp/convert_hf_to_gguf.py <ST_dir> --outfile model-f16.gguf --outtype f16 \
    --sentence-transformers-dense-modules
# log deve mostrar: "Setting dense layer dense_2 ... (768, 3072)" e "dense_3 ... (3072, 768)"

# 2) quantizações
llama.cpp/build/bin/llama-quantize model-f16.gguf model-Q8_0.gguf  Q8_0
llama.cpp/build/bin/llama-quantize model-f16.gguf model-Q6_K.gguf  Q6_K
llama.cpp/build/bin/llama-quantize model-f16.gguf model-Q5_K_M.gguf Q5_K_M
llama.cpp/build/bin/llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```
> **Nota MRL/Matryoshka:** o llama.cpp reproduz a **dim cheia (768)**; `truncate_dim < 768` (256/128) **não** é
> fiel. Para retrieval/rerank use 768 (irrelevante para nós). Fonte: PR ggml-org/llama.cpp#16367.

### 1.4 TESTES pra garantir que está funcionando (obrigatórios)
**(a) Cosseno HF-vs-GGUF > 0.99** — o teste decisivo (pega tokenização/Dense/pooling errados):
```python
import json, subprocess, numpy as np
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("<ST_dir>", trust_remote_code=True, device="cpu")
for txt in ["task: search result | query: quero comer na escola",
            "title: none | text: cachorro; cão; animal. tags: animal"]:
    out = subprocess.run(["llama.cpp/build/bin/llama-embedding","-m","model-Q8_0.gguf","-p",txt,
                          "-t","4","--pooling","mean","--embd-output-format","json"], capture_output=True, text=True)
    gg = np.array(json.loads(out.stdout)["data"][0]["embedding"], "float32"); gg/=np.linalg.norm(gg)
    hf = m.encode([txt], normalize_embeddings=True)[0]; hf/=np.linalg.norm(hf)
    print(round(float(hf@gg),4), txt[:40])   # esperar >= 0.985
```
**(b) Contagem de tokens deve bater:** `llama-embedding ... --verbose-prompt 2>&1 | grep "number of tokens"` deve
dar o **mesmo nº** que `AutoTokenizer(...)(txt)["input_ids"]`. (Se divergir → o SPM não está sendo usado.)
**(c) Sem warnings graves** no load do llama-embedding (ex.: "missing 256 byte pieces", "byte piece found").

### 1.5 Armadilhas (o que PRECISA saber pra testar/depurar)
| Sintoma | Causa | Fix |
|---|---|---|
| cosseno ~0 (ortogonal) | faltou `--sentence-transformers-dense-modules` (só transformer+pooling) | usar a flag |
| cosseno ~0.36 | Dense certas mas **tokenização errada** (path BPE/`gpt-2`) | emitir o `tokenizer.model` SPM |
| `AssertionError max(vocab) < vocab_size` no convert | added_tokens vestigiais (id ≥ vocab_size) | limpar (ver 2.1) |
| `there are not 256 byte pieces although byte_fallback is true` | trim removeu bytes | `byte_fallback=False` no SPM |
| `byte piece <0xA0> found although byte_fallback is false` | peças-byte com type=BYTE | reclassificar p/ USER_DEFINED |
| embedding "no espaço errado" | `--pooling` errado | **`--pooling mean`** (nunca `cls`); a arch gemma-embedding assume mean |
| resultado ruim mas cosseno ok | faltou o **prompt** | prepender: query `task: search result | query: `, doc `title: none | text: ` |

**Serving (multi-usuário):** `llama-server -m model-Q8_0.gguf --embedding --pooling mean -t 4 -c 2048`
→ endpoint OpenAI-compat `POST /v1/embeddings {"input": [...]}`. Carrega **1×**, RAM (~188 MB Q8) **amortizada
entre todos os usuários**. O cliente prependa os prompts (o server não).

---

## PARTE 2 — Para a sessão do modelo `colibri` (corrigir o modelo publicado)

> ✅ **Feito em 2026-11-09:** `tardellirs/colibri-embed-ptbr` publicado com os added_tokens vestigiais removidos e com
> `add_bos_token`/`add_eos_token` restaurados no `tokenizer_config.json`. O transformers 5.x apagou essas flags ao
> salvar; sem elas o transformers 4.x omite o `<eos>`, e o embedding fica a ~0.87 de cosseno do certo. GGUFs pelo
> caminho BPE (§0) públicos em **`tardellirs/colibri-embed-ptbr-GGUF`**: f16, Q8_0 (cosseno 0.9996, 229 MB de RAM),
> Q6_K, Q5_K_M e Q4_K_M. Ao comparar GGUF × HF, a referência HF tem que ter o `<eos>`: no transformers 4.x, sem a
> flag, o cosseno cai para 0.87 e parece erro do GGUF.

### 2.1 Os 2 problemas do `tardellirs/colibri-embed-ptbr` (e o finetune)
1. **Inconsistência tokenizer × embedding:** o `tokenizer.json` tem **70.246 tokens** (max id 70245), mas a matriz
   de embedding tem só **64.000 linhas**. Os ~6.246 tokens com id 64000–70245 são **added_tokens vestigiais do
   trim** (controle/multimodais do gemma). **Funciona** em pt-BR (esses ids nunca são emitidos — o texto codifica
   tudo < 64000, e bos/eos/pad/unk = 2/1/0/3), mas **quebra conversores** (GGUF, e qualquer coisa que valide
   `max(vocab) < vocab_size`).
2. **Falta o `tokenizer.model` (SentencePiece).** Só há o `tokenizer.json` fast → incompatível com llama.cpp.

### 2.2 Como corrigir o colibri (republicar uma versão limpa)
**(i) Remover os added_tokens vestigiais** (id ≥ vocab_size) do `tokenizer.json` e `tokenizer_config.json`:
```python
import json
d = "<dir_do_modelo>"; VSZ = 64000
tj = json.load(open(d+"/tokenizer.json"))
tj["added_tokens"] = [t for t in tj.get("added_tokens",[]) if t["id"] < VSZ]
if isinstance(tj["model"].get("vocab"), dict):
    tj["model"]["vocab"] = {k:v for k,v in tj["model"]["vocab"].items() if v < VSZ}
json.dump(tj, open(d+"/tokenizer.json","w"), ensure_ascii=False)
tc = json.load(open(d+"/tokenizer_config.json"))
if "added_tokens_decoder" in tc:
    tc["added_tokens_decoder"] = {k:v for k,v in tc["added_tokens_decoder"].items() if int(k) < VSZ}
json.dump(tc, open(d+"/tokenizer_config.json","w"), ensure_ascii=False)
# validar: bos/eos/pad preservados e todo pt-BR codifica < VSZ
```
> Verificado seguro: bos=2, eos=1, pad=0, unk=3 (todos < 64000); "quero comer na escola" → todos ids < 64000.

**(ii) Adicionar o `tokenizer.model` SPM** com o `rebuild_spm.py` (Parte 1.2) — precisa do `tokenizer.model` do
**embeddinggemma-base**:
```bash
huggingface-cli download google/embeddinggemma-300m tokenizer.model --local-dir /tmp/base
python rebuild_spm.py /tmp/base/tokenizer.model <dir_do_modelo>/tokenizer.json <dir_do_modelo>/tokenizer.model 64000
```
**(iii) Idealmente:** fazer o (i)+(ii) **dentro do trimmer** (Parte 1.2) para que o `tardellirs/colibri-embed-ptbr`
(e o finetune `colibri-caa`) já saiam limpos e com SPM — beneficia todo uso futuro (GGUF, ONNX, portabilidade).

### 2.3 Resultado esperado depois de corrigido
- GGUF converte pelo path SPM → **cosseno 0.99 vs HF**, Q8_0 near-lossless, ~188 MB, `llama-server` multi-usuário.
- Sem inconsistência tokenizer/embedding → portável para qualquer runtime.

---

## Artefatos que geramos (validados) — para subir ao HF junto do finetune
Gerados na VM e no volume `c1783695` (Verda). Baixar/anexar:
- `colibri-spm-Q8_0.gguf` (160 MB) — **o recomendado p/ deploy** (cosseno 0.991, 188 MB RAM).
- `colibri-spm-f16.gguf` (299 MB) — base p/ requantizar (Q6/Q5/Q4).
- `tokenizer.model` (SPM reconstruído do vocab 64k) — **o artefato-chave** que faltava.
- `rebuild_spm.py` — neste diretório (`embedding-vocab-trimmer/rebuild_spm.py`) e em `bench-small-llm/distill/`.

**Números medidos (Verda CPU.4V.16G, Zen2, 4 threads, via llama-server):**
| GGUF | tamanho | RAM* | cosseno vs HF | nota |
|---|---|---|---|---|
| f16 | 299 MB | 336 MB | 0.991 | referência |
| **Q8_0** | 160 MB | 188 MB | 0.985–0.993 | **escolhido** (near-lossless, safe) |
| Q6_K | 129 MB | ~216 MB | 0.985–0.991 | = Q8 na qualidade, menor |
| Q5_K_M | 115 MB | ~202 MB | 0.9825–0.9900 | near-lossless |
| Q4_K_M | 105 MB | ~234 MB | 0.9802–0.9870 | menor arquivo; leve queda |

*RAM: o Q8_0 no `llama-server` (serving real, batelado) = **188 MB**; as demais são do `llama-embedding` standalone
(dominadas pelo runtime/contexto, não pelos pesos — por isso não caem monotônico com o quant). No serving todas
ficam ~200 MB. **Rerank/latência (Q8 vs f16): idênticos** (accL 0.948 / FA 0.052, ~28 ms/query, ~320 ms/prancha).
→ **Todos os quants preservam a qualidade** (cosseno > 0.98 → mesmo ranking de retrieval/rerank). **Q8_0** é o
default seguro; **Q4_K_M** (105 MB) se quiser o menor download. Retrieval = colibri-ft (r@5 0.902).

**No HF:** os GGUFs validados estão em **`tardellirs/colibri-caa-scaled-GGUF`** (privado): `colibri-caa-Q8_0.gguf`,
`colibri-caa-Q6_K.gguf`, `colibri-caa-f16.gguf` e o **`tokenizer.model`** (SPM reconstruído).

---

## PARTE 3 — Modelos LLM de GERAÇÃO (gemma) — o caminho FIEL (bit-exato)

> **Isto é DIFERENTE das Partes 1-2 (embedding).** No embedding a tokenização só precisa ser *quase* fiel
> (cosseno 0.99 basta). Um **LLM de geração** decodifica **greedy** (temp 0) e é **autoregressive** → **amplifica
> qualquer diferença de tokenização**: até **1 token** a mais no prompt derruba a qualidade. Medido: gemma-3-1b-caa
> trimado com a receita "embedding" (SPM reconstruído) **cai −0.33 de juiz**; com o tokenizer **bit-exato** volta a
> **lossless** (juiz idêntico ao original) a 1.34× (270m: 1.72×). Validado na VM Verda `trim-juiz`, 2026-07-21.

### 3.1 Descoberta central: o slice do embedding é LOSSLESS; toda a perda vem do TOKENIZER
A/B controlado (gemma-3-1b-caa, Q8, llama-server, `/completion` prompt cru, 12 pranchas, juiz gemini 0-2):

| variante do trim | juiz 1b | o que estava errado |
|---|---|---|
| ingênuo (kept-set retrieval + estrutura de embedding + SPM reconstruído) | 1.583 | tokeniza diferente do original |
| **← alimentado com os token-ids EXATOS do original** (ver 3.4c) | **1.917 = original** | **prova: o corte de vocab é lossless** |
| **fiel (BPE bit-exato)** | **1.917** ✅ | tokeniza idêntico (584 = 584) |

O **teste de isolamento** é a prova: pegue os ids que o tokenizer ORIGINAL produz, remapeie pros ids do trim (mesmas
strings) e mande via `/completion` como **array de ids** (não texto). A saída fica **byte-idêntica** ao original → os
logits dos tokens mantidos são inalterados (o argmax greedy é preservado). Logo **a única fonte de perda é o tokenizer
do trim segmentar diferente** — e num modelo pequeno greedy isso vira efeito borboleta.

### 3.2 As 3 condições pra tokenização bit-exata (todas obrigatórias pro LLM)
1. **Kept-set cobre o domínio do LLM.** O `select_kept_ids` do `trim_vocab.py` já monta por frequência de corpus —
   **use o corpus do LLM** (system prompt + pedidos + saídas), não um de embedding. Se faltar um token que o prompt usa,
   ele fragmenta. (No nosso caso faltavam 3 do system prompt: `▁CAA`, `▁pictogram`, `▁"|"`.) Inclua o **fecho de
   merge-ancestrais** dos tokens do domínio (as peças intermediárias que os merges do gemma usam pra construí-los).
2. **Herde a ESTRUTURA do tokenizer do modelo-base (gemma), NUNCA de um modelo de embedding.** O `trim_vocab.py` já faz
   certo (`tj2 = copy(tj_base)`, só troca vocab/merges/added_tokens). ⚠️ Um tokenizer de **embedding** (colibri/
   embeddinggemma) tem `post_processor` que **anexa `<eos>`** ao fim (562 vs 561 tokens) — isso sozinho já custa ~0.17
   de juiz. Confirme que `post_processor`/`normalizer`/`pre_tokenizer`/`decoder` são os do gemma-base.
3. **No GGUF, use o caminho BPE do `tokenizer.json` — NÃO a SPM reconstruída.** Diferença **crítica** vs a Parte 1: pro
   embedding, reconstruir a SPM dá 0.99 (ok). Pro LLM, a **SPM unigram reconstruída é inerentemente infiel** — a
   segmentação por score muda ao remover peças: medimos **602 tokens vs 584** do original no mesmo prompt → −0.33. O
   **BPE** do `tokenizer.json` é **exato** (584 = 584).

### 3.3 Habilitar o caminho BPE fiel no convert (2 patches de 1 linha em `llama.cpp/conversion/base.py`)
Sem a SPM, o convert do gemma cai no BPE e valida o **pre-tokenizer** por um hash (`chkhsh`) que depende do vocab. Como
o trim tem vocab custom, o hash **não é reconhecido** → `NotImplementedError` (ou fallback `gpt-2` silencioso e errado):
```python
# (a) token órfão <image_soft_token> (id == vocab_size) estoura a assertion do gemma.
#     Em get_vocab_base(), o loop já é range(vocab_size) (ignora o órfão), então basta soltar a assertion:
#         trocar:  assert max(tokenizer.vocab.values()) < vocab_size
#         por:     pass   # órfãos id>=vocab_size ignorados; o loop abaixo limita a vocab_size

# (b) registrar o chkhsh do pre-tokenizer do trim como o MESMO 'res' do gemma-base.
#     O pre-tokenizer É idêntico ao gemma (regex GPT2); só o hash muda porque hasheia a tokenização, que depende do
#     vocab. Em get_vocab_base_pre(), ANTES do NotImplementedError:
if chkhsh == "<hash-do-seu-trim>":      # aparece no WARNING "chkhsh: ..." quando o convert falha
    res = "granite-embed-multi-311m"    # o MESMO 'res' que o gemma-3 original casa (só um rótulo p/ o mesmo regex)
```
> Como achar o `res` certo: compute o chkhsh do modelo-ORIGINAL (não-trimado) do mesmo jeito e veja em que branch
> `if chkhsh == "..."` ele cai no `base.py` — use o `res` dessa branch. (O gemma-3 casa em `granite-embed-multi-311m`,
> que é só um rótulo compartilhado do pre-tokenizer GPT2 default.)

Pipeline completo (LLM):
```bash
# 1) trimar com o corpus do LLM -> <trim_dir>  (kept-set + estrutura gemma: condições 3.2.1 e 3.2.2)
# 2) REMOVER a SPM pra forçar o path BPE fiel:
rm -f <trim_dir>/tokenizer.model
# 3) convert BPE (com os 2 patches) + quantizar   (LLM NÃO usa --sentence-transformers-dense-modules)
python llama.cpp/convert_hf_to_gguf.py <trim_dir> --outfile m-f16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize m-f16.gguf m-Q8_0.gguf Q8_0
```

### 3.4 TESTES obrigatórios (LLM)
- **(a) Contagem de tokens GGUF == original.** Via `POST /tokenize` do `llama-server` nos DOIS GGUFs (orig e trim) no
  MESMO prompt longo (system+pedido). Tem que bater **exato** (ex.: 584 = 584). Se o trim der mais → a SPM ainda está
  sendo usada, ou o kept-set/estrutura estão errados.
- **(b) Juiz A/B no mesmo harness** (`llama-server` Q8, `/completion` prompt gemma cru, N pranchas): trim ≈ original.
  Um −0.1..−0.3 denuncia tokenização infiel.
- **(c) Isolamento (diagnóstico decisivo):** se o juiz cair, gere no trim alimentando os **ids exatos do original**
  (tokeniza no orig, remapeia strings→ids do trim, manda `/completion` com `"prompt": [ids...]`). Se voltar ao
  original, o problema é 100% tokenizer, não o slice.

> **Cuidado com o `/completion`:** o `--chat-template gemma` built-in do llama.cpp **dropa o system prompt** do
> gemma (que não tem role system). Monte o prompt cru você mesmo: `<start_of_turn>user\n{SYS}\n\n{PEDIDO}<end_of_turn>\n<start_of_turn>model\n` e mande via `/completion` (não `/v1/chat/completions`).

### 3.5 Relação com as Partes 1-2 (embedding) — mesmo problema, fix melhor
| | Embedding (Partes 1-2) | LLM geração (Parte 3) |
|---|---|---|
| Tolerância a tokenização | cosseno 0.99 basta | **bit-exato** (greedy amplifica) |
| Fix do GGUF documentado | reconstruir SPM (`rebuild_spm.py`) | **BPE + patch do chkhsh do pre-tokenizer** |
| SPM reconstruída | funciona (~0.99) | **falha (−0.33)** — unigram infiel |
| Estrutura do tokenizer | (herdada do base — ok) | idem, mas **cuidado com `<eos>`** se vier de embedding |
| Precisa patchar o convert? | não (path SPM) | sim (2 linhas de 3.3) |
| Flag `--sentence-transformers-dense-modules` | **sim** | **não** (não é pipeline ST) |

**Hipótese TESTADA no colibri (2026-07-21) — REFUTADA (de forma útil).** Conjecturamos que o BPE-com-patch seria
*superior* também pro embedding. **É igual, não superior.** No `aac-pictogram-retriever-ptbr` (embeddinggemma trim 64k),
cosseno vs HF (Q8, `--sentence-transformers-dense-modules`, `--pooling mean`):

| inputs | **BPE (patch do chkhsh)** | **SPM (reconstruído)** |
|---|---|---|
| 3 queries curtas (13–18 tok) | **0.9998** | **0.9998** |
| 4 strings longas c/ domínio (28–49 tok; "CAA", "pictograma") | **0.9998** | **0.9998** |

Nenhuma divergência. **Por quê:** inputs de embedding são curtos (~≤50 tok) e o **mean-pooling é robusto** a pequenas
diferenças de segmentação; a infidelidade do SPM unigram só machuca em **prompt longo + geração greedy** (o LLM: 602 vs
584 → −0.33). Notas: (a) o BPE-com-patch **funciona** pro embedding (0.9998) — o "cosseno 0.36" da Parte 1.5 era o
fallback `gpt-2` **sem** o patch do chkhsh; com o patch, BPE é uma alternativa **equivalente** ao SPM. (b) **Conclusão
prática:** manter a Parte 1.2 (SPM) pro embedding (funciona, já validado) **ou** unificar no BPE-com-patch (serve pros
dois; mesma qualidade no embedding) — **mas não há ganho de qualidade** no embedding em unificar. O BPE só é
**obrigatório** pro LLM.

### 3.6 Números (VM Verda `trim-juiz` CPU.4V.16G, Q8, llama-server, 12 pranchas)
| modelo | orig | trim ingênuo | **trim fiel (BPE)** | velocidade (llama-bench tg128) |
|---|---|---|---|---|
| gemma-3-1b-caa | 1.917 | 1.583 | **1.917** (lossless) | 25.8 → 34.7 tok/s (**1.34×**) |
| gemma-3-270m-caa | 1.000 | 0.667 | **1.083** (lossless) | 97.5 → 167.9 tok/s (**1.72×**) |

Tokenização GGUF do prompt (system+pedido): orig **584** = trim fiel **584** (a SPM reconstruída dava 602). SAMPLEs do
trim fiel **byte-idênticos** aos do original. Artefatos/scripts: `bench-small-llm/distill/trim2.py` (trim fiel),
`juiz2.py` (`/completion` prompt cru), e `bench-small-llm/llm/RESULTADOS-VOCAB-TRIM.md`.
