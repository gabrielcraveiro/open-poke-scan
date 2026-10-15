"""open-poke-scan recognition server.

POST /recognize: photo of one card -> the matching catalog card.

Pipeline: DINOv2-small embedding -> cosine similarity against the index, minus
the hub penalty of each card -> pHash tie-break between prints with the same
art -> OCR (RapidOCR) of the card name and then of the collector number, only
when the match is ambiguous.

    uvicorn main:app --host 0.0.0.0 --port 8000

Environment:
    MODELS_DIR     folder with model.onnx, emb.f16.bin, meta.json and the
                   optional hub.json (default: ./models)
    ALLOW_ORIGINS  comma-separated CORS origins (default: *)
    DEBUG_DIR      when set, saves each uploaded photo there for 6 hours
"""
import asyncio
import io
import json
import os
import re
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

import cardtext
from cardvision import EMB_DIM, embed, phash

MODELS = os.environ.get("MODELS_DIR") or os.path.join(os.path.dirname(__file__), "models")
emb_sess = ort.InferenceSession(os.path.join(MODELS, "model.onnx"), providers=["CPUExecutionProvider"])

MAT = np.fromfile(os.path.join(MODELS, "emb.f16.bin"), dtype=np.float16).astype(np.float32).reshape(-1, EMB_DIM)
MAT /= np.linalg.norm(MAT, axis=1, keepdims=True) + 1e-9
with open(os.path.join(MODELS, "meta.json")) as _f:
    META = json.load(_f)["cards"]

# TCG Pocket (A1…A4a, B1…B2, P-A) só existe no jogo digital: ninguém escaneia
# essas cartas em papel, e elas eram o 1º lugar de fotos de outras cartas.
_POCKET_SET_RE = re.compile(r"^(A\d|B\d|P-[AB])")
_keep = [i for i, c in enumerate(META) if not _POCKET_SET_RE.match(c.get("set_id") or "")]
MAT = MAT[_keep]
META = [META[i] for i in _keep]

PHASH = []
for _c in META:
    _p = _c.get("phash")
    try:
        PHASH.append(int(_p, 16) if _p and len(_p) == 32 else None)
    except (TypeError, ValueError):
        PHASH.append(None)

# Penalidade de "hub" por carta (models/hub.json, de scripts/build_hub_penalty.py
# com fotos reais). Algumas cartas ficam perto de MUITAS fotos (holo lavado,
# pouco contraste) e ganham o 1º lugar de fotos de outras cartas. A ordem usa
# cos - _HUB_ALPHA * penalidade; os limites absolutos (não é carta, confiança)
# continuam no cosseno puro. No servidor publicado (121 scans com gabarito):
# 1º lugar certo de 71% para 79%, sem estragar nenhum. Sem o arquivo, não há
# penalidade.
_HUB_ALPHA = 0.25
# Preferência de idioma: desconto fixo no score das coleções japonesas (set_id
# "-jp", seed da Liga). Muitas japonesas têm a MESMA arte da versão em inglês,
# e aí a japonesa ganhava o 1º lugar por um fio (Bubbly Energy me04-084 →
# m6-jp-106). Carta japonesa sem versão em inglês ganha por muito (Altaria
# M6-087: 0,87 contra 0,70) e não é afetada. Eval de 2026-10-01: 93 → 94 de 121,
# Altaria japonesa certa nas 3 fotos com qualquer desconto entre 0,02 e 0,08.
_JP_PRIOR = 0.03
try:
    with open(os.path.join(MODELS, "hub.json")) as _f:
        _hub = json.load(_f)
    HUB = np.array([_hub["cards"].get(c["api_id"], _hub["median"]) for c in META], dtype=np.float32)
except (OSError, ValueError, KeyError):
    HUB = np.zeros(len(META), dtype=np.float32)
JP = np.array([str(c.get("set_id") or "").endswith("-jp") for c in META], dtype=np.float32)

# Nomes do catálogo para casar o nome lido no topo da carta (cardtext). As
# coleções "-jp" ficam de fora: no servidor publicado elas vêm da Liga com o
# nome em português, e uma carta em português casava com a japonesa pelo nome.
NAMES = cardtext.NameIndex([c["name"] for c in META], skip=JP > 0)

# ── OCR do número ────────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"(\d{1,4})\s*/\s*(\d{1,3})(?!\d)")


def _digits(s) -> str:
    return re.sub(r"\D", "", str(s or "")).lstrip("0") or "0"


def _find_num(text):
    """First plausible NNN/NNN in the OCR text, or None."""
    for m in _NUM_RE.finditer(text):
        left, right = m.group(1), m.group(2)
        # Total de 1 dígito ou < 10 nunca é printed_total ("2022/0").
        if len(right) < 2 or int(right) < 10:
            continue
        if len(left) == 4:
            # 1990–2035 é o ano do copyright ("©2024/…"). Senão é o número com
            # um dígito espúrio colado pelo OCR ("3072/084" → 072/084).
            if 1990 <= int(left) <= 2035:
                continue
            left = left[1:]
        return left, right
    return None


_NUM_ONLY_RE = re.compile(r"(?<!\d)(\d{3})\s*/")


def _find_num_only(text):
    """Number of 3 digits with an unreadable total ("055/33718"), or None.

    Use it only as a fallback of _find_num. decide() uses a number without a
    total only to choose between reprints with the name of the first candidate.
    """
    m = _NUM_ONLY_RE.search(text)
    return (m.group(1), None) if m else None


# O OCR (RapidOCR, cardtext.py) não tem prazo que mata o processo, como o
# Tesseract tinha: o custo é limitado pelo tamanho da entrada (640px de
# largura, no máximo 6 caixas de texto), ~100ms o nome e ~200ms o número.
# Limiares calibrados com 160 scans reais rotulados (d = (1−cos)×100):
# - Nenhuma carta certa passou de d=39; quadros sem carta (jeans, tela preta,
#   teclado) ficaram em d=40-89. Abaixo deste cosseno não é carta.
_NOT_CARD_COS = 0.55
# - "confident" com cos>=0.55 errava 15% dos aceites. cos>=0.72 E folga>=0.03
#   sobre o 2º deu 80/81 certos. Folga pequena = mesma arte em outro print.
_CONF_COS = 0.72
_CONF_GAP = 0.03
# - O total impresso separa sets com a mesma arte: desempata candidatos quase
#   empatados e, quando só UM deles tem esse total, vale como confiança
#   (reimpressões como Duraludon sv07-106 /142 e sv08.5-069 /131).
_TOTAL_FUSION_TOPN = 5
_TOTAL_FUSION_GAP = 0.03
# - Nome lido no topo da carta (servidor publicado, 129 cartas reais
#   rotuladas): 1º lugar certo de 106 para 111 e confiantes de 69 para 74,
#   todas certas, sem piorar nenhum scan. O nome só troca a carta escolhida
#   quando a nova tem cosseno >= este valor: abaixo dele, o nome lido era de
#   uma carta vizinha no recorte, ou o nome certo levava a outro print com
#   arte diferente.
_NAME_MOVE_COS = 0.55
# - Nome confere e o cosseno passa disto, com folga >= _CONF_GAP só sobre as
#   cartas de MESMO nome: confiante. Com 0,65 entrava outro print do mesmo
#   Pokémon com arte diferente.
_NAME_CONF_COS = 0.70
# Gap top1−top2 abaixo disso = empate de arte, e o pHash decide.
_PHASH_TIE_GAP = 0.02

# Uma inferência por vez. Com 1 vCPU, duas em paralelo levam o dobro cada e
# a fila colapsa sob carga.
_INFER_SEM = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENCY", "1")))


def _name_key(i: int) -> str:
    # "Buzzwole-GX" e "Buzzwole GX" são a mesma carta com grafias diferentes.
    return re.sub(r"[^a-z0-9]", "", str(META[i]["name"]).lower())


def _guided_number(text, topk):
    """Candidate whose collector number is in the OCR text, or None.

    OCR can read the stylized slash of the modern cards as "7"
    ("035/131" → "03577370"), so _find_num does not see a number. Here each
    candidate's own number, padded to 3 digits, must be in the text followed
    by "/" or "7". Only cards with a 3-digit printed number take part (the
    padded "035" is specific; "12" of an old card is not). All matches must
    have the same name, else the result is None.
    """
    hits = []
    for i in topk:
        n = str(META[i]["number"])
        if n.isdigit() and len(n) <= 3 and re.search(rf"(?<!\d){n.zfill(3)}[/7]", text):
            hits.append(i)
    if hits and len({_name_key(i) for i in hits}) == 1:
        return hits[0]
    return None


def decide(sims, score, topk, pick, num=None, total=None, text="", name_rows=None):
    """Apply the name and number fusion and the confidence gate to one ranking.

    Args:
        sims: cosine similarity of the photo to every catalog row.
        score: ranking score per row (cosine minus the hub penalty). The
            margins between candidates use it; the absolute gates (not a card,
            confident) use the raw cosine.
        topk: row indices of the best matches, best first (ordered by `score`).
        pick: row that the ranking (plus the pHash tie-break) chose.
        num, total: collector number read by OCR, or None. total is None when
            OCR read only the number: then num picks only among reprints that
            have the name of the first candidate.
        text: all OCR text, for _guided_number when num gives no match.
        name_rows: catalog rows with the name that OCR read on the card, best
            `score` first (NameIndex.rows), or None when no name was read.

    Returns:
        dict with keys pick, name_match, number_match, total_match, confident,
        not_card, and order: the candidate rows after the name fusion, best
        first.
    """
    name_match = False
    if name_rows:
        if pick in name_rows:
            name_match = True
        elif float(sims[name_rows[0]]) >= _NAME_MOVE_COS:
            pick, name_match = name_rows[0], True
        if name_match:
            # As cartas do nome lido vão para a frente: a fusão do número
            # escolhe entre elas.
            first = name_rows[:len(topk)]
            topk = first + [i for i in topk if i not in set(first)]
    number_match = total_match = False
    if num:
        dn = _digits(num)
        pool = [i for i in topk if _digits(META[i]["number"]) == dn]
        with_total = [i for i in pool
                      if total and str(META[i].get("printed_total")) == str(int(_digits(total)))]
        if with_total:
            pick, number_match = with_total[0], True
        elif not total:
            # Número sem total: leitura fraca, então só escolhe entre
            # reimpressões com o mesmo nome do 1º colocado.
            same = [i for i in pool[:_TOTAL_FUSION_TOPN] if _name_key(i) == _name_key(topk[0])]
            if same:
                pick, number_match = same[0], True
        # Número bate mas o total contradiz: a carta provavelmente não está no
        # catálogo. Não força; fica o embedding.
    if total and not number_match:
        dt = str(int(_digits(total)))
        near = [i for i in topk[:_TOTAL_FUSION_TOPN]
                if str(META[i].get("printed_total")) == dt
                and float(score[topk[0]] - score[i]) < _TOTAL_FUSION_GAP]
        if near and near[0] != pick:
            pick, total_match = near[0], True
        total_ok = len(near) == 1 and near[0] == pick
    else:
        total_ok = False
    if text and not number_match:
        g = _guided_number(text, topk)
        if g is not None:
            pick, number_match = g, True
    top_cos = float(max(sims[i] for i in topk))
    others = [float(score[i]) for i in topk if i != pick]
    gap = float(score[pick]) - (max(others) if others else 0.0)
    confident = number_match or (float(sims[pick]) >= _CONF_COS and (gap >= _CONF_GAP or total_ok))
    if name_match and not confident and pick in name_rows:
        # A folga conta só contra as cartas de mesmo nome: um Pokémon
        # diferente com arte parecida não tira mais a confiança.
        same = [float(score[i]) for i in name_rows if i != pick]
        name_gap = float(score[pick]) - (max(same) if same else 0.0)
        confident = float(sims[pick]) >= _NAME_CONF_COS and name_gap >= _CONF_GAP
    return {"pick": pick, "name_match": name_match, "number_match": number_match,
            "total_match": total_match, "confident": confident, "not_card": top_cos < _NOT_CARD_COS,
            "order": topk}


DEBUG_DIR = os.environ.get("DEBUG_DIR") or ""
if DEBUG_DIR:
    os.makedirs(DEBUG_DIR, exist_ok=True)
_recognize_count = 0


def _sweep_debug_dir(max_age_s: float = 6 * 3600) -> None:
    now = time.time()
    for name in os.listdir(DEBUG_DIR):
        path = os.path.join(DEBUG_DIR, name)
        try:
            if now - os.path.getmtime(path) > max_age_s:
                os.remove(path)
        except OSError:
            pass


app = FastAPI()
_origins = [o.strip() for o in (os.environ.get("ALLOW_ORIGINS") or "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health():
    """Liveness check. The client calls it to wake a suspended machine."""
    return {"ok": True, "cards": len(META)}


def _card(i: int, sims: np.ndarray) -> dict:
    c = META[i]
    return {"api_id": c["api_id"], "name": c["name"], "number": c["number"],
            "set_id": c["set_id"], "set_name": c["set_name"],
            "printed_total": c.get("printed_total"), "image_url": c["image_url"],
            "cos": round(float(sims[i]), 3)}


def recognize_image(img: Image.Image, footer: Image.Image = None):
    """Identify the card in one photo: all the CPU work of POST /recognize.

    The endpoint and scripts/eval_engine.py call this same function, so the
    eval measures the code that the server runs.

    Args:
        img: the card, cropped by the client. This server always treats the
            whole image as the card.
        footer: optional bottom 28% of the card at full camera resolution, for
            the number OCR. None = read the number in the bottom of `img`.

    Returns:
        (body, log): the JSON body of POST /recognize and a one-line summary
        for the server log.

    Example:
        >>> body, _ = recognize_image(Image.open("card.jpg"))
        >>> body["card"]["api_id"], body["confident"]
        ('sv07-158', True)
    """
    t0 = time.time()
    img = img.convert("RGB")
    vec = embed(emb_sess, img)
    t_emb = time.time()
    sims = MAT @ vec
    score = sims - _HUB_ALPHA * HUB - _JP_PRIOR * JP
    topk = [int(i) for i in np.argsort(-score)[:25]]
    pick = topk[0]
    gap = float(score[topk[0]] - score[topk[1]]) if len(topk) > 1 else 1.0
    if gap < _PHASH_TIE_GAP:
        tied = [i for i in topk if float(score[topk[0]] - score[i]) < _PHASH_TIE_GAP and PHASH[i] is not None]
        if len(tied) > 1:
            qh = phash(img)
            pick = min(tied, key=lambda i: (qh ^ PHASH[i]).bit_count())

    # O OCR só roda quando pode mudar a resposta. Pula quando o embedding já
    # daria a resposta como confiante E os dois primeiros não têm o mesmo
    # nome. Mesmo nome = mesma arte em outro set: aí só o número desempata.
    # Primeiro o nome (~100ms); o número (~200ms) só se o nome não deu
    # confiança.
    num = total = None
    name_texts, name_rows = [], None
    top_cos = float(sims[pick])
    if top_cos < _NOT_CARD_COS:
        raw = "(ocr skipped: not a card)"
    elif (top_cos >= _CONF_COS and gap >= _CONF_GAP
          and (len(topk) < 2 or _name_key(topk[0]) != _name_key(topk[1]))):
        raw = "(ocr skipped: decisive)"
    else:
        name_texts = cardtext.read_name_texts(img)
        bases = NAMES.match(name_texts)
        name_rows = NAMES.rows(bases, score) if bases else None
        if decide(sims, score, topk, pick, name_rows=name_rows)["confident"]:
            raw = "(number skipped: name)"
        else:
            strip = footer if footer is not None else img.crop((0, int(img.height * 0.72), img.width, img.height))
            raw = cardtext.read_number_text(strip).replace("\n", " | ")
            hit = _find_num(raw) or _find_num_only(raw)
            if hit:
                num, total = hit
            if footer is not None:
                raw = "[hi] " + raw
    verdict = decide(sims, score, topk, pick, num, total, raw, name_rows)
    pick = verdict["pick"]
    # A escolhida vem primeiro; com o nome lido, as cartas desse nome vêm
    # logo depois, e o seletor "Qual destas é a sua?" mostra as variações dele.
    cands = [pick] + [i for i in verdict["order"] if i != pick]
    name_text = " ".join(name_texts)
    t_end = time.time()

    log = (f"name={name_text[:24]!r} nmatch={verdict['name_match']} ocr={num}/{total} "
           f"match={verdict['number_match']} notcard={verdict['not_card']} "
           f"-> {META[pick]['api_id']} cos={float(sims[pick]):.3f} gap={gap:.3f} "
           f"ms={round((t_end - t0) * 1000)} raw={raw[:40]!r}")
    # `card` continua presente com not_card=True: o cliente mostra o aviso
    # de quadro sem carta, e clientes antigos seguem vendo um resultado.
    body = {
        "card": _card(pick, sims),
        "confident": verdict["confident"],
        "not_card": verdict["not_card"],
        "ocr": {"number": num, "total": total, "match": verdict["number_match"],
                "total_match": verdict["total_match"], "raw": raw[:60],
                "name": name_text[:40], "name_match": verdict["name_match"]},
        "candidates": [_card(i, sims) for i in cands[:5]],
        "ms": {"compute": round((t_emb - t0) * 1000), "total": round((t_end - t0) * 1000)},
    }
    return body, log


# Teto de pixels por upload, conferido no cabeçalho ANTES de decodificar. O
# cliente manda a carta com até 900px e o rodapé com até 1200px de largura
# (~1-2 milhões de px); uma imagem gigante decodificada ocupava centenas de MB
# da máquina de 1GB antes de qualquer redução.
_MAX_UPLOAD_PX = 16_000_000


def _open_upload(data: bytes):
    """Open an uploaded image without decoding the pixels yet.

    Returns:
        (image, None), or (None, http_status): 400 when the data is not an
        image, 413 when it has more than _MAX_UPLOAD_PX pixels.
    """
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        return None, 400
    if img.width * img.height > _MAX_UPLOAD_PX:
        return None, 413
    return img, None


@app.post("/recognize")
async def recognize(request: Request, file: UploadFile = File(...), pre: str = Form(None),
                    footer: UploadFile = File(None)):
    """Identify one card.

    Form fields:
        file: JPEG of the card. This server always treats the whole image as
            the card (the client crops it).
        pre: "1" when `file` is already the perspective-corrected card.
        footer: optional JPEG of the bottom 28% of that card at full camera
            resolution. With it (and pre=1), the OCR reads the collector
            number from `footer` instead of from the downscaled `file`.

    Returns {card, confident, not_card, ocr, candidates, ms}. See recognize_image.
    """
    global _recognize_count
    t0 = time.time()
    data = await file.read()
    footer_data = await footer.read() if footer is not None else b""
    if DEBUG_DIR:
        _recognize_count += 1
        try:
            with open(os.path.join(DEBUG_DIR, f"{int(t0 * 1000)}.jpg"), "wb") as f:
                f.write(data)
            if _recognize_count % 50 == 0:
                await asyncio.to_thread(_sweep_debug_dir)
        except OSError:
            pass

    # Cliente desistiu enquanto esperava na fila: não gasta a CPU num
    # resultado que ninguém vai ler (era o que colapsava a fila sob carga).
    if await request.is_disconnected():
        return {"dropped": "client_gone"}

    async with _INFER_SEM:
        if await request.is_disconnected():
            return {"dropped": "client_gone"}
        img, status = _open_upload(data)
        if img is None:
            return JSONResponse({"error": "invalid image" if status == 400 else "image too large"},
                                status_code=status)
        # Rodapé inválido ou grande demais: segue sem ele, e o número sai do
        # rodapé da própria carta.
        strip = _open_upload(footer_data)[0] if footer_data and pre == "1" else None
        body, log = await asyncio.to_thread(recognize_image, img, strip)
        print(f"[REC] {log}", flush=True)
        return body
