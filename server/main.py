"""open-poke-scan recognition server.

POST /recognize: photo of one card -> the matching catalog card.

Pipeline: DINOv2-small embedding -> cosine similarity against the index, minus
the hub penalty of each card -> pHash tie-break between prints with the same
art -> OCR of the collector number (Tesseract) only when the match is ambiguous.

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
import pytesseract
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

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

# ── OCR do número ────────────────────────────────────────────────────────────
# Configs de um grid search offline (24 combinações de tira/escala/PSM):
# o composto abaixo leu 5/10 com 0 leituras erradas em ~470ms.
_TESS_BLOCK = "--psm 6 -c tessedit_char_whitelist=0123456789/"
_TESS_SPARSE = "--psm 11 -c tessedit_char_whitelist=0123456789/"
_TESS_LINE = "--psm 7 -c tessedit_char_whitelist=0123456789/"
_NUM_RE = re.compile(r"(\d{1,4})\s*/\s*(\d{1,3})(?!\d)")
# Largura da tira antes do upscale 3×. Sem normalizar, uploads grandes
# estouravam o timeout do OCR em todo scan.
_OCR_STRIP_W = 600
# Cantos de baixo do rodapé em alta (campo `footer`), onde fica o número:
# esquerdo nas cartas atuais, direito nas antigas. A faixa inteira tinha
# ataque, fraqueza e copyright: o Tesseract gastava tempo e se confundia.
# 32 rodapés reais com gabarito: 7 números certos contra 6 da faixa inteira, e
# o pior caso de 804 para 584ms.
_OCR_CORNERS = ((0.0, 0.66, 0.50, 1.0), (0.55, 0.66, 1.0, 1.0))
# Altura do recorte depois do upscale: a linha do número ocupa ~1/3 dele, e o
# Tesseract lê melhor com letras de ~30-40px.
_OCR_CORNER_H = 110


def _digits(s) -> str:
    return re.sub(r"\D", "", str(s or "")).lstrip("0") or "0"


def _find_num(text):
    """First plausible NNN/NNN in the Tesseract text, or None."""
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


def _norm_strip(strip: Image.Image) -> Image.Image:
    if strip.width > _OCR_STRIP_W:
        nh = max(1, round(strip.height * _OCR_STRIP_W / strip.width))
        strip = strip.resize((_OCR_STRIP_W, nh))
    return strip.resize((max(1, strip.width * 3), max(1, strip.height * 3)))


def _tess(deadline: float):
    # Cada chamada recebe o tempo que sobra e é MORTA quando ele acaba: uma
    # chamada abandonada seguiria queimando a única CPU enquanto o próximo
    # scan espera.
    def run(im, cfg):
        left = deadline - time.time()
        if left <= 0.05:
            return None
        try:
            return pytesseract.image_to_string(im, config=cfg, timeout=left)
        except RuntimeError:  # pytesseract: "Tesseract process timeout"
            return None
    return run


def read_number(img: Image.Image, budget_s: float):
    """OCR of the collector number in the bottom of `img`. Returns (number, total, raw_text)."""
    tess = _tess(time.time() + budget_s)
    w, h = img.size
    gray = img.convert("L")
    raws = []
    band = _norm_strip(gray.crop((0, int(h * 0.86), w, h)))
    text = tess(band, _TESS_SPARSE)
    if text is None:
        return None, None, "(ocr timeout)"
    raws.append(text.strip())
    hit = _find_num(text)
    if hit:
        return hit[0], hit[1], " | ".join(raws)
    # O número fica embaixo à esquerda OU à direita, conforme a era do set.
    strip = _norm_strip(gray.crop((0, int(h * 0.72), w, h)))
    sw, sh = strip.size
    for half in (strip.crop((0, 0, sw // 2, sh)), strip.crop((sw // 2, 0, sw, sh))):
        text = tess(half, _TESS_BLOCK)
        if text is None:
            return None, None, " | ".join(raws + ["(ocr timeout)"])
        raws.append(text.strip())
        hit = _find_num(text)
        if hit:
            return hit[0], hit[1], " | ".join(raws)
    return None, None, " | ".join(raws)


def read_number_footer(strip: Image.Image, budget_s: float):
    """OCR of the collector number in the full-resolution footer (bottom 28% of the card).

    Reads only the two bottom corners. Returns (number, total, raw_text).
    """
    tess = _tess(time.time() + budget_s)
    gray = strip.convert("L")
    w, h = gray.size
    raws, partial = [], None
    for x0, y0, x1, y1 in _OCR_CORNERS:
        c = gray.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
        up = max(1.0, _OCR_CORNER_H / max(1, c.height))
        c = ImageOps.autocontrast(c.resize((round(c.width * up), round(c.height * up))))
        for cfg in (_TESS_LINE, _TESS_SPARSE):
            text = tess(c, cfg)
            if text is None:
                return None, None, " | ".join(raws + ["(ocr timeout)"])
            raws.append(text.strip())
            hit = _find_num(text)
            if hit:
                return hit[0], hit[1], " | ".join(raws)
            partial = partial or _find_num_only(text)
    if partial:
        return partial[0], None, " | ".join(raws)
    return None, None, " | ".join(raws)


# Prazo do OCR. Estourou, segue sem número: o embedding reconhece sozinho.
_OCR_TIMEOUT_S = 1.1
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
# Gap top1−top2 abaixo disso = empate de arte, e o pHash decide.
_PHASH_TIE_GAP = 0.02

# Uma inferência por vez. Com 1 vCPU, duas em paralelo levam o dobro cada,
# o OCR estoura o timeout e a fila colapsa.
_INFER_SEM = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENCY", "1")))


def _name_key(i: int) -> str:
    # "Buzzwole-GX" e "Buzzwole GX" são a mesma carta com grafias diferentes.
    return re.sub(r"[^a-z0-9]", "", str(META[i]["name"]).lower())


def decide(sims, score, topk, pick, num=None, total=None):
    """Apply the OCR fusion and the confidence gate to one ranking.

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

    Returns:
        dict with keys pick, number_match, total_match, confident, not_card.
    """
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
    top_cos = float(max(sims[i] for i in topk))
    others = [float(score[i]) for i in topk if i != pick]
    gap = float(score[pick]) - (max(others) if others else 0.0)
    confident = number_match or (float(sims[pick]) >= _CONF_COS and (gap >= _CONF_GAP or total_ok))
    return {"pick": pick, "number_match": number_match, "total_match": total_match,
            "confident": confident, "not_card": top_cos < _NOT_CARD_COS}


async def _ocr_bounded(img, footer):
    # O prazo real é o do Tesseract (morto ao estourar); o wait_for com folga
    # só protege a resposta se algo fora dele travar.
    def job():
        if footer is not None:
            return read_number_footer(footer, _OCR_TIMEOUT_S)
        return read_number(img, _OCR_TIMEOUT_S)
    try:
        return await asyncio.wait_for(asyncio.to_thread(job), timeout=_OCR_TIMEOUT_S + 0.3)
    except asyncio.TimeoutError:
        return None, None, "(ocr timeout)"


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

    Returns {card, confident, not_card, ocr, candidates, ms}.
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
        img = Image.open(io.BytesIO(data)).convert("RGB")
        vec = await asyncio.to_thread(embed, emb_sess, img)
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

        # O OCR só roda quando pode mudar a resposta. Pula quando o embedding
        # já daria a resposta como confiante E os dois primeiros não têm o
        # mesmo nome. Mesmo nome = mesma arte em outro set: aí só o número
        # desempata.
        num = total = None
        top_cos = float(sims[pick])
        if top_cos < _NOT_CARD_COS:
            raw = "(ocr skipped: not a card)"
        elif (top_cos >= _CONF_COS and gap >= _CONF_GAP
              and (len(topk) < 2 or _name_key(topk[0]) != _name_key(topk[1]))):
            raw = "(ocr skipped: decisive)"
        else:
            strip = None
            if footer_data and pre == "1":
                try:
                    strip = Image.open(io.BytesIO(footer_data))
                except Exception:
                    strip = None
            num, total, raw = await _ocr_bounded(img, strip)
            if strip is not None:
                raw = "[hi] " + raw
        verdict = decide(sims, score, topk, pick, num, total)
        pick = verdict["pick"]

        print(f"[REC] ocr={num}/{total} match={verdict['number_match']} notcard={verdict['not_card']} "
              f"-> {META[pick]['api_id']} cos={float(sims[pick]):.3f} gap={gap:.3f} "
              f"ms={round((time.time() - t0) * 1000)} raw={raw[:40]!r}", flush=True)
        # `card` continua presente com not_card=True: o cliente mostra o aviso
        # de quadro sem carta, e clientes antigos seguem vendo um resultado.
        return {
            "card": _card(pick, sims),
            "confident": verdict["confident"],
            "not_card": verdict["not_card"],
            "ocr": {"number": num, "total": total, "match": verdict["number_match"],
                    "total_match": verdict["total_match"], "raw": raw[:60]},
            "candidates": [_card(i, sims) for i in topk[:5]],
            "ms": {"compute": round((t_emb - t0) * 1000), "total": round((time.time() - t0) * 1000)},
        }
