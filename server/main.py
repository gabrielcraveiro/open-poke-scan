"""open-poke-scan recognition server.

POST /recognize: photo of one card -> the matching catalog card.

Pipeline: DINOv2-small embedding -> cosine similarity against the index ->
pHash tie-break between prints with the same art -> OCR of the collector
number (Tesseract) only when the match is ambiguous.

    uvicorn main:app --host 0.0.0.0 --port 8000

Environment:
    MODELS_DIR     folder with model.onnx, emb.f16.bin, meta.json (default: ./models)
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
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from cardvision import EMB_DIM, embed, phash

MODELS = os.environ.get("MODELS_DIR") or os.path.join(os.path.dirname(__file__), "models")
emb_sess = ort.InferenceSession(os.path.join(MODELS, "model.onnx"), providers=["CPUExecutionProvider"])

MAT = np.fromfile(os.path.join(MODELS, "emb.f16.bin"), dtype=np.float16).astype(np.float32).reshape(-1, EMB_DIM)
MAT /= np.linalg.norm(MAT, axis=1, keepdims=True) + 1e-9
with open(os.path.join(MODELS, "meta.json")) as _f:
    META = json.load(_f)["cards"]
PHASH = []
for _c in META:
    _p = _c.get("phash")
    try:
        PHASH.append(int(_p, 16) if _p and len(_p) == 32 else None)
    except (TypeError, ValueError):
        PHASH.append(None)

# ── OCR do número ────────────────────────────────────────────────────────────
# Configs de um grid search offline (24 combinações de tira/escala/PSM):
# o composto abaixo leu 5/10 com 0 leituras erradas em ~470ms. PSM 7 (linha
# única) leu zero: a tira tem várias linhas.
_TESS_BLOCK = "--psm 6 -c tessedit_char_whitelist=0123456789/"
_TESS_SPARSE = "--psm 11 -c tessedit_char_whitelist=0123456789/"
_NUM_RE = re.compile(r"(\d{1,4})\s*/\s*(\d{1,3})(?!\d)")
# Largura da tira antes do upscale 3×. Sem normalizar, uploads grandes
# estouravam o timeout do OCR em todo scan.
_OCR_STRIP_W = 600


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


def _norm_strip(strip: Image.Image) -> Image.Image:
    if strip.width > _OCR_STRIP_W:
        nh = max(1, round(strip.height * _OCR_STRIP_W / strip.width))
        strip = strip.resize((_OCR_STRIP_W, nh))
    return strip.resize((max(1, strip.width * 3), max(1, strip.height * 3)))


def read_number(img: Image.Image):
    """OCR of the collector number. Returns (number, total, raw_text)."""
    w, h = img.size
    gray = img.convert("L")
    raws = []
    band = _norm_strip(gray.crop((0, int(h * 0.86), w, h)))
    text = pytesseract.image_to_string(band, config=_TESS_SPARSE)
    raws.append(text.strip())
    hit = _find_num(text)
    if hit:
        return hit[0], hit[1], " | ".join(raws)
    # O número fica embaixo à esquerda OU à direita, conforme a era do set.
    strip = _norm_strip(gray.crop((0, int(h * 0.72), w, h)))
    sw, sh = strip.size
    for half in (strip.crop((0, 0, sw // 2, sh)), strip.crop((sw // 2, 0, sw, sh))):
        text = pytesseract.image_to_string(half, config=_TESS_BLOCK)
        raws.append(text.strip())
        hit = _find_num(text)
        if hit:
            return hit[0], hit[1], " | ".join(raws)
    return None, None, " | ".join(raws)


# Tesseract em tira ruidosa pode levar segundos e segura a fila inteira.
# Estourou, segue sem número: o embedding reconhece sozinho.
_OCR_TIMEOUT_S = 1.5
# O OCR só roda quando muda a resposta. Match decisivo (cosseno alto e folga
# sobre o 2º) dispensa; cosseno muito baixo = nem é uma carta.
_OCR_SKIP_COS = 0.80
_OCR_SKIP_GAP = 0.05
_OCR_MIN_COS = 0.35
# Gap top1−top2 abaixo disso = empate de arte, e o pHash decide.
_PHASH_TIE_GAP = 0.02

# Uma inferência por vez. Com 1 vCPU, duas em paralelo levam o dobro cada,
# o OCR estoura o timeout e a fila colapsa.
_INFER_SEM = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENCY", "1")))


async def _ocr_bounded(img):
    try:
        return await asyncio.wait_for(asyncio.to_thread(read_number, img), timeout=_OCR_TIMEOUT_S)
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
async def recognize(request: Request, file: UploadFile = File(...)):
    """Identify one card.

    Form field `file`: JPEG of the card. The web client also sends `pre=1`;
    this server ignores it and always treats the whole image as the card
    (the client already crops it).
    Returns {card, confident, ocr, candidates, ms}.
    """
    global _recognize_count
    t0 = time.time()
    data = await file.read()
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
        topk = [int(i) for i in np.argsort(-sims)[:25]]
        pick = topk[0]
        gap = float(sims[topk[0]] - sims[topk[1]]) if len(topk) > 1 else 1.0
        if gap < _PHASH_TIE_GAP:
            tied = [i for i in topk if float(sims[topk[0]] - sims[i]) < _PHASH_TIE_GAP and PHASH[i] is not None]
            if len(tied) > 1:
                qh = phash(img)
                pick = min(tied, key=lambda i: (qh ^ PHASH[i]).bit_count())

        num = total = None
        top_cos = float(sims[pick])
        if top_cos < _OCR_MIN_COS:
            raw = "(ocr skipped: not a card)"
        elif top_cos >= _OCR_SKIP_COS and gap >= _OCR_SKIP_GAP:
            raw = "(ocr skipped: decisive)"
        else:
            num, total, raw = await _ocr_bounded(img)
        number_match = False
        if num:
            dn = _digits(num)
            pool = [i for i in topk if _digits(META[i]["number"]) == dn]
            if pool:
                if total:
                    dt = str(int(_digits(total)))
                    with_total = [i for i in pool if str(META[i].get("printed_total")) == dt]
                    # Número bate mas o total contradiz: a carta provavelmente
                    # não está no catálogo. Não força; fica o embedding.
                    if with_total:
                        pick, number_match = with_total[0], True
                else:
                    pick, number_match = pool[0], True
        confident = number_match or float(sims[pick]) >= 0.55

        print(f"[REC] ocr={num}/{total} match={number_match} -> {META[pick]['api_id']} "
              f"cos={float(sims[pick]):.3f} gap={gap:.3f} ms={round((time.time() - t0) * 1000)}", flush=True)
        return {
            "card": _card(pick, sims),
            "confident": confident,
            "ocr": {"number": num, "total": total, "match": number_match, "raw": raw[:60]},
            "candidates": [_card(i, sims) for i in topk[:5]],
            "ms": {"compute": round((t_emb - t0) * 1000), "total": round((time.time() - t0) * 1000)},
        }
