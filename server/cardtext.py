"""Text on the card: the collector number and the card name.

OCR is RapidOCR (PP-OCRv4 detector + recognizer, ONNX, Apache-2.0) on the CPU.
The published server (cartinhas-app/server) and the open server
(open-poke-scan/server) use this same file. Keep the two copies identical.
"""
import difflib
import re
import unicodedata

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# Uma thread: o servidor roda um reconhecimento por vez numa máquina de 1 vCPU.
# limit_type "max" com 960: abaixo de 960px o detector NÃO amplia a imagem. O
# padrão do RapidOCR ("min" 736) ampliava o rodapé até ~1900px e o detector
# levava ~1,1s por imagem em vez de ~90ms.
_ENGINE = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1,
                   det_limit_type="max", det_limit_side_len=960)

# Bancada de 2026-10-15 (70 rodapés sintéticos de cartas oficiais com blur e
# JPEG, e 98 rodapés reais do scan_log):
# - Tesseract (o OCR anterior): 21/70 e 1/98 números certos, ~500ms.
# - Rodapé em 640px de largura, só caixas curtas, as 6 mais baixas: 57/70 e
#   28/98 certos, 1 leitura errada nos sintéticos, ~180ms (p90 210ms).
# - Rodapé inteiro sem filtro: o mesmo acerto, mas até 2,4s, porque o texto
#   de ataque e de descrição vira dezenas de caixas para o reconhecedor.
_NUMBER_W = 640
_NUMBER_MAX_RATIO = 9.0
_NUMBER_MAX_BOXES = 6
# Faixa do nome: 14% de cima da carta. Com 11% o nome de cartas um pouco
# tortas ficava de fora: 46 contra 62 nomes lidos em 129 scans reais.
_NAME_BAND = 0.14
_NAME_W = 640
_NAME_MAX_BOXES = 6


def _box_size(box: np.ndarray) -> tuple[float, float]:
    w = float(np.linalg.norm(box[0] - box[1]))
    h = float(np.linalg.norm(box[0] - box[3]))
    return w, max(h, 1.0)


# Teto da altura depois de levar a imagem para a largura-alvo. O rodapé real
# tem ~0,4 de altura por largura e a faixa do nome ~0,2. Sem o teto, uma
# imagem estreita e alta (10×100000px) virava 640×6 milhões de px e esgotava a
# memória do servidor.
_MAX_H_PER_W = 1.0


def _to_width(img: Image.Image, width: int) -> Image.Image:
    h = round(img.height * width / max(1, img.width))
    h = min(max(1, h), round(width * _MAX_H_PER_W))
    return img.convert("RGB").resize((width, h), Image.Resampling.LANCZOS)


def _ocr(img: Image.Image, keep=None, order=None, max_boxes: int = 6) -> list[str]:
    bgr = np.asarray(img)[:, :, ::-1].copy()
    boxes, _ = _ENGINE.text_det(bgr)
    if boxes is None or len(boxes) == 0:
        return []
    boxes = [b for b in boxes if keep is None or keep(b, bgr.shape[0])]
    boxes = (sorted(boxes, key=order) if order else _ENGINE.sorted_boxes(np.array(boxes)))[:max_boxes]
    if not boxes:
        return []
    crops = _ENGINE.get_crop_img_list(bgr, [np.asarray(b, dtype=np.float32) for b in boxes])
    rec, _ = _ENGINE.text_rec(crops)
    return [r[0] for r in rec if r[0]]


def read_number_text(footer: Image.Image) -> str:
    """Read the short text lines at the bottom of a card footer.

    Args:
        footer: the bottom part of the card (the client sends the bottom 28%).

    Returns:
        The text of up to 6 short lines, bottom line first, one per line.
        The collector number ("092/182") is one of them when OCR can read it.
        Empty string when OCR finds no text.

    Example:
        >>> read_number_text(Image.open("footer.jpg"))
        '092/182\\nDRI PT\\nIllus. Shibuzoh.'
    """
    # O número é uma linha curta perto da borda de baixo. Texto de ataque e
    # de descrição são linhas longas: ficam de fora e não custam CPU.
    def short(box, _h):
        w, h = _box_size(box)
        return w / h <= _NUMBER_MAX_RATIO

    texts = _ocr(_to_width(footer, _NUMBER_W), keep=short,
                 order=lambda b: -float(b[:, 1].mean()), max_boxes=_NUMBER_MAX_BOXES)
    return "\n".join(texts)


def read_name_texts(card: Image.Image) -> list[str]:
    """Read the text lines of the name band (the top 14% of the card).

    Args:
        card: the whole card, cropped and upright.

    Returns:
        Up to 6 text lines, top to bottom. The name is usually one of them,
        next to the stage ("BASIC", "BÁSICO") and the HP. Empty list when OCR
        finds no text.

    Example:
        >>> read_name_texts(Image.open("card.jpg"))
        ['BÁSICO', 'Jynx', 'PS110']
    """
    w, h = card.size
    band = card.crop((0, 0, w, max(1, int(h * _NAME_BAND))))
    return _ocr(_to_width(band, _NAME_W), max_boxes=_NAME_MAX_BOXES)


def name_norm(text: str) -> str:
    """Lowercase ASCII letters and digits only: "Ho-Oh" -> "hooh"."""
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


# Sufixos de mecânica: a carta "Lapras ex" e a "Lapras" são o mesmo Pokémon
# para o nome lido, e o embedding separa as duas.
_SUFFIX_RE = re.compile(r"[\s-]+(ex|gx|v|vmax|vstar|v-union|break|lv\.?\s*x|prime|legend|star|δ)$", re.I)
# Nomes base com menos letras casam por acaso dentro de outro texto.
_MIN_BASE = 4


def name_base(name: str) -> str:
    """Card name without the mechanic suffix, normalized: "Lapras ex" -> "lapras"."""
    n, prev = str(name or "").strip(), None
    while prev != n:
        prev, n = n, _SUFFIX_RE.sub("", n).strip()
    return name_norm(n)


class NameIndex:
    """Catalog card names, for matching the name that OCR reads.

    Args:
        names: card name per catalog row.
        skip: optional flag per row. Rows with True never match. The catalog
            of the published server has Japanese sets with Portuguese names
            from the Liga; a Portuguese card would match them by name.
    """

    def __init__(self, names, skip=None):
        self._rows: dict[str, list[int]] = {}
        for i, name in enumerate(names):
            if skip is not None and skip[i]:
                continue
            base = name_base(name)
            if len(base) >= _MIN_BASE:
                self._rows.setdefault(base, []).append(i)
        # O nome mais longo primeiro: "Kadabra" contém "abra".
        self._keys = sorted(self._rows, key=len, reverse=True)

    def match(self, texts, fuzzy: float = 0.8) -> set[str]:
        """Return the base names found in the OCR texts.

        A text matches a name when it contains the whole base name
        ("BASICOJynx" contains "jynx"). Without that, the closest name with a
        similarity of at least `fuzzy` matches ("Cranide" -> "cranidos").
        """
        found = set()
        for text in texts:
            t = name_norm(text)
            if len(re.sub(r"[0-9]", "", t)) < _MIN_BASE:
                continue
            hit = next((k for k in self._keys if k in t), None)
            if hit is None and fuzzy:
                close = difflib.get_close_matches(t, self._keys, n=1, cutoff=fuzzy)
                hit = close[0] if close else None
            if hit:
                found.add(hit)
        return found

    def rows(self, bases, score: np.ndarray) -> list[int]:
        """Catalog rows with any of the base names, best `score` first."""
        rows = {i for b in bases for i in self._rows.get(b, [])}
        return sorted(rows, key=lambda i: -float(score[i]))
