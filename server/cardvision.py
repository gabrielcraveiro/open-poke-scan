"""Image functions shared by the server and the index builder.

The server and the index MUST preprocess cards the same way. A change here
changes the vector space: rebuild the index after any change.
"""
import numpy as np
from PIL import Image

EMB_IN = 336
EMB_DIM = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: Image.Image) -> np.ndarray:
    """Card image -> model input, float32 [1, 3, 336, 336]."""
    a = np.asarray(img.convert("RGB").resize((EMB_IN, EMB_IN), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.transpose(a, (2, 0, 1))[None].astype(np.float32)


def embed(session, img: Image.Image) -> np.ndarray:
    """Card image -> L2-normalized embedding, float32 [384]."""
    x = preprocess(img)
    v = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    return v / (np.linalg.norm(v) + 1e-9)


# pHash da arte: resize 32×32, DCT-II 2D, bloco 8×8, limiar pela média sem o
# termo DC; 64 bits de cinza + 64 bits de matiz. Serve para desempatar prints
# com a mesma arte, onde o embedding quase empata.
_PH_N = 32
_PH_HS = 8
_PH_D = np.cos(np.pi / _PH_N * (np.arange(_PH_N) + 0.5) * np.arange(_PH_N).reshape(_PH_N, 1))
_PH_CROP = (0.08, 0.15, 0.92, 0.68)


def _ph_channel(px: np.ndarray) -> int:
    dct2 = _PH_D @ (_PH_D @ px.T).T
    low = dct2[:_PH_HS, :_PH_HS].flatten()
    mean = low[1:].mean()
    h = 0
    for b in (low > mean).astype(np.uint8):
        h = (h << 1) | int(b)
    return h


def phash(img: Image.Image) -> int:
    """128-bit perceptual hash of the card art area."""
    w, h = img.size
    x0, y0, x1, y1 = _PH_CROP
    art = img.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    arr = np.array(art.resize((_PH_N, _PH_N), Image.Resampling.BILINEAR), dtype=float)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    r, g, b = arr[:, :, 0] / 255, arr[:, :, 1] / 255, arr[:, :, 2] / 255
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-9
    hue = np.where(cmax == r, (g - b) / delta % 6,
          np.where(cmax == g, (b - r) / delta + 2,
                              (r - g) / delta + 4)) / 6 * 255
    hue = np.where(delta < 1e-8, 0.0, hue)
    return (_ph_channel(gray) << 64) | _ph_channel(hue)
