#!/usr/bin/env python3
"""Build the card index from the public TCGdex catalog.

Downloads every English card image from TCGdex, computes the embedding and the
pHash of each card, and writes the two files the server loads:

    models/emb.f16.bin   fp16 vectors, one 384-d row per card
    models/meta.json     {dim, count, cards: [...]}, aligned 1:1 with the rows

Resumable: images and per-card results are cached in --cache, so a second run
only processes new cards. A full run (~22k cards) downloads ~600 MB and takes
about 1-3 hours on a laptop CPU.

Usage:
    python scripts/build_index.py                       # all sets
    python scripts/build_index.py --sets sv07,sv08      # only some sets (quick test)
"""
import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cardvision import EMB_DIM, embed, phash  # noqa: E402

API = "https://api.tcgdex.net/v2/en"
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "open-poke-scan-index-builder"


def get_json(url: str, tries: int = 3):
    for i in range(tries):
        try:
            r = HTTP.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def list_cards(only_sets: set[str] | None) -> list[dict]:
    """All cards with an image, with set name and printed total."""
    cards = []
    sets = get_json(f"{API}/sets") or []
    for s in sets:
        if only_sets and s["id"] not in only_sets:
            continue
        detail = get_json(f"{API}/sets/{s['id']}")
        if not detail:
            continue
        # official = total impresso no rodapé ("158/142" → 142). Promos não têm (0).
        official = (detail.get("cardCount") or {}).get("official") or None
        for c in detail.get("cards") or []:
            if not c.get("image"):
                continue
            cards.append({
                "api_id": c["id"],
                "name": c.get("name") or "",
                "number": c.get("localId") or "",
                "set_id": detail["id"],
                "set_name": detail.get("name") or "",
                "printed_total": official,
                "image_url": c["image"] + "/low.webp",
            })
        print(f"  {detail['id']:<12} {len(detail.get('cards') or []):>4} cards", flush=True)
    return cards


def fetch_image(card: dict, cache: str) -> str | None:
    path = os.path.join(cache, "img", card["api_id"].replace("/", "_") + ".webp")
    if os.path.exists(path):
        return path
    try:
        r = HTTP.get(card["image_url"], timeout=30)
        if r.status_code != 200:
            return None
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except requests.RequestException:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/model.onnx")
    ap.add_argument("--out", default="models")
    ap.add_argument("--cache", default=".index-cache")
    ap.add_argument("--sets", default="", help="comma-separated TCGdex set ids (default: all)")
    ap.add_argument("--workers", type=int, default=8, help="parallel image downloads")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.cache, "img"), exist_ok=True)
    os.makedirs(os.path.join(args.cache, "vec"), exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    only = {s.strip() for s in args.sets.split(",") if s.strip()} or None

    print("listing cards from TCGdex…", flush=True)
    cards = list_cards(only)
    print(f"{len(cards)} cards with image", flush=True)

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    rows, meta = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Download em paralelo; o embedding roda na thread principal, um por vez.
        for i, (card, path) in enumerate(zip(cards, pool.map(lambda c: fetch_image(c, args.cache), cards))):
            vec_path = os.path.join(args.cache, "vec", card["api_id"].replace("/", "_") + ".npz")
            if os.path.exists(vec_path):
                z = np.load(vec_path)
                vec, ph = z["v"], str(z["p"])
            elif path:
                try:
                    img = Image.open(io.BytesIO(open(path, "rb").read()))
                    vec = embed(session, img).astype(np.float16)
                    ph = f"{phash(img):032x}"
                    np.savez(vec_path, v=vec, p=ph)
                except Exception as e:  # imagem corrompida: pula a carta, não o lote
                    print(f"  skip {card['api_id']}: {e}", flush=True)
                    continue
            else:
                print(f"  skip {card['api_id']}: image download failed", flush=True)
                continue
            rows.append(vec)
            meta.append({**card, "phash": ph})
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(cards)}  ({rate:.1f} cards/s)", flush=True)

    if not rows:
        sys.exit("no cards indexed")
    np.stack(rows).astype(np.float16).tofile(os.path.join(args.out, "emb.f16.bin"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"dim": EMB_DIM, "count": len(meta), "cards": meta}, f, ensure_ascii=False)
    print(f"wrote {len(meta)} cards to {args.out}/emb.f16.bin + meta.json", flush=True)


if __name__ == "__main__":
    main()
