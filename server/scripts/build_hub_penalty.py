"""Build models/hub.json: the per-card "hub" penalty that main.py subtracts from the cosine.

Some catalog cards sit close to MANY real photos (washed-out holo, low
contrast) and win first place for photos of other cards. This script measures
that pull per card from a folder of real scan photos:

    r[card] = mean cosine between the card and its K most similar photos

Each photo is left out of the penalty of its own most similar card, so a card
that you scan often is not penalized for looking like itself. There are no
labels here, so "its own card" is the nearest one: a good proxy when most scans
were right.

The server's DEBUG_DIR keeps the uploaded photos for 6 hours. Copy them to a
folder from time to time, then:

    python scripts/build_hub_penalty.py --photos ~/scan-photos
    # restart the server (or redeploy) to load models/hub.json

On the published server (121 labeled scans, K=20, alpha=0.25 in main.py): first
place right went from 71% to 79%, with no scan made worse. Use at least a few
hundred photos: with few, the penalty is noise.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cardvision import EMB_DIM, embed  # noqa: E402

_POCKET_SET_RE = re.compile(r"^(A\d|B\d|P-[AB])")   # the same filter as main.py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", required=True, help="folder with real scan photos (.jpg, .png)")
    ap.add_argument("--models", default="models", help="folder with model.onnx, emb.f16.bin, meta.json")
    ap.add_argument("--k", type=int, default=20, help="photos averaged per card")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.models, "meta.json")))["cards"]
    mat = np.fromfile(os.path.join(args.models, "emb.f16.bin"), dtype=np.float16).astype(np.float32).reshape(-1, EMB_DIM)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    keep = [i for i, c in enumerate(meta) if not _POCKET_SET_RE.match(c.get("set_id") or "")]
    mat = mat[keep]
    ids = [meta[i]["api_id"] for i in keep]

    files = sorted(f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(args.photos, f"*.{ext}")))
    if len(files) < args.k * 5:
        sys.exit(f"only {len(files)} photos in {args.photos}: use at least {args.k * 5}.")
    sess = ort.InferenceSession(os.path.join(args.models, "model.onnx"), providers=["CPUExecutionProvider"])
    bank = []
    for n, f in enumerate(files, 1):
        try:
            bank.append(embed(sess, Image.open(f).convert("RGB")))
        except Exception:
            continue
        if n % 100 == 0:
            print(f"{n}/{len(files)} photos", flush=True)
    bank = np.stack(bank)

    sims = mat @ bank.T                  # card × photo
    own = np.argmax(sims, axis=0)        # each photo's nearest card
    pen = np.empty(len(ids), np.float32)
    for i in range(len(ids)):
        s = sims[i][own != i]
        pen[i] = np.partition(s, -args.k)[-args.k:].mean()
    out = {"k": args.k, "photos": int(len(bank)), "median": round(float(np.median(pen)), 4),
           "cards": {a: round(float(p), 4) for a, p in zip(ids, pen)}}
    path = os.path.join(args.models, "hub.json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    worst = np.argsort(-pen)[:8]
    print(f"photos={len(bank)} cards={len(ids)} median={out['median']} -> {path}")
    print("highest:", [(ids[i], round(float(pen[i]), 3)) for i in worst])


if __name__ == "__main__":
    main()
