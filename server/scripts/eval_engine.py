"""Measure the recognition engine on labeled photos and on card footers.

Run it before and after a change to the engine (index, hub penalty,
thresholds, OCR) and compare the two runs. It imports main.py and calls
recognize_image(), the same function that POST /recognize runs.

    cd server
    python scripts/eval_engine.py --photos ~/my-scans --save /tmp/before.json
    # ... change the engine ...
    python scripts/eval_engine.py --photos ~/my-scans --compare /tmp/before.json
    python scripts/eval_engine.py --synth-footers 70     # number OCR only

Photos: one JPEG per scan, with the card already cropped (what the site
sends). The file name starts with the right api_id and two underscores:
"sv07-158__1.jpg". When two prints have identical art and both are right,
join them with "+": "mep-007+me02.5-039__1.jpg". A frame without a card:
"NOT_CARD__1.jpg". With DEBUG_DIR set, the server keeps the photos it gets;
rename the ones you checked.

Synthetic footers: the bottom 28% of official TCGdex images, with blur and
JPEG like a phone photo. The collector number has one exact answer. They go
to .index-cache/eval-footers/ and are downloaded once.

Output:
    photos   cards: right 1st place, right in the top 5, confident answers and
             how many of them are right; frames without a card: how many the
             server flags not_card
    footers  collector number read: ok, number only, wrong, nothing read
"""
import argparse
import io
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
FOOTERS = os.path.join(SERVER, ".index-cache", "eval-footers")
sys.path.insert(0, SERVER)

import cardtext  # noqa: E402
import main as engine  # noqa: E402


def eval_photos(folder: str) -> dict:
    items, ms = {}, []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) or "__" not in name:
            continue
        gold = name.split("__", 1)[0].split("+")
        t = time.perf_counter()
        body, _ = engine.recognize_image(Image.open(os.path.join(folder, name)))
        ms.append((time.perf_counter() - t) * 1000)
        items[name] = {"gold": gold, "pick": body["card"]["api_id"], "confident": body["confident"],
                       "not_card": body["not_card"], "top5": [c["api_id"] for c in body["candidates"]],
                       "name_match": body["ocr"]["name_match"]}
    return {"items": items, "ms": ms}


def summarize_photos(res: dict) -> None:
    items = res["items"].values()
    cards = [x for x in items if x["gold"] != ["NOT_CARD"]]
    frames = [x for x in items if x["gold"] == ["NOT_CARD"]]
    ok = [x for x in cards if x["pick"] in x["gold"] and not x["not_card"]]
    conf = [x for x in cards if x["confident"] and not x["not_card"]]
    ms = np.array(res["ms"]) if res["ms"] else np.zeros(1)
    print(f"photos: {len(cards)} cards, {len(frames)} frames without a card")
    print(f"  right 1st place   {len(ok)}/{len(cards)}")
    print(f"  right in top 5    {sum(bool(set(x['top5']) & set(x['gold'])) for x in cards)}/{len(cards)}")
    print(f"  confident         {len(conf)} (right {sum(x['pick'] in x['gold'] for x in conf)})")
    print(f"  name matched      {sum(x['name_match'] for x in cards)}")
    print(f"  no card           {sum(x['not_card'] for x in frames)}/{len(frames)} flagged not_card, "
          f"{sum(x['confident'] and not x['not_card'] for x in frames)} confident")
    print(f"  ms per photo      p50 {np.median(ms):.0f}  p90 {np.percentile(ms, 90):.0f}")


def make_footers(count: int, seed: int) -> None:
    """Download official images and save their degraded bottom 28%."""
    pool = [c for c in engine.META
            if "assets.tcgdex.net" in str(c.get("image_url")) and str(c.get("number", "")).isdigit()
            and c.get("printed_total")]
    random.seed(seed)
    os.makedirs(FOOTERS, exist_ok=True)
    for c in random.sample(pool, min(count, len(pool))):
        path = os.path.join(FOOTERS, f"{c['api_id']}.jpg")
        if os.path.exists(path):
            continue
        url = c["image_url"].replace("/low.webp", "/high.webp")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "open-poke-scan-eval"})
            data = urllib.request.urlopen(req, timeout=15).read()
        except OSError as e:
            print(f"  skipped {c['api_id']}: {e}", file=sys.stderr)
            continue
        im = Image.open(io.BytesIO(data)).convert("RGB").resize((900, 1257))
        im = im.filter(ImageFilter.GaussianBlur(1.2))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
        im = Image.open(buf)
        im.crop((0, int(im.height * 0.72), im.width, im.height)).save(path, quality=85)
        time.sleep(0.2)


def _grade(num, total, card) -> str:
    if not num:
        return "none"
    if engine._digits(num) != engine._digits(card["number"]):
        return "wrong"
    if total is None:
        return "ok_num"
    return "ok" if str(int(engine._digits(total))) == str(card.get("printed_total")) else "wrong"


def eval_footers() -> dict:
    by_id = {c["api_id"]: c for c in engine.META}
    items, ms = {}, []
    for name in sorted(os.listdir(FOOTERS)) if os.path.isdir(FOOTERS) else []:
        card = by_id.get(name.rsplit(".", 1)[0])
        if card is None:
            continue
        t = time.perf_counter()
        text = cardtext.read_number_text(Image.open(os.path.join(FOOTERS, name)))
        ms.append((time.perf_counter() - t) * 1000)
        hit = engine._find_num(text) or engine._find_num_only(text) or (None, None)
        items[name] = {"grade": _grade(hit[0], hit[1], card)}
    return {"items": items, "ms": ms}


def summarize_footers(res: dict) -> None:
    grades = [x["grade"] for x in res["items"].values()]
    if not grades:
        return
    ms = np.array(res["ms"])
    print(f"footers: {len(grades)}  ok {grades.count('ok')}  number only {grades.count('ok_num')}  "
          f"wrong {grades.count('wrong')}  nothing {grades.count('none')}  "
          f"ms p50 {np.median(ms):.0f} p90 {np.percentile(ms, 90):.0f}")


def compare(old: dict, new: dict) -> None:
    """Print the photos whose answer changed between two runs."""
    fixed, broken = [], []
    for name, n in new.get("photos", {}).get("items", {}).items():
        o = old.get("photos", {}).get("items", {}).get(name)
        if o is None or n["gold"] == ["NOT_CARD"]:
            continue
        was, now = o["pick"] in n["gold"], n["pick"] in n["gold"]
        if now and not was:
            fixed.append(f"{name}: {o['pick']} → {n['pick']}")
        elif was and not now:
            broken.append(f"{name}: {o['pick']} → {n['pick']}")
    print(f"compare: {len(fixed)} fixed, {len(broken)} broken")
    for line in fixed:
        print("  +", line)
    for line in broken:
        print("  -", line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", help="folder with labeled photos (see above)")
    ap.add_argument("--synth-footers", type=int, default=0, help="download this many synthetic footers")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", help="write the results to this JSON file")
    ap.add_argument("--compare", help="JSON of an earlier run; prints what changed")
    args = ap.parse_args()
    if not args.photos and not args.synth_footers and not os.path.isdir(FOOTERS):
        ap.error("give --photos and/or --synth-footers")

    res = {}
    if args.photos:
        res["photos"] = eval_photos(args.photos)
        summarize_photos(res["photos"])
    if args.synth_footers:
        make_footers(args.synth_footers, args.seed)
    res["footers"] = eval_footers()
    summarize_footers(res["footers"])
    if args.compare:
        with open(args.compare) as f:
            compare(json.load(f), res)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
