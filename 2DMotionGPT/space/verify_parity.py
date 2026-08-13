"""Check that the exported bundle reproduces the captions recorded during the
full evaluation run, byte for byte, and time it.

By default it checks the four demo clips against space/assets/clips.yaml:

    PYTHONPATH=.:space python space/verify_parity.py --device cpu

Pass --reference <all_clips.json> to check every clip of the evaluation run.
"""

import argparse
import json
import time

from omegaconf import OmegaConf

from captioner import MotionCaptioner


def load_reference(args):
    """-> {id: {"adapter": str, "no_adapter": str, "frames_used": int, "json": path}}"""
    if args.reference:
        ref = {}
        for r in json.load(open(args.reference)):
            ref[r["id"]] = {"adapter": r["pred_adapter"],
                            "no_adapter": r["pred_no_adapter"],
                            "frames_used": r["n_frames_used"],
                            "json": f"{args.pred_dir}/{r['id']}.json"}
        return ref
    cfg = OmegaConf.load(f"{args.assets}/clips.yaml")
    return {c.id: {"adapter": c.expected.adapter,
                   "no_adapter": c.expected.no_adapter,
                   "frames_used": c.frames_used,
                   "json": f"{args.assets}/{c.keypoints}"} for c in cfg.clips}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="./space/bundle")
    p.add_argument("--assets", default="./space/assets")
    p.add_argument("--reference", default=None,
                   help="all_clips.json from the full run; default is the demo clips")
    p.add_argument("--pred_dir", default="./dataset/real_world_dataset_ver2/pred/json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ids", nargs="*", default=None)
    a = p.parse_args()

    ref = load_reference(a)
    ids = a.ids if a.ids else list(ref)

    t0 = time.time()
    cap = MotionCaptioner(a.bundle, device=a.device)
    print(f"model load: {time.time() - t0:.1f}s on {a.device}\n")

    n_ok = n_tot = 0
    times = []
    for i in ids:
        r = ref[i]
        rows = []
        for use_adapter, key in ((True, "adapter"), (False, "no_adapter")):
            t = time.time()
            out = cap.caption_json(r["json"], use_adapter=use_adapter)
            times.append(time.time() - t)
            ok = out["caption"] == r[key]
            n_ok += ok
            n_tot += 1
            rows.append((key, ok, out, r[key]))
        frames = rows[0][2]["frames_used"]
        print(f"{i}  frames {frames} (ref {r['frames_used']}) "
              f"{'OK' if frames == r['frames_used'] else 'MISMATCH'}  "
              f"tokens {rows[0][2]['n_tokens']}")
        for key, ok, out, expected in rows:
            print(f"    {key:11s} {'OK  ' if ok else 'DIFF'} {out['caption']}")
            if not ok:
                print(f"    {'':11s}      expected: {expected}")

    print(f"\nexact match: {n_ok}/{n_tot}")
    print(f"per-caption latency on {a.device}: "
          f"mean {sum(times)/len(times):.2f}s  max {max(times):.2f}s")


if __name__ == "__main__":
    main()
