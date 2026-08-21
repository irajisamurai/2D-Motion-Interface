"""Stage 2: ViTPose keypoint JSON -> caption (mgpt_2d env).

Wraps ``space/captioner.py``, the inference-only pipeline
(A_real -> E_2D -> codebook -> MotionGPT LM) that needs nothing but the weight
bundle -- no HumanML3D, no glove, no evaluation packages.

Run with the mgpt_2d environment's python:
    python demo/caption_from_json.py --json kp.json
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPACE_DIR = os.path.join(os.path.dirname(HERE), "space")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="keypoint JSON from stage 1")
    ap.add_argument("--bundle", default=os.path.join(SPACE_DIR, "bundle"),
                    help="weight bundle directory (default: 2DMotionGPT/space/bundle)")
    ap.add_argument("--out_json", default=None, help="where to write the result")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--mode", default="both", choices=["both", "adapter", "raw"],
                    help="adapter = with A_real, raw = without, both = side by side")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.bundle):
        sys.exit(f"[error] weight bundle not found: {args.bundle}")

    sys.path.insert(0, SPACE_DIR)
    from captioner import MotionCaptioner

    device = args.device
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("[warn] CUDA unavailable, falling back to CPU")
            device = "cpu"

    captioner = MotionCaptioner(args.bundle, device=device)

    n_frames = len(json.load(open(args.json)))
    result = {"json": os.path.abspath(args.json), "frames_in_json": n_frames}

    modes = {"both": [True, False], "adapter": [True], "raw": [False]}[args.mode]
    for use_adapter in modes:
        out = captioner.caption(*_load(args.json), use_adapter=use_adapter)
        key = "with_adapter" if use_adapter else "without_adapter"
        result[key] = out["caption"]
        result["frames_used"] = out["frames_used"]
        result["n_tokens"] = out["n_tokens"]

    # frames_used is also rounded down to a multiple of unit_length, which is
    # not worth reporting; only the hard length cap is.
    if n_frames > captioner.max_motion_length:
        result["truncated"] = True
        print(f"[warn] clip is longer than the model's {captioner.max_motion_length}-frame "
              f"limit ({n_frames} frames); only the first {result['frames_used']} frames "
              f"({result['frames_used'] / 20:.1f} s) were captioned")

    print()
    for key, label in (("with_adapter", "2D + A_real (ours)"),
                       ("without_adapter", "2D, no adapter")):
        if key in result:
            print(f"  {label:<22} {result[key]}")
    print()

    if args.out_json:
        json.dump(result, open(args.out_json, "w"), indent=2)
        print(f"[caption] wrote {args.out_json}")


def _load(path):
    import features2d as F2D
    return F2D.load_vitpose_json(path)


if __name__ == "__main__":
    main()
