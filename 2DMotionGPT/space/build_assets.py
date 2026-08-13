"""Assemble the demo assets for the Space: the four clips' overlay videos,
their ViTPose keypoints, and a manifest with the ground-truth captions and the
captions recorded during the full 132-clip evaluation run.

    PYTHONPATH=. python space/build_assets.py --reference <all_clips.json>
"""

import argparse
import csv
import json
import os
import shutil

from omegaconf import OmegaConf
from os.path import join as pjoin

# id -> label shown in the UI
DEMO_CLIPS = {
    "M004947": "Sitting down cross-legged, then standing back up",
    "000099":  "Jogging in place",
    "005486":  "Jumping and throwing with both hands",
    "M001523": "Reaching for the back of the head",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./dataset/real_world_dataset_ver2")
    p.add_argument("--reference", required=True,
                   help="all_clips.json from the full evaluation run")
    p.add_argument("--out", default="./space/assets")
    a = p.parse_args()

    ref = {r["id"]: r for r in json.load(open(a.reference))}
    gt_csv = {r["data_name"]: r["text"]
              for r in csv.DictReader(open(pjoin(a.dataset, "humanml3d_dataset.csv")))}

    os.makedirs(pjoin(a.out, "video"), exist_ok=True)
    os.makedirs(pjoin(a.out, "json"), exist_ok=True)

    clips = []
    for cid, label in DEMO_CLIPS.items():
        r = ref[cid]
        # keypoint-overlay video, matching the README demo gif
        shutil.copy(pjoin(a.dataset, "pred", "video", f"{cid}.mp4"),
                    pjoin(a.out, "video", f"{cid}.mp4"))
        shutil.copy(pjoin(a.dataset, "pred", "json", f"{cid}.json"),
                    pjoin(a.out, "json", f"{cid}.json"))
        assert cid in gt_csv, f"{cid} has no ground-truth text"
        clips.append({
            "id": cid,
            "label": label,
            "video": f"video/{cid}.mp4",
            "keypoints": f"json/{cid}.json",
            "frames_raw": r["n_frames_raw"],
            "frames_used": r["n_frames_used"],
            "gt": r["gt_raw"],
            # recorded during the evaluation run; verify_parity.py checks these
            "expected": {"adapter": r["pred_adapter"],
                         "no_adapter": r["pred_no_adapter"]},
        })

    OmegaConf.save(OmegaConf.create({"clips": clips}), pjoin(a.out, "clips.yaml"))

    total = sum(os.path.getsize(pjoin(dp, f))
                for dp, _, fs in os.walk(a.out) for f in fs)
    print(f"{len(clips)} clips -> {a.out}  ({total/1e6:.1f} MB)")
    for c in clips:
        print(f"  {c['id']:9s} {c['frames_raw']:3d}f  {c['label']}")


if __name__ == "__main__":
    main()
