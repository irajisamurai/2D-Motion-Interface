"""Create/update the Hugging Face Space from this directory.

The weight bundle is NOT uploaded — the Space pulls it from BUNDLE_REPO at startup.

    PYTHONPATH=. python space/upload_space.py \
        --repo KanameYOkoYAMA/2d-motion-interface \
        --bundle_repo KanameYOkoYAMA/2d-motion-interface --private
"""

import argparse
import os

from huggingface_hub import HfApi

# bundle/ is fetched from BUNDLE_REPO at runtime; model_card.md belongs to that repo
IGNORE = ["bundle/*", "bundle/**", "__pycache__/*", "**/__pycache__/**",
          "*.pyc", "model_card.md", ".git*", "**/.git*"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Space id, e.g. user/name")
    p.add_argument("--folder", default="./space")
    p.add_argument("--bundle_repo", required=True)
    p.add_argument("--sdk", default="gradio")
    # A free account cannot create a Gradio Space on cpu-basic (402 Payment
    # Required); zero-a10g is allowed. The app itself still runs on CPU.
    p.add_argument("--hardware", default="zero-a10g")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    api = HfApi()

    skip_dirs = {"bundle", "__pycache__"}
    files = []
    for dp, dirs, fs in os.walk(a.folder):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in fs:
            if f.endswith(".pyc") or f == "model_card.md":
                continue
            full = os.path.join(dp, f)
            files.append((os.path.relpath(full, a.folder), os.path.getsize(full)))
    files.sort()

    print(f"space       : {a.repo} ({'private' if a.private else 'PUBLIC'}, sdk={a.sdk})")
    print(f"BUNDLE_REPO : {a.bundle_repo}")
    print("upload      :")
    for rel, size in files:
        print(f"    {rel:38s} {size/1e6:7.2f} MB")
    print(f"    {'TOTAL':38s} {sum(s for _, s in files)/1e6:7.2f} MB")
    if a.dry_run:
        print("\ndry run - nothing uploaded")
        return

    # Only pass space_hardware on first creation: re-requesting zero-a10g on an
    # existing Space returns 402 for a free account.
    if not api.repo_exists(a.repo, repo_type="space"):
        api.create_repo(a.repo, repo_type="space", space_sdk=a.sdk,
                        space_hardware=a.hardware, private=a.private)
    else:
        print("space exists - leaving sdk/hardware/visibility untouched")
    api.add_space_variable(a.repo, "BUNDLE_REPO", a.bundle_repo)
    api.upload_folder(repo_id=a.repo, repo_type="space", folder_path=a.folder,
                      ignore_patterns=IGNORE,
                      commit_message="Deploy 2D Motion Interface demo")

    print("\nuploaded:")
    for f in sorted(api.list_repo_files(a.repo, repo_type="space")):
        print(f"    {f}")
    print(f"\nhttps://huggingface.co/spaces/{a.repo}")
    print("\nRemaining manual step: add an HF_TOKEN secret with read access to "
          f"{a.bundle_repo} (it is private).")


if __name__ == "__main__":
    main()
