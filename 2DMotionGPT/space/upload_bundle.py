"""Upload the inference bundle to a Hugging Face model repo.

    PYTHONPATH=. python space/upload_bundle.py --repo KanameYOkoYAMA/2d-motion-interface --private
"""

import argparse
import os

from huggingface_hub import HfApi
from os.path import join as pjoin

# local-only files that should not end up in the model repo
IGNORE = ["export_report.json", "README.md", ".git*", "**/.git*"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--bundle", default="./space/bundle")
    p.add_argument("--card", default="./space/model_card.md")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    api = HfApi()
    files = []
    for dp, _, fs in os.walk(a.bundle):
        for f in fs:
            full = pjoin(dp, f)
            rel = os.path.relpath(full, a.bundle)
            if rel in IGNORE or rel.startswith(".git"):
                continue
            files.append((rel, os.path.getsize(full)))
    files.sort(key=lambda kv: -kv[1])

    print(f"repo    : {a.repo} ({'private' if a.private else 'PUBLIC'})")
    print(f"card    : {a.card}")
    print("upload  :")
    for rel, size in files:
        print(f"    {rel:34s} {size/1e6:8.1f} MB")
    print(f"    {'TOTAL':34s} {sum(s for _, s in files)/1e6:8.1f} MB")
    if a.dry_run:
        print("\ndry run - nothing uploaded")
        return

    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=a.bundle,
                      ignore_patterns=IGNORE,
                      commit_message="Add inference-only bundle (fp32, 1.03 GB)")
    api.upload_file(repo_id=a.repo, repo_type="model", path_or_fileobj=a.card,
                    path_in_repo="README.md", commit_message="Add model card")

    print("\nuploaded:")
    for f in sorted(api.list_repo_files(a.repo)):
        print(f"    {f}")
    print(f"\nhttps://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
