"""Download the released in-vivo raw outputs (all ten models) from Hugging Face.

This replaces step 1 of the run order (run_binding_benchmark.py, the API-calling
step): it fetches the per-model archives of
the invivo/ folder of https://huggingface.co/datasets/FMantovani/do-llms-bind-episodes, checks them
against the dataset's MANIFEST.json and unpacks them into

    invivo_outputs/athletes/<model>/fail_<item>/...
    invivo_outputs/custom/<model>/fail_<item>/...

binding_gate.py reads ./invivo_outputs automatically when it exists, so every
script in this folder then runs unchanged (from the folder that contains
invivo_outputs/):

    python src/In_vivo_benchmark/results/download_invivo_outputs.py
    python src/In_vivo_benchmark/results/table_threshold_sweep.py
"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "FMantovani/do-llms-bind-episodes"


def main(out_dir, revision):
    snapshot = Path(snapshot_download(REPO_ID, repo_type="dataset", revision=revision,
                                      allow_patterns=["invivo/raw/*"]))
    snapshot = snapshot / "invivo"
    manifest = json.loads((snapshot / "raw" / "MANIFEST.json").read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    for name, entry in sorted(manifest.items()):
        archive = snapshot / name
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(f"checksum mismatch for {name}; delete the Hugging Face cache and retry")
        with tarfile.open(archive) as tar:
            tar.extractall(out_dir, filter="data")
        print(f"  {name}: {entry['files']} files")
    print(f"unpacked {len(manifest)} archives into {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="invivo_outputs", help="target folder (default: ./invivo_outputs)")
    parser.add_argument("--revision", default=None, help="dataset revision (default: latest)")
    args = parser.parse_args()
    main(args.out, args.revision)
