"""Download the transfer-learning training sets (Sec. 4.2.3, Fig. 5 right) from Hugging Face.

    python src/In_vitro_experiments/download_transfer_sets.py

Fetches the invitro/ folder of https://huggingface.co/datasets/FMantovani/do-llms-bind-episodes into
data/transfer_training_sets/{10,30}_chapter_books/training_sets_<N>/*.jsonl, the files that
step_2_trainsfer_learning_exp.py takes as TRAINING_FILE_PATH. Each file holds the chat-format
SEQ+MEQ questions of N disjoint narratives; its name gives the fold and the held-out narrative,
which is released in data/narratives_transfer/.
"""
import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from In_vitro_experiments.narratives import REPO_ROOT

REPO_ID = "FMantovani/do-llms-bind-episodes"
OUT_DIR = REPO_ROOT / "data" / "transfer_training_sets"


def main(revision):
    snapshot = Path(snapshot_download(REPO_ID, repo_type="dataset", revision=revision,
                                      allow_patterns=["invitro/transfer_training_sets/*"]))
    snapshot = snapshot / "invitro"
    files = sorted((snapshot / "transfer_training_sets").rglob("*.jsonl"))
    for src in files:
        dst = OUT_DIR / src.relative_to(snapshot / "transfer_training_sets")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"{len(files)} training sets in {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default=None, help="dataset revision (default: latest)")
    main(ap.parse_args().revision)
