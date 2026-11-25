#!/usr/bin/env python3
"""
Transfer learning (Sec. 4.2.3, Fig. 5 right, App. D): train on N disjoint narratives, test on a
held-out one. The disjoint 10/30-event narratives were generated with Claude Sonnet 4.6.

Usage (Mistral-7B, 80 epochs, lr 1e-5, bs 16):
    python src/In_vitro_experiments/download_transfer_sets.py      # training sets from Hugging Face
    export TRAINING_FILE_PATH=data/transfer_training_sets/10_chapter_books/training_sets_4/training_sets_4_fold1_default_heldout_default_9_seed_90.jsonl
    python src/In_vitro_experiments/experiments/step_2_trainsfer_learning_exp.py
The held-out narrative of the fold (here: default universe, seed 90) is loaded from
data/narratives_transfer/.
"""

import sys
import os
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/


# ============================================================================
# USER INPUT (can be overridden via env var)
# ============================================================================

TRAINING_FILE_PATH = os.environ.get("TRAINING_FILE_PATH", None)

if not TRAINING_FILE_PATH:
    raise ValueError(
        "TRAINING_FILE_PATH must be set via environment variable."
    )


# ============================================================================
# FOLD CONFIG DERIVATION
# ============================================================================

def derive_fold_config(training_file_path: str) -> dict:
    """
    Derive fold configuration from dataset path + filename.
    Fully portable and repo-agnostic.
    """

    path = Path(training_file_path)

    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")

    path_str = str(path)

    # --- nb_events inference ---
    if "30_chapter_books" in path_str:
        nb_events = 30
    elif "10_chapter_books" in path_str:
        nb_events = 10
    else:
        raise ValueError(
            f"Expected '10_chapter_books' or '30_chapter_books' in path: {path_str}"
        )

    filename = path.name

    pattern = (
        r'^training_sets_(\d+)'      # dataset size
        r'_fold(\d+)'                # fold id
        r'_([a-zA-Z]+)'              # fold universe
        r'_heldout_([a-zA-Z]+)'      # held-out universe
        r'_(\d+)'                    # held-out book id
        r'_seed_(\d+)'               # seed
        r'\.jsonl$'
    )

    match = re.match(pattern, filename)
    if not match:
        raise ValueError(
            "Filename does not match expected format."
        )

    training_set_size = int(match.group(1))
    fold_id = int(match.group(2))
    fold_universe = match.group(3)
    held_out_universe = match.group(4)
    held_out_book_id = int(match.group(5))
    held_out_seed = int(match.group(6))

    if fold_universe != held_out_universe:
        raise ValueError("Universe mismatch in filename.")

    training_dir = path.parent
    expected_dir = f"training_sets_{training_set_size}"

    if training_dir.name != expected_dir:
        raise ValueError("Directory does not match expected dataset size.")

    return {
        "fold_id": fold_id,
        "held_out_key": f"{held_out_universe}_{held_out_book_id}",
        "held_out_universe": held_out_universe,
        "held_out_seed": held_out_seed,
        "held_out_book_id": held_out_book_id,
        "training_file": filename,
        "training_dir": training_dir,
        "training_set_size": training_set_size,
        "nb_events": nb_events,
    }


def print_fold_config(cfg, path):
    print("=" * 80)
    print("DERIVED FOLD CONFIG")
    print("=" * 80)
    print(f"Input path        : {path}")
    print(f"Training file     : {cfg['training_file']}")
    print(f"Training dir      : {cfg['training_dir']}")
    print(f"Dataset size      : {cfg['training_set_size']}")
    print(f"nb_events         : {cfg['nb_events']}")
    print(f"Fold id           : {cfg['fold_id']}")
    print(f"Held-out key      : {cfg['held_out_key']}")
    print(f"Held-out universe : {cfg['held_out_universe']}")
    print(f"Held-out seed     : {cfg['held_out_seed']}")
    print("=" * 80)


FOLD_CONFIG = derive_fold_config(TRAINING_FILE_PATH)
print_fold_config(FOLD_CONFIG, TRAINING_FILE_PATH)


# ============================================================================
# MODEL CONFIG (NO LOCAL MACHINE PATHS)
# ============================================================================

MODEL_CONFIGS = {
    "mistral7B": {
        "short_name": "mistral7B",
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",

        # Optional override via env var (recommended for reproducibility)
        "use_local_path": os.environ.get("USE_LOCAL_MODEL_PATH", "0") == "1",
        "local_path": os.environ.get("LOCAL_MODEL_PATH", None),

        "hyperparams": {
            "target_epochs": 80,
            "learning_rate": 1e-5,
            "batch_size": 16,
        }
    }
}

SELECTED_MODEL = os.environ.get("MODEL_NAME", "mistral7B")


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Put src/ on sys.path and pick the GPU from CUDA_VISIBLE_DEVICES (default 0)."""
    from In_vitro_experiments.narratives import REPO_ROOT
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(f"[INFO] Repo: {REPO_ROOT} | CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    return REPO_ROOT

def get_model_path(model_cfg):
    if model_cfg.get("use_local_path") and model_cfg.get("local_path"):
        base = model_cfg["local_path"]

        snapshots_dir = os.path.join(base, "snapshots")
        if os.path.exists(snapshots_dir):
            snaps = sorted(os.listdir(snapshots_dir))
            if snaps:
                return os.path.join(snapshots_dir, snaps[0])

        return base

    return model_cfg["hf_name"]


# ============================================================================
# DATA
# ============================================================================

def load_training_dataset(project_root):
    from In_vitro_experiments.io.io import import_jsonl

    path = Path(FOLD_CONFIG["training_dir"]) / FOLD_CONFIG["training_file"]

    if not path.exists():
        raise FileNotFoundError(path)

    data = import_jsonl(path)
    print(f"Loaded dataset: {len(data)} samples")

    return data


def create_held_out_benchmark(project_root):
    from In_vitro_experiments.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper

    data_folder = Path(project_root) / "data" / "narratives_transfer"
    env_file = Path(project_root) / ".env"

    prompt_parameters = {
        "nb_events": FOLD_CONFIG["nb_events"],
        "name_universe": FOLD_CONFIG["held_out_universe"],
        "name_styles": "concise",
        "seed": FOLD_CONFIG["held_out_seed"],
        "disjoint_book_id": FOLD_CONFIG["held_out_book_id"],
        "n_total_books": 10,
        "distribution_events": {"name": "geometric", "param": 0.3},
    }

    model_parameters = {
        "model_name": os.environ.get("GEN_MODEL", "claude-sonnet-4-6"),
        "max_new_tokens": 4096,
        "itermax": 10,
    }

    print(f"Creating benchmark: {FOLD_CONFIG['held_out_key']}")

    benchmark = BenchmarkGenerationWrapper(
        prompt_parameters,
        model_parameters,
        {"indexing": "default", "nb_summaries": 0},
        data_folder,
        env_file,
        rechecking=False,
    )

    return benchmark, data_folder, env_file


# ============================================================================
# PIPELINE
# ============================================================================

def run_pipeline():
    project_root = setup_environment()

    model_cfg = MODEL_CONFIGS[SELECTED_MODEL]
    model_path = get_model_path(model_cfg)
    hp = model_cfg["hyperparams"]

    print("\n" + "=" * 80)
    print(f"TRANSFER LEARNING - FOLD {FOLD_CONFIG['fold_id']}")
    print("=" * 80)

    print(f"Model: {model_cfg['short_name']}")
    print(f"Dataset size: {FOLD_CONFIG['training_set_size']}")
    print(f"nb_events: {FOLD_CONFIG['nb_events']}")
    print("=" * 80)

    training_dataset = load_training_dataset(project_root)
    benchmark_eval, data_folder, env_file = create_held_out_benchmark(project_root)

    from In_vitro_experiments.evaluation.evaluation_wrapper import EvaluationWrapper

    answering_params = {
        "kind": "ftuning-manual",
        "ftuning_type": "lora",
        "model_name": model_path,
        "n_epochs": hp["target_epochs"],
        "max_new_tokens": 4096,
        "sleeping_time": 2,
        "ftuning_input_data_policy": "custom",  # allows providing a user-defined / pre-built dataset instead of default sampling pipeline  
        "ftuning_need_actual_tune": False,
        "batch_size": hp["batch_size"],
        "learning_rate_multiplier": hp["learning_rate"],
        "policy": "remove_duplicates",
        "max_questions_commonsense": 9000,
        "debug": False,
        "not_eval_hall": True,
        "save_every_n_epochs": 10,
        "n_supervision_books": FOLD_CONFIG.get("n_supervision_books", None),
    }

    evaluation_wrapper = EvaluationWrapper(
        my_benchmark=benchmark_eval,
        answering_parameters=answering_params,
        data_folder=data_folder,
        env_file=env_file,
        judge_model_name=os.environ.get("JUDGE_MODEL", "gemini-2.5-flash"),
        ftuning_input_if_custom=training_dataset,
    )

    print("\n[DONE] Pipeline initialized for fold:", FOLD_CONFIG["fold_id"])


if __name__ == "__main__":
    run_pipeline()