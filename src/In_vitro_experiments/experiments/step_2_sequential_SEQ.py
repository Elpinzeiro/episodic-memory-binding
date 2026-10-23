#!/usr/bin/env python3
"""
Sequential-SEQ training: the hyperparameter grid search (App. D.2, Tab. 18) and the
sequential-training baseline of the RSS comparison (Sec. 5, Fig. 7).

The model reads the narrative one chapter at a time: after chapter c it is LoRA-fine-tuned on the
single-event questions (SEQ) of chapter c, starting from the adapter of chapter c-1. After the last
chapter it is evaluated on the evaluated SEQ/MEQ set and on CommonsenseQA; answers are judged by
an LLM (default gemini-2.5-flash, key in .env).

Examples
    # Llama-3-8B, 30-event default narrative, Tab. 18 hyperparameters
    python src/In_vitro_experiments/experiments/step_2_sequential_SEQ.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct --events 30 --universe default

    # the grid of App. D.2 for one model/narrative (as in the paper: lr 1e-6..1e-4, epochs 20/40)
    python src/In_vitro_experiments/experiments/step_2_sequential_SEQ.py \
        --model meta-llama/Llama-3.2-3B-Instruct --events 10 --grid --lrs 1e-6 1e-5 1e-4 --epochs 20 40

Outputs go to data/narratives/<universe>/answers/ (one folder per configuration).
Without internet on the GPU node, add --skip-judge: answers are saved, and judge_saved_answers.py judges them
later on a machine with API access (src/In_vitro_experiments/evaluation/judge_saved_answers.py --help).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import NARRATIVES_DIR, ENV_FILE, UNIVERSES, EVENT_COUNTS, load_narrative

from In_vitro_experiments.hyperparams import BEST_HYPERPARAMS, LLAMA_SIZE as SIZE_OF


def train_sequentially(benchmark, model, epochs, lr, bs, judge, ftuning_type="lora", skip_judge=False):
    from In_vitro_experiments.evaluation.evaluation_wrapper import EvaluationWrapper

    nb_chapters = benchmark.nb_chapters()
    for chapter in range(1, nb_chapters + 1):
        params = {
            "kind": "ftuning-manual",
            "ftuning_type": ftuning_type,
            "model_name": model,
            "n_epochs": epochs,
            "max_new_tokens": 4096,
            "max_new_tokens_generation": 1024,
            "sleeping_time": 2,
            "ftuning_input_data_policy": "single",   # SEQ only
            "ftuning_need_upload": False,
            "ftuning_need_actual_tune": False,
            "ftuning_policy": "episodic_chapter",     # one chapter at a time
            "evaluation": "mini",                     # evaluate once, after the last chapter
            "batch_size": bs,
            "learning_rate_multiplier": lr,
            "max_chapter": chapter,
            "previous_model": {"chapter": chapter - 1, "epochs": epochs},
            "policy": "remove_duplicates",
            "max_questions_commonsense": 9000,
            "1_commonsense_test_per_book": True,
            "debug": False,
            "skip_eval": skip_judge,                 # answers are saved; judge them later with judge_saved_answers.py
        }
        print(f"[chapter {chapter}/{nb_chapters}] {model} epochs={epochs} lr={lr} bs={bs}", flush=True)
        EvaluationWrapper(my_benchmark=benchmark, answering_parameters=params,
                          data_folder=NARRATIVES_DIR, env_file=ENV_FILE, judge_model_name=judge)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id (or local path)")
    ap.add_argument("--events", type=int, choices=EVENT_COUNTS, required=True)
    ap.add_argument("--universe", choices=UNIVERSES, default="default")
    ap.add_argument("--size", choices=["3B", "8B", "13B", "70B"],
                    help="Tab. 18 row to use (default: from --model; non-Llama models need it)")
    ap.add_argument("--fft", action="store_true", help="full fine-tuning instead of LoRA")
    ap.add_argument("--grid", action="store_true", help="grid search over --lrs x --epochs x --bss")
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4])
    ap.add_argument("--epochs", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--bss", type=int, nargs="+", default=[16])
    ap.add_argument("--judge", default="gemini-2.5-flash")
    ap.add_argument("--skip-judge", action="store_true", help="train and answer only (no API calls); judge later with judge_saved_answers.py")
    args = ap.parse_args()

    benchmark = load_narrative(args.universe, args.events)
    ftuning_type = "full" if args.fft else "lora"

    if args.grid:
        configs = [{"epochs": e, "lr": lr, "bs": bs} for bs in args.bss for lr in args.lrs for e in args.epochs]
    else:
        size = args.size or SIZE_OF.get(args.model)
        if size is None:
            ap.error(f"no Tab. 18 row known for {args.model}: pass --size (e.g. --size 8B for a 7B model)")
        configs = [BEST_HYPERPARAMS[(args.events, size)]]

    for cfg in configs:
        train_sequentially(benchmark, args.model, cfg["epochs"], cfg["lr"], cfg["bs"], args.judge, ftuning_type, args.skip_judge)


if __name__ == "__main__":
    main()
