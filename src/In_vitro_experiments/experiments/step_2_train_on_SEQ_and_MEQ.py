#!/usr/bin/env python3
"""
Static multi-epoch training on a whole narrative (Sec. 4.2).

  --policy single   Train(SEQ): single-event questions only. Fig. 3, and the adapters analysed in
                    Fig. 4 (step_5_logit_lens.py).
  --policy all      Train(SEQ+MEQ): capacity verification, Fig. 10 (App. D).

The model is LoRA-fine-tuned once on all chapters, then evaluated on the evaluated SEQ/MEQ set and
on CommonsenseQA; answers are judged by an LLM (default gemini-2.5-flash, key in .env).
Hyperparameters and the narrative set (Claude 3.5 Sonnet for Llama, Sonnet 4.6 for Qwen/Mistral)
come from hyperparams.py unless given. Fig. 3 averages 3 sizes x 3 universes x 3 runs.

    python src/In_vitro_experiments/experiments/step_2_train_on_SEQ_and_MEQ.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct --universe default --events 30 --policy single

Outputs go to data/narratives*/<universe>/answers/ (one folder per configuration).
Without internet on the GPU node, add --skip-judge: answers are saved, and judge_saved_answers.py judges them
later on a machine with API access (src/In_vitro_experiments/evaluation/judge_saved_answers.py --help).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import ENV_FILE, FOLDERS, UNIVERSES, EVENT_COUNTS, load_narrative
from In_vitro_experiments.hyperparams import hyperparams_for, generator_for


def train_static(benchmark, data_folder, model, policy, epochs, lr, bs, judge, ftuning_type="lora", skip_judge=False):
    from In_vitro_experiments.evaluation.evaluation_wrapper import EvaluationWrapper

    params = {
        "kind": "ftuning-manual",
        "ftuning_type": ftuning_type,
        "model_name": model,
        "n_epochs": epochs,
        "max_new_tokens": 4096,
        "sleeping_time": 2,
        "ftuning_input_data_policy": policy,   # 'single' = SEQ, 'all' = SEQ+MEQ
        "ftuning_need_upload": False,
        "ftuning_need_actual_tune": False,
        "batch_size": bs,
        "learning_rate_multiplier": lr,
        "policy": "remove_duplicates",
        "max_questions_commonsense": 9000,
        "debug": False,
        "not_eval_hall": True,
        "save_strategy": "epoch",
        "skip_eval": skip_judge,             # answers are saved; judge them later with judge_saved_answers.py
    }
    return EvaluationWrapper(my_benchmark=benchmark, answering_parameters=params,
                             data_folder=data_folder, env_file=ENV_FILE, judge_model_name=judge)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model id (or local path)")
    ap.add_argument("--universe", choices=UNIVERSES, default="default")
    ap.add_argument("--events", type=int, choices=EVENT_COUNTS, nargs="+", default=EVENT_COUNTS)
    ap.add_argument("--policy", choices=["single", "all"], nargs="+", default=["single"])
    ap.add_argument("--size", choices=["3B", "8B", "13B", "70B"], help="force a Tab. 18 row")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--fft", action="store_true", help="full fine-tuning instead of LoRA (Llama-3-8B FFT row of Fig. 3)")
    ap.add_argument("--generator", choices=sorted(FOLDERS), help="narrative set (default: by model)")
    ap.add_argument("--judge", default="gemini-2.5-flash")
    ap.add_argument("--skip-judge", action="store_true", help="train and answer only (no API calls); judge later with judge_saved_answers.py")
    args = ap.parse_args()

    generator = args.generator or generator_for(args.model)
    for nb_events in args.events:
        hp = hyperparams_for(args.model, nb_events, args.size) if None in (args.epochs, args.lr, args.bs) else {}
        epochs, lr, bs = args.epochs or hp["epochs"], args.lr or hp["lr"], args.bs or hp["bs"]
        benchmark = load_narrative(args.universe, nb_events, generator=generator)
        for policy in args.policy:
            print(f"[{args.universe}, {nb_events} events, {generator}] {args.model} policy={policy} "
                  f"epochs={epochs} lr={lr} bs={bs}", flush=True)
            train_static(benchmark, FOLDERS[generator], args.model, policy, epochs, lr, bs, args.judge,
                         "full" if args.fft else "lora", args.skip_judge)


if __name__ == "__main__":
    main()
