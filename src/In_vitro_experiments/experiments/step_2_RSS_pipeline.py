#!/usr/bin/env python3
"""
Recollect-Store-Sequence (RSS) pipeline (Sec. 5, Fig. 7, App. E) on the released 100-event
narrative of $UNIVERSE.

    UNIVERSE=default python src/In_vitro_experiments/experiments/step_2_RSS_pipeline.py
"""
# =======================================================
# REQUIRED CONFIGURATION (IMPORTANT)
# =======================================================
#
# You MUST configure the recollection strategy before running this script.
#
# This script supports ONE active ablation setting at a time:
#
#   recollecting_approach ∈ {
#       "magic_12",
#       "generic_qa",
#       "prompt_with_context"
#   }
#
# -------------------------------------------------------
# STRATEGY DEFINITIONS (Appendix E.1.2)
# -------------------------------------------------------
#
# 1) "magic_12"  → RSS-RICH
#    - Uses 12 handcrafted template questions per event
#    - High-coverage structured recall
#
# 2) "generic_qa" → RSS-SIMPLE
#    - Uses 4 minimal / naive template questions per event
#    - Lightweight baseline retrieval
#
# 3) "prompt_with_context" → RSS-GENERATED
#    - Uses LLM-generated recall questions conditioned on event context
#    - Fully automated, no handcrafted templates
#
# -------------------------------------------------------
# HOW TO SET IT
# -------------------------------------------------------
#
# You MUST pass the strategy through the evaluation parameters:
#
#   answering_params["recollecting_approach"] = "<strategy>"
#
# Example:
#
#   answering_params["recollecting_approach"] = "magic_12"
#
# =======================================================
# OPTIONAL CONFIGURATION: HALLUCINATION FILTER
# =======================================================
#
# You can optionally enable a hallucination filtering step
# on recollected QA pairs.
#
# This is controlled by:
#
#   answering_parameters["clean_recall_from_hall"] = True/False
#
# -------------------------------------------------------
# EFFECT
# -------------------------------------------------------
#
# If enabled (True):
#
#   - Recollection answers are post-processed
#   - Hallucinated or inconsistent answers are removed/corrected
#   - The cleaned answer replaces the original "correct_answer"

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/


# -----------------------------
# Environment Setup
# -----------------------------

def setup_environment():
    """Put src/ on sys.path and pick the GPU from CUDA_VISIBLE_DEVICES (default 0)."""
    from In_vitro_experiments.narratives import REPO_ROOT
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(f"[INFO] Repo: {REPO_ROOT} | CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    return REPO_ROOT

# -----------------------------
# Benchmark Creation
# -----------------------------

def create_benchmarks(project_root):
    """Load the released 100-event narrative (data/narratives) of $UNIVERSE (default: default)."""
    from In_vitro_experiments.narratives import load_narrative, NARRATIVES_DIR, ENV_FILE

    universe = os.environ.get("UNIVERSE", "default")
    print(f"[INFO] Loading narrative: 100 events | universe={universe}")
    return {"benchmark_100": load_narrative(universe, 100)}, NARRATIVES_DIR, ENV_FILE

# -----------------------------
# 8B Model Pipeline
# -----------------------------

def run_8b_model_pipeline(benchmarks, data_folder, env_file):
    """
    Run evaluation pipeline for a single model configuration.
    """

    from In_vitro_experiments.evaluation.evaluation_wrapper import EvaluationWrapper

    print("\n🚀 STARTING MODEL PIPELINE")
    print("=" * 50)

    batch_size = int(os.environ.get("BATCH_SIZE", 16))
    learning_rate = float(os.environ.get("LR", 1e-5))
    target_epochs = int(os.environ.get("EPOCHS", 40))

    model_names = [
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]

    benchmark_list = [
        benchmarks[list(benchmarks.keys())[0]]
    ]

    recollecting_approaches = [
        "generic_and_magic",
        "generic_qa"
    ]

    for my_benchmark in benchmark_list:

        invalid_chapters = set(
            idx + 1 for idx in my_benchmark.invalid_debug_event_idx_func()
        )

        valid_chapters = [
            ch for ch in range(1, my_benchmark.nb_chapters() + 1)
            if ch not in invalid_chapters
        ]

        for model_name in model_names:
            for approach in recollecting_approaches:

                print(
                    f"[INFO] batch_size={batch_size}, lr={learning_rate}, epochs={target_epochs}"
                )

                for max_chapter in valid_chapters:

                    previous_chapter = max(
                        [ch for ch in valid_chapters if ch < max_chapter],
                        default=0
                    )

                    answering_params = {
                        "kind": "ftuning-manual",
                        "model_name": model_name,
                        "n_epochs": target_epochs,
                        "max_new_tokens": 4096,
                        "max_new_tokens_generation": 1024,
                        "sleeping_time": 2,
                        "ftuning_input_data_policy": "single",
                        "ftuning_need_upload": False,
                        "ftuning_need_actual_tune": False,
                        "ftuning_policy": "episodic_chapter_perfect_recoll",
                        "evaluation": "mini",
                        "batch_size": batch_size,
                        "learning_rate_multiplier": learning_rate,
                        "max_chapter": max_chapter,
                        "previous_model": {
                            "chapter": previous_chapter,
                            "epochs": target_epochs
                        },
                        "policy": "remove_duplicates",
                        "max_questions_commonsense": 9000,
                        "1_commonsense_test_per_book": True,
                        "debug": False,
                        "recollecting_approach": approach
                    }

                    print(
                        f"[RUNNING] chapters={my_benchmark.nb_chapters()} "
                        f"| chapter={max_chapter} | approach={approach}"
                    )

                    evaluation_wrapper = EvaluationWrapper(
                        my_benchmark=my_benchmark,
                        answering_parameters=answering_params,
                        data_folder=data_folder,
                        env_file=env_file,
                        judge_model_name=os.environ.get(
                            "JUDGE_MODEL",
                            "gemini-2.5-flash"
                        )
                    )

                    print(
                        f"[DONE] chapter={max_chapter} "
                        f"| approach={approach} "
                        f"| results={len(evaluation_wrapper.df_generated_answers)}"
                    )

    print("\n🎉 PIPELINE COMPLETE")
    print("=" * 50)


# -----------------------------
# Main
# -----------------------------

def main():
    try:
        project_root = setup_environment()

        benchmarks, data_folder, env_file = create_benchmarks(project_root)

        run_8b_model_pipeline(benchmarks, data_folder, env_file)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()