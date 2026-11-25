#!/usr/bin/env python3
"""
Judge saved answers of an in-vitro run, e.g. a run trained with --skip-judge on a GPU node without internet.

A run folder (data/narratives*/<universe>/answers/.../<configuration>/) holds raw_answers/<q>.json and
raw_answers/<q>_question.txt. This script sends the answers of the scored questions (get == 'all') to the
LLM judge (same prompt and rule as the training scripts), writes evaluated_answers/<q>.json next to raw_answers/ (existing files are kept,
so a rerun continues), and prints the score per number of matching events k: % of the evaluated
questions with recall = 1 (SEQ: k = 1, MEQ: k > 1).

    python src/In_vitro_experiments/evaluation/judge_saved_answers.py \
        --run "data/narratives_sonnet46/scifi/answers/.../<configuration>" --universe scifi --events 10 \
        --generator claude-sonnet-4-6 --judge gemini-2.5-flash

The judge key must be in .env (see .env.example). An API error stops the script; answers the judge
returns in a malformed format are scored 0 and counted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import ENV_FILE, FOLDERS, UNIVERSES, EVENT_COUNTS, GENERATOR_MODEL, load_narrative


def k_bin(k):
    return "1" if k == 1 else "2" if k == 2 else "3-5" if k <= 5 else "6+"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder that contains raw_answers/")
    ap.add_argument("--universe", choices=UNIVERSES, required=True)
    ap.add_argument("--events", type=int, choices=EVENT_COUNTS, required=True)
    ap.add_argument("--generator", choices=sorted(FOLDERS), default=GENERATOR_MODEL,
                    help="narrative set of the run (claude-sonnet-4-6 for Qwen/Mistral runs)")
    ap.add_argument("--judge", default="gemini-2.5-flash")
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    from In_vitro_experiments.models.settings_wrapper import SettingsWrapper
    from In_vitro_experiments.models.models_wrapper import ModelsWrapper
    from In_vitro_experiments.evaluation.scoring_answers import evaluate_answer

    run = Path(args.run)
    raw_dir, out_dir = run / "raw_answers", run / "evaluated_answers"
    if not raw_dir.is_dir():
        sys.exit(f"no raw_answers/ folder in {run}")
    out_dir.mkdir(exist_ok=True)

    benchmark = load_narrative(args.universe, args.events, generator=args.generator)
    df_qa = benchmark.get_df_qa().drop_duplicates("question").set_index("question")
    chapters = benchmark.split_chapters

    judge = None
    results = []
    files = sorted(raw_dir.glob("*_question.txt"), key=lambda p: int(p.name.split("_")[0]))
    for i, qfile in enumerate(files):
        q = qfile.name.split("_")[0]
        question = qfile.read_text(encoding="utf-8").strip()
        if question not in df_qa.index:
            sys.exit(f"question {q} of {run} is not in the {args.universe}/{args.events} narrative of {args.generator}: "
                     "check --universe, --events and --generator")
        row = df_qa.loc[question]
        if row["get"] != "all":
            continue  # chronological / latest-event questions are not in the scores
        correct_answer = list(row["correct_answer"])
        eval_file = out_dir / f"{q}.json"
        if not eval_file.is_file():
            if judge is None:
                params = {"kind": "prompting", "model_name": args.judge, "max_new_tokens": 2048}
                judge = ModelsWrapper(args.judge, SettingsWrapper(_env_file=ENV_FILE), params)
            llm_answer = json.loads((raw_dir / f"{q}.json").read_text(encoding="utf-8"))
            # full-event questions are judged against the chapter text, as in generate_evaluation_func
            long_answer = None
            if len(correct_answer) == 1 and str(correct_answer[0]).startswith("Full chapter "):
                long_answer = chapters[int(str(correct_answer[0]).split()[-1])]
            print(f"judge {i + 1}/{len(files)} (question {q}) with {args.judge}", flush=True)
            for attempt in range(args.max_retries + 1):
                try:
                    out = evaluate_answer(llm_answer, correct_answer, row["retrieval_type"], judge, long_answer,
                                          row["get"], max_new_tokens=2048 * 2 ** attempt)
                    break
                except (json.JSONDecodeError, AttributeError, KeyError, ValueError) as e:
                    # the judge answered, but not in the expected format
                    if attempt < args.max_retries:
                        time.sleep(5)
                        continue
                    scores = [{x: 0} for x in correct_answer]
                    out = {"predicted_items": [], "groundtruth_items": correct_answer,
                           "matching_groundtruth_items_score": scores,
                           "explanation": f"Evaluation failed due to error: {e}. Assigned score 0.",
                           "nb_preds": 0, "nb_gt": len(correct_answer), "sum_scores": 0.0,
                           "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
                except Exception as e:
                    raise RuntimeError(f"Judge call failed on question {q} ({args.judge}): {type(e).__name__}: {e}\n"
                                       "Check the judge API key in .env and the network access; rerun to continue.") from e
            eval_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
        ev = json.loads(eval_file.read_text(encoding="utf-8"))
        if row["n_chapters_correct_answer"] > 0:
            results.append((k_bin(int(row["n_chapters_correct_answer"])), (ev.get("recall") or 0) >= 1 - 1e-9))

    failed = sum("Evaluation failed" in json.loads(f.read_text(encoding="utf-8")).get("explanation", "")
                 for f in out_dir.glob("*.json"))
    print(f"\n{len(results)} scored answers judged, evaluations in {out_dir}")
    if failed:
        print(f"WARNING: {failed} answers could not be judged (malformed judge output) and are scored 0; "
              "delete those files in evaluated_answers/ and rerun to judge them again.")
    print("\n| k | #q | % correct (recall = 1) |\n|---|---|---|")
    for b in ["1", "2", "3-5", "6+"]:
        v = [ok for kb, ok in results if kb == b]
        if v:
            print(f"| {b} | {len(v)} | {100 * sum(v) / len(v):.2f} |")


if __name__ == "__main__":
    main()
