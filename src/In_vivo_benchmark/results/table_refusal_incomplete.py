"""Table 10 (App. B.4.4): failure rate split into refusal (R) vs. incomplete-
answer (I), six prompting strategies.

A binding failure can be an explicit refusal (the model declines to answer,
or states it lacks reliable information) or a confident-but-incomplete answer
(a subset of the expected items, or the inclusion of a wrong/hallucinated
entity). For every failing instance (`binding_gate.known_population`, same
population as Fig. 2), classify every incorrect trial into one of these two
buckets (plus a few generation/judge-artifact buckets that are measurement
noise rather than a model decision) and label the instance by the majority
vote of its incorrect trials. R% + I% for a cell always equals that cell's
failure rate in Fig. 2, since both are computed on the same population.

Run from the folder that contains `binding_failures_athletes_rebuttal_temp0.3/`
and `binding_failures_custom_rebuttal_temp0.3/`:

    python src/In_vivo_benchmark/results/table_refusal_incomplete.py
"""

import ast
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from binding_gate import (
    BIND_THRESH, DATASETS, MODEL_DIRS, MODEL_LABELS, ROWS,
    PAPER_COLUMNS, PAPER_STRATEGIES, MC_STRATEGIES,
    iter_instances, known_population, parse_expected, read_trials, warm_cache,
)

OUT_DIR = "rebuttal_analysis_arxiv"

APOSTROPHE = r"[’ʼ'`´]"
REFUSAL_RE = re.compile(
    rf"do(?:n{APOSTROPHE}?t| not) have (?:any |enough |sufficient |specific |detailed )*"
    rf"(?:reliable )?(?:information|knowledge|data|details)"
    rf"|no reliable information"
    rf"|i (?:do not|don{APOSTROPHE}?t) know"
    rf"|(?:cannot|can{APOSTROPHE}?t|unable to|not able to) (?:reliably |confidently )*"
    rf"(?:provide|recall|answer|verify|confirm|determine|identify)"
    rf"|(?:need to |should )?consult (?:a |an |the )?(?:gaming |official |external )*"
    rf"(?:database|source|record)"
    rf"|refer to (?:official |external )?(?:sources|records)", re.I)
TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?\]\}\)\"'’]\s*$")

_stop_reason_cache = {}


def _raw_file_of(eval_file):
    path = Path(eval_file)
    parent = path.parent
    return parent.parent / parent.name[len("eval_"):] / path.name[len("eval_"):]


def _stop_reason(eval_file):
    raw_path = _raw_file_of(eval_file)
    if raw_path not in _stop_reason_cache:
        value = None
        if raw_path.exists():
            try:
                data = json.load(open(raw_path))
                value = data.get("stop_reason") or data.get("finish_reason")
            except Exception:
                value = "UNREADABLE"
        _stop_reason_cache[raw_path] = value
    return _stop_reason_cache[raw_path]


def _api_truncated(eval_file):
    return str(_stop_reason(eval_file)).lower() in {"max_tokens", "length", "2", "max_output_tokens"}


def _last_json_list(text):
    for block in reversed(re.findall(r"\[[^\[\]]*\]", text, re.S)):
        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(block)
                if isinstance(value, list):
                    return value
            except Exception:
                pass
    return None


def classify_trial(trial, k_gold):
    """Exhaustive taxonomy for one incorrect trial. Returns (bucket, detail).

    Buckets: correct, incomplete, confident_wrong, abstention (a bucket-2
    refusal), artifact / judge_artifact (generation or judging noise, not a
    model decision -- e.g. truncated output, or the judge returning no
    verdict).
    """
    eval_file = trial["eval_file"]
    response = (trial.get("response") or "").strip()
    found = trial.get("found_expected")
    n_found = len(found) if found is not None else None

    if trial.get("judge_error") or found is None:
        return "judge_artifact", "judge_batch_parse_failure"
    if k_gold == 0:
        return "judge_artifact", "no_gold_items"
    if trial["correct"]:
        return "correct", "correct"
    if n_found is not None and n_found >= k_gold:
        return "judge_artifact", "alias_mismatch_full_recall_marked_wrong"

    if response == "":
        return "artifact", "empty_response"
    if _api_truncated(eval_file) or response.count("[") != response.count("]"):
        return "artifact", "truncated_max_tokens"
    if re.fullmatch(r"[`\s\[\]{}(),.\-:*\"'’]*", response):
        return "artifact", "degenerate_output"
    if re.search(r"follow[- ]?up\s*:", response, re.I) and n_found == 0:
        return "artifact", "selfask_loop_no_answer"

    if n_found and n_found > 0:
        return "incomplete", "incomplete"

    if REFUSAL_RE.search(response):
        return "abstention", "explicit_refusal"
    items = _last_json_list(response)
    if items == []:
        return "abstention", "empty_list"
    if items:
        return "confident_wrong", "wrong_list"
    if not TERMINAL_PUNCTUATION_RE.search(response):
        return "artifact", "truncated_mid_sentence"
    if len(response) < 40:
        return "artifact", "fragment"
    return "confident_wrong", "wrong_prose"


def refusal_or_incomplete(trial, k_gold, is_mc):
    """-> 'correct' | 'refusal' | 'incomplete', collapsing the taxonomy above
    to the two buckets Table 10 reports (everything that isn't an explicit
    refusal counts as incomplete, per the paper's call)."""
    if trial["correct"]:
        return "correct"
    if is_mc:
        response = (trial.get("response") or "").strip()
        return "refusal" if (response and REFUSAL_RE.search(response)) else "incomplete"
    bucket, _ = classify_trial(trial, k_gold)
    return "refusal" if bucket == "abstention" else "incomplete"


def build_instance_labels():
    """One row per (dataset, model, entity, strategy): 'pass' | 'refusal' |
    'incomplete', labelled by the majority vote of a failing instance's
    incorrect trials."""
    records = []
    for tag, cfg in DATASETS.items():
        for model_dir in MODEL_DIRS:
            population = known_population(tag, model_dir, 0.6)
            if not population:
                continue
            for fail_dir in iter_instances(tag, model_dir):
                if fail_dir.name not in population:
                    continue
                for strategy, binding_dir in PAPER_STRATEGIES.items():
                    trials = read_trials(fail_dir, binding_dir)
                    if not trials:
                        continue
                    k_gold = len(parse_expected(trials[0]["expected"]))
                    accuracy = sum(t["correct"] for t in trials) / len(trials)
                    if accuracy >= BIND_THRESH:
                        label = "pass"
                    else:
                        votes = Counter(refusal_or_incomplete(t, k_gold, strategy in MC_STRATEGIES)
                                        for t in trials if not t["correct"])
                        label = "refusal" if votes["refusal"] > votes["incomplete"] else "incomplete"
                    records.append({"model": MODEL_LABELS[model_dir], "strategy": strategy,
                                     "entity": f"{tag}:{fail_dir.name}", "label": label})
    return pd.DataFrame(records)


def format_cell(row):
    total_pct = round((row.R + row.Inc) / row.N * 100) if row.N else 0
    refusal_pct = round(row.R / row.N * 100) if row.N else 0
    refusal_pct = min(refusal_pct, total_pct)
    return f"{total_pct}% ({refusal_pct}R / {total_pct - refusal_pct}I)"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    warm_cache()
    labels = build_instance_labels()
    labels.to_csv(f"{OUT_DIR}/table8_trials.csv", index=False)

    if labels.empty:
        print("no data found (both dataset base_dirs missing?)")
        return

    grouped = (labels.groupby(["model", "strategy"])
               .agg(N=("label", "size"),
                    R=("label", lambda s: (s == "refusal").sum()),
                    Inc=("label", lambda s: (s == "incomplete").sum()))
               .reset_index())
    grouped["cell"] = [format_cell(r) for r in grouped.itertuples()]

    table = grouped.pivot(index="model", columns="strategy", values="cell").reindex(ROWS)[PAPER_COLUMNS]
    n_known = grouped.pivot(index="model", columns="strategy", values="N").reindex(ROWS)["Instant"]
    table.insert(0, "N_Know", n_known.astype("Int64"))

    print("=" * 118)
    print("  TABLE 8 -- failure rate split into refusal (R) / incomplete (I)")
    print("=" * 118)
    print(table.to_string())
    print(f"\nsaved -> {OUT_DIR}/table8_trials.csv")


if __name__ == "__main__":
    main()
