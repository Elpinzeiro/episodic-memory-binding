"""Table 11 (App. B.4.5): failure rate and precision under Instant and Reason
prompts, in vivo.

Recall-based scoring (the paper's main metric) cannot be inflated by verbose
over-generation, but it also cannot rule out that successes come from
indiscriminate enumeration. This computes precision on the same trials --
extracted from the raw, unedited response by the same LLM judge, then scored
against the expected set -- as a check on that.

Run from the folder that contains `binding_failures_athletes_rebuttal_temp0.3/`
and `binding_failures_custom_rebuttal_temp0.3/`:

    python src/In_vivo_benchmark/results/table_precision.py
"""

import ast
import json
import os
import re

import numpy as np
import pandas as pd

from binding_gate import (
    BIND_THRESH, DATASETS, MODEL_DIRS, MODEL_LABELS, ROWS,
    PAPER_STRATEGIES, iter_instances, known_population, raw_table, warm_cache,
)

OUT_DIR = "rebuttal_analysis_arxiv"
PRECISION_STRATEGIES = ("Instant", "Reason")
ALLOW_PROSE_LISTS = True  # count "a, b and c" when the response has no JSON list

_FENCED_LIST_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.S)
_BRACKETED_LIST_RE = re.compile(r"\[[^\[\]]*\]", re.S)
_BULLET_LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.{2,80})$", re.M)


def _gold_items(expected):
    if isinstance(expected, list):
        return expected
    for parser in (ast.literal_eval, json.loads):
        try:
            value = parser(str(expected))
            if isinstance(value, list):
                return value
        except Exception:
            pass
    return []


def _true_positives(evaluation, gold):
    """Number of gold items the judge found in the response, from whichever
    field the eval schema for this strategy happens to store it in."""
    for key in ("found_expected", "identified_items", "found", "matched"):
        if isinstance(evaluation.get(key), list):
            return len(evaluation[key])
    if isinstance(evaluation.get("missing"), list) and gold:
        return max(len(gold) - len(evaluation["missing"]), 0)
    if gold and isinstance(evaluation.get("correct"), bool):
        return len(gold) if evaluation["correct"] else 0
    return None


def _predicted_items(response):
    """Items the model's raw response asserts, last list wins (final answer)."""
    if not isinstance(response, str):
        return None
    for pattern in (_FENCED_LIST_RE, _BRACKETED_LIST_RE):
        for candidate in reversed(pattern.findall(response)):
            for parser in (json.loads, ast.literal_eval):
                try:
                    value = parser(candidate)
                    if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                        return [x.strip() for x in value if x.strip()]
                except Exception:
                    pass
    if ALLOW_PROSE_LISTS:
        bullets = _BULLET_LIST_RE.findall(response)
        if len(bullets) >= 2:
            return [b.strip(" .;") for b in bullets]
        match = re.search(r"(?:are|were|is|:)\s+([^.\n]{4,200})$", response.strip(), re.M)
        if match and ("," in match.group(1) or " and " in match.group(1)):
            items = [s.strip(" .;") for s in re.split(r",| and ", match.group(1)) if s.strip()]
            if len(items) >= 2:
                return items
    return None


def collect_trials():
    """One row per trial, Instant/Reason only, restricted to Fig. 2's known
    population."""
    rows = []
    for tag, cfg in DATASETS.items():
        for model_dir in MODEL_DIRS:
            population = known_population(tag, model_dir, 0.6)
            if not population:
                continue
            for fail_dir in iter_instances(tag, model_dir):
                if fail_dir.name not in population:
                    continue
                for strategy in PRECISION_STRATEGIES:
                    eval_dir = fail_dir / f"eval_{PAPER_STRATEGIES[strategy]}"
                    if not eval_dir.is_dir():
                        continue
                    for f in sorted(eval_dir.glob("eval_*.json")):
                        if f.name.startswith("_debug"):
                            continue
                        try:
                            data = json.load(open(f))
                        except Exception:
                            continue
                        evaluation = data.get("evaluation", data)
                        evaluation = evaluation if isinstance(evaluation, dict) else {}
                        gold = _gold_items(data.get("expected"))
                        true_pos = _true_positives(evaluation, gold)
                        predicted = _predicted_items(data.get("response"))
                        rows.append({"model": MODEL_LABELS[model_dir], "strategy": strategy,
                                     "tp": true_pos, "n_gold": len(gold),
                                     "n_pred": len(predicted) if predicted else None})
    return pd.DataFrame(rows)


def failure_rate_table():
    """Instant and Reason failure rate per model -- same population and rule
    as Fig. 2, so these always match its Instant/Reason columns exactly."""
    fail = {s: pd.Series(0, index=ROWS, dtype=float) for s in PRECISION_STRATEGIES}
    total = {s: pd.Series(0, index=ROWS, dtype=float) for s in PRECISION_STRATEGIES}
    for tag in DATASETS:
        for model_dir in MODEL_DIRS:
            label = MODEL_LABELS[model_dir]
            entity_table, _ = raw_table(tag, model_dir)
            population = known_population(tag, model_dir, 0.6)
            if entity_table.empty or not population:
                continue
            sub = entity_table.loc[entity_table.index.isin(population)]
            for strategy in PRECISION_STRATEGIES:
                binding_dir = PAPER_STRATEGIES[strategy]
                if binding_dir not in sub.columns:
                    continue
                col = sub[binding_dir].dropna()
                fail[strategy][label] += int((col < BIND_THRESH).sum())
                total[strategy][label] += len(col)
    return fail, total


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    warm_cache()
    trials = collect_trials()
    trials.to_csv(f"{OUT_DIR}/table9_precision_recall_trials.csv", index=False)

    if trials.empty:
        print("no data (both dataset base_dirs missing?)")
        return

    trials["precision"] = np.where(
        trials.tp.notna() & trials.n_pred.notna() & (trials.n_pred > 0),
        np.minimum(trials.tp, trials.n_pred) / trials.n_pred, np.nan)

    mean_precision = (trials.dropna(subset=["precision"])
                      .groupby(["model", "strategy"])["precision"].mean()
                      .unstack("strategy").reindex(ROWS).reindex(columns=PRECISION_STRATEGIES))

    fail, total = failure_rate_table()

    table = pd.DataFrame({
        "Failure Instant": (fail["Instant"] / total["Instant"].replace(0, np.nan) * 100).round(1),
        "Failure Reason": (fail["Reason"] / total["Reason"].replace(0, np.nan) * 100).round(1),
        "Precision Instant": mean_precision["Instant"].mul(100).round().astype("Int64"),
        "Precision Reason": mean_precision["Reason"].mul(100).round().astype("Int64"),
    }, index=ROWS)

    print("=" * 100)
    print("  TABLE 9 -- failure rate and precision (%) under Instant and Reason prompts")
    print("=" * 100)
    print(table.to_string())
    table.to_csv(f"{OUT_DIR}/table9_summary.csv")
    print(f"\nsaved -> {OUT_DIR}/table9_precision_recall_trials.csv, {OUT_DIR}/table9_summary.csv")


if __name__ == "__main__":
    main()
