"""Table 7 (App. B.4.1): failure rate under the lenient (3/5) vs. strict (5/5)
known-facts threshold, and the per-failure audit of Table 9 (App. B.4.3).

Salience check: if binding failures reflected marginal, weakly-known facts,
tightening the known-facts gate should shrink them. This recomputes every
Instant/Reason/SelfAsk failure rate at known@3, known@4 and known@5 (nested
subsets of the same population, via `binding_gate.known_population` at
thresholds 0.6/0.8/1.0) and reports the Spearman rank correlation between the
lenient and strict rates across models -- close to 1 means model rankings are
preserved, which is the opposite of what a salience account predicts.

Run from the folder that contains `binding_failures_athletes_rebuttal_temp0.3/`
and `binding_failures_custom_rebuttal_temp0.3/`:

    python src/In_vivo_benchmark/results/table_threshold_sweep.py
"""

import os

import numpy as np
import pandas as pd

from binding_gate import (
    BIND_THRESH, DATASETS, MODEL_DIRS, MODEL_LABELS, ROWS,
    PAPER_STRATEGIES, known_population, raw_table, warm_cache,
)

OUT_DIR = "rebuttal_analysis_arxiv"
THRESHOLD_STRATEGIES = ("Instant", "Reason", "SelfAsk")


def _rankdata(values):
    """Average ranks, ties included -- avoids a scipy dependency for one call."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    n = len(x)
    mean_x, mean_y = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    std_x = sum((rx[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    std_y = sum((ry[i] - mean_y) ** 2 for i in range(n)) ** 0.5
    return cov / (std_x * std_y)


def threshold_sweep(strategy):
    """N and failure rate at known@3 / known@4 / known@5 for one strategy,
    one row per model."""
    binding_dir = PAPER_STRATEGIES[strategy]
    table = pd.DataFrame(index=ROWS)
    for tier, thresh in ((3, 0.6), (4, 0.8), (5, 1.0)):
        n_col = pd.Series(0, index=ROWS)
        fail_col = pd.Series(0, index=ROWS)
        for tag in DATASETS:
            for model_dir in MODEL_DIRS:
                label = MODEL_LABELS[model_dir]
                entity_table, _ = raw_table(tag, model_dir)
                if entity_table.empty or binding_dir not in entity_table.columns:
                    continue
                population = known_population(tag, model_dir, thresh)
                accuracies = entity_table.loc[entity_table.index.isin(population), binding_dir].dropna()
                n_col[label] += len(accuracies)
                fail_col[label] += int((accuracies < BIND_THRESH).sum())
        table[f"N@{tier}"] = n_col
        table[f"rate@{tier}_%"] = (fail_col / n_col.replace(0, np.nan) * 100).round(1)
    return table


def audit_failures(model_label, strategy="Instant"):
    """Per-failure audit: of a model's failures on one strategy, which ones
    have every verification slot at 5/5, and which failed all 5/5 trials.

    Both fall straight out of the cached raw_table: "all slots 5/5" is exactly
    known_population(..., fact_know_thresh=1.0) membership, and "failed 0/5"
    is exactly an Instant/Reason/... accuracy of 0.0.
    """
    model_dir = next(m for m, label in MODEL_LABELS.items() if label == model_label)
    binding_dir = PAPER_STRATEGIES[strategy]
    rows = []
    for tag in DATASETS:
        entity_table, _ = raw_table(tag, model_dir)
        if entity_table.empty or binding_dir not in entity_table.columns:
            continue
        known_at_3 = known_population(tag, model_dir, 0.6)
        known_at_5 = known_population(tag, model_dir, 1.0)
        population = entity_table.loc[entity_table.index.isin(known_at_3)]
        failing = population[population[binding_dir] < BIND_THRESH]
        for name, row in failing.iterrows():
            rows.append({"entity": name, "acc": row[binding_dir],
                         "all_slots_5of5": name in known_at_5,
                         "failed_0of5": row[binding_dir] == 0.0})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    warm_cache()
    sweep_by_strategy = {}
    print("=" * 118)
    print("  TABLE 7 -- failure rate (%) as the known-fact threshold tightens (App. B.4.1)")
    print("=" * 118)
    for strategy in THRESHOLD_STRATEGIES:
        sweep = threshold_sweep(strategy)
        sweep_by_strategy[strategy] = sweep
        print(f"\n--- {strategy} ---")
        print(sweep[["N@3", "rate@3_%", "N@4", "rate@4_%", "N@5", "rate@5_%"]].to_string())

        paired = sweep[["rate@3_%", "rate@5_%"]].dropna()
        if len(paired) >= 3:
            rho = spearman(paired["rate@3_%"].tolist(), paired["rate@5_%"].tolist())
            print(f"  Spearman rho(@3, @5) across models = {rho:.2f}  (n={len(paired)})"
                  "  <- model ordering preserved if close to 1")

    pd.concat(sweep_by_strategy, axis=1).to_csv(f"{OUT_DIR}/table7_threshold_sweep.csv")
    print(f"\nsaved -> {OUT_DIR}/table7_threshold_sweep.csv")

    print("\n" + "=" * 100)
    print("  Per-failure audit (Claude Opus 4.6, Instant)")
    print("=" * 100)
    audit = audit_failures("Claude Opus 4.6", "Instant")
    n = len(audit)
    if n == 0:
        print("  no failures found (dataset missing, or Opus 4.6 has 0 Instant failures here)")
    else:
        print(f"  {n} failures")
        print(f"  every verification slot at 5/5 : {int(audit.all_slots_5of5.sum())}/{n}")
        print(f"  failed all 5/5 binding trials  : {int(audit.failed_0of5.sum())}/{n}")
        print(audit.to_string(index=False))
        audit.to_csv(f"{OUT_DIR}/table7_audit_claude_opus_4.6_instant.csv", index=False)
        print(f"\nsaved -> {OUT_DIR}/table7_audit_claude_opus_4.6_instant.csv")


if __name__ == "__main__":
    main()
