"""Table 8 (App. B.4.2) cardinality check: does the failure rate increase with the
number of episodes a binding query asks for?

Splits each model's failures by cardinality (2, 3 or 4 episodes -- the in-vivo
benchmark's range) both as a share of that model's total failures and as a
rate *within* each bucket (failures at k episodes / instances with k
episodes). The paper's point is that this rate does not climb with k in vivo
(unlike in vitro, where it does by design).

Run from the folder that contains `binding_failures_athletes_rebuttal_temp0.3/`
and `binding_failures_custom_rebuttal_temp0.3/`:

    python src/In_vivo_benchmark/results/table_cardinality.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from binding_gate import (
    BIND_THRESH, DATASETS, MODEL_DIRS, MODEL_LABELS, ROWS,
    PAPER_STRATEGIES, known_population, raw_table, warm_cache,
)

OUT_DIR = "rebuttal_analysis_arxiv"


def n_episodes(tag, model_dir, entity):
    """Number of constituent episodes for one instance: the count of
    single-letter subdirectories (a, b, c, ...), one per fact."""
    fail_dir = Path(DATASETS[tag]["base_dir"]) / model_dir / entity
    if not fail_dir.exists():
        return -1
    return sum(1 for d in fail_dir.iterdir() if d.is_dir() and len(d.name) == 1 and d.name.isalpha())


def cardinality_breakdown(strategy="Instant"):
    binding_dir = PAPER_STRATEGIES[strategy]
    instance_rows, failure_rows = [], []
    for tag in DATASETS:
        for model_dir in MODEL_DIRS:
            label = MODEL_LABELS[model_dir]
            entity_table, _ = raw_table(tag, model_dir)
            population = known_population(tag, model_dir, 0.6)
            if entity_table.empty or not population:
                continue
            for entity in population:
                cardinality = n_episodes(tag, model_dir, entity)
                instance_rows.append({"model": label, "n_events": cardinality})
                if binding_dir not in entity_table.columns or pd.isna(entity_table.at[entity, binding_dir]):
                    continue
                if entity_table.at[entity, binding_dir] < BIND_THRESH:
                    failure_rows.append({"model": label, "n_events": cardinality})

    instances = pd.DataFrame(instance_rows)
    failures = pd.DataFrame(failure_rows)
    if instances.empty:
        return None, None
    cardinalities = sorted(c for c in instances.n_events.unique() if c > 0)

    n_per_model = instances.groupby("model").size().reindex(ROWS)
    fail_by_bucket = (failures.groupby(["model", "n_events"]).size().unstack("n_events")
                      .reindex(ROWS).reindex(columns=cardinalities).fillna(0).astype(int)
                      if len(failures) else pd.DataFrame(0, index=ROWS, columns=cardinalities))
    fail_by_bucket.columns = [f"fail@{c}ev" for c in cardinalities]

    summary = fail_by_bucket.copy()
    summary["fail_total"] = fail_by_bucket.sum(axis=1)
    summary["N"] = n_per_model.astype(int)
    summary["rate_%"] = (summary.fail_total / summary.N.replace(0, np.nan) * 100).round(1)

    n_by_bucket = (instances.groupby(["model", "n_events"]).size().unstack("n_events")
                   .reindex(ROWS).reindex(columns=cardinalities))
    rate_within_bucket = pd.DataFrame(
        (fail_by_bucket.values / n_by_bucket.values * 100),
        index=ROWS, columns=[f"rate@{c}ev_%" for c in cardinalities]).round(1)

    return summary, pd.concat([n_by_bucket.add_prefix("n@").add_suffix("ev"), rate_within_bucket], axis=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    warm_cache()
    for strategy in ("Instant",):
        summary, rate_within_bucket = cardinality_breakdown(strategy)
        print("=" * 100)
        print(f"  {strategy}: binding failures decomposed by number of episodes")
        print("=" * 100)
        if summary is None:
            print("  no data (both dataset base_dirs missing?)")
            continue
        print(summary.to_string())
        print("\n  failure RATE within each episode-count bucket (n@kev = instances with k episodes):")
        print(rate_within_bucket.to_string())
        summary.to_csv(f"{OUT_DIR}/table_cardinality_{strategy}.csv")
        print(f"\nsaved -> {OUT_DIR}/table_cardinality_{strategy}.csv")


if __name__ == "__main__":
    main()
