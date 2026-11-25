"""Table 12 (App. B.5.1): bootstrap confidence intervals and significance tests for specific
prompting-strategy claims (e.g. "extended thinking reduces failures"), plus a
pooled, cluster-bootstrapped version across all ten models.

Builds a per-model instance x strategy accuracy matrix straight from
`binding_gate.raw_table`, restricted to the known population, then for each
(model, baseline, variant) claim reports the paired failure-rate difference
with a 10,000-resample bootstrap 95% CI and a paired Wilcoxon signed-rank
p-value. The pooled section extends the same idea across all models at once,
cluster-bootstrapping over `dataset|entity` (the same instance recurs once
per model, so those rows are not independent) and Holm-correcting across the
five strategy comparisons.

Needs `scipy` and `statsmodels` (not required by any other script here).

Run from the folder that contains `binding_failures_athletes_rebuttal_temp0.3/`
and `binding_failures_custom_rebuttal_temp0.3/`:

    python src/In_vivo_benchmark/results/significance_bootstrap_ci.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from binding_gate import (
    DATASETS, MODEL_DIRS, MODEL_LABELS, ROWS,
    PAPER_STRATEGIES, PAPER_COLUMNS, known_population, raw_table, warm_cache,
)

OUT_DIR = "rebuttal_analysis_arxiv"
N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000

# (model, baseline strategy, variant strategy) triples to test. Add any other
# pair you want a CI for.
CLAIMS = [(m, "Instant", "Reason") for m in ROWS] + [
    ("Gemini 3.1 Pro", "Instant", "SelfAsk"),
    ("Mistral Large 25.12", "Instant", "SelfAsk"),
    ("Gemini 3.1 Pro", "Instant", "MC"),
    ("Claude Sonnet 4", "Instant", "MC+Reason"),
]


def _percentile_ci(values):
    return [round(100 * v, 1) for v in np.percentile(values, [2.5, 97.5])]


def build_accuracy_matrix():
    """{model_label: DataFrame(index=instance_key, columns=PAPER_COLUMNS)} of
    per-instance accuracy, restricted to the known population."""
    matrices = {}
    for model_dir in MODEL_DIRS:
        label = MODEL_LABELS[model_dir]
        parts = []
        for tag in DATASETS:
            entity_table, _ = raw_table(tag, model_dir)
            population = known_population(tag, model_dir, 0.6)
            if entity_table.empty or not population:
                continue
            columns_present = [d for d in PAPER_STRATEGIES.values() if d in entity_table.columns]
            sub = entity_table.loc[entity_table.index.isin(population), columns_present].rename(
                columns={directory: strategy for strategy, directory in PAPER_STRATEGIES.items()})
            sub.index = [f"{tag}|{model_dir}|{entity}" for entity in sub.index]
            parts.append(sub)
        matrices[label] = pd.concat(parts) if parts else pd.DataFrame(columns=PAPER_COLUMNS)
    return matrices


def per_claim_ci(matrices, rng):
    """One bootstrap CI + Wilcoxon p-value per (model, baseline, variant)."""
    rows, skipped = [], []
    for model, baseline, variant in CLAIMS:
        if model not in matrices or not {baseline, variant} <= set(matrices[model].columns):
            skipped.append((model, baseline, variant))
            continue
        paired = (1 - matrices[model][[baseline, variant]]).dropna()
        if len(paired) < 2:
            skipped.append((model, baseline, variant))
            continue
        boot = paired.values[rng.integers(0, len(paired), (N_BOOTSTRAP, len(paired)))].mean(axis=1)
        delta = paired[variant] - paired[baseline]
        p_value = wilcoxon(delta, zero_method="wilcox").pvalue if delta.any() else 1.0
        rows.append({
            "model": model, "n_paired": len(paired), "baseline": baseline,
            "base_fail%": round(100 * paired[baseline].mean(), 1), "base_CI95": _percentile_ci(boot[:, 0]),
            "variant": variant, "var_fail%": round(100 * paired[variant].mean(), 1), "var_CI95": _percentile_ci(boot[:, 1]),
            "delta_pp": round(100 * delta.mean(), 1), "delta_CI95": _percentile_ci(boot[:, 1] - boot[:, 0]),
            "p_wilcoxon": round(p_value, 5), "sig_p<0.05": p_value < 0.05,
        })
    return pd.DataFrame(rows), skipped


def pooled_cluster_bootstrap(matrices, rng, baseline="Instant"):
    """Instant vs. every other strategy, pooled over all models, cluster-
    bootstrapped over dataset|entity (see module docstring)."""
    pooled = pd.concat([m.assign(model=label) for label, m in matrices.items() if len(m)])
    pooled["cluster"] = [k.split("|")[0] + "|" + k.split("|")[2] for k in pooled.index]

    rows = []
    for variant in [c for c in PAPER_COLUMNS if c != baseline]:
        paired = pooled.dropna(subset=[baseline, variant])
        if len(paired) < 5:
            continue
        deltas = np.c_[1 - paired[baseline], 1 - paired[variant], paired[baseline] - paired[variant]]
        clusters, cluster_ix = np.unique(paired.cluster.values, return_inverse=True)
        cluster_sums = np.zeros((len(clusters), 3))
        np.add.at(cluster_sums, cluster_ix, deltas)
        cluster_sizes = np.bincount(cluster_ix, minlength=len(clusters))

        draws = rng.integers(0, len(clusters), (N_BOOTSTRAP, len(clusters)))
        boot = np.concatenate([
            cluster_sums[draws[i:i + 1000]].sum(1) / cluster_sizes[draws[i:i + 1000]].sum(1)[:, None]
            for i in range(0, N_BOOTSTRAP, 1000)])
        observed = deltas[:, 2].mean()
        permutations = rng.choice([-1., 1.], (N_PERMUTATIONS, len(clusters))) @ cluster_sums[:, 2] / len(paired)
        per_model = paired.groupby("model")[[baseline, variant]].mean()
        per_model_delta = per_model[baseline] - per_model[variant]

        rows.append({
            "variant": variant, "n_pairs": len(paired), "n_models": paired.model.nunique(),
            "n_clusters": len(clusters),
            f"{baseline}_fail%": round(100 * deltas[:, 0].mean(), 1), f"{baseline}_CI95": _percentile_ci(boot[:, 0]),
            "var_fail%": round(100 * deltas[:, 1].mean(), 1), "var_CI95": _percentile_ci(boot[:, 1]),
            "delta_pp": round(100 * observed, 1), "delta_CI95": _percentile_ci(boot[:, 2]),
            "p_perm": round((1 + (np.abs(permutations) >= abs(observed)).sum()) / (1 + N_PERMUTATIONS), 5),
            "delta_model_avg_pp": round(100 * per_model_delta.mean(), 1),
            "models_worse/better": f"{(per_model_delta > 0).sum()}/{(per_model_delta < 0).sum()}",
        })

    table = pd.DataFrame(rows)
    if len(table):
        table["p_perm_holm"] = multipletests(table.p_perm, method="holm")[1].round(5)
        table["sig_holm"] = table.p_perm_holm < 0.05
    return table


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    warm_cache()
    matrices = build_accuracy_matrix()
    print(f"accuracy matrix built: {sum(len(m) for m in matrices.values())} instances across {len(ROWS)} models")

    rng = np.random.default_rng(0)
    claims, skipped = per_claim_ci(matrices, rng)
    print("\n" + "=" * 118)
    print("  per-claim bootstrap CI + Wilcoxon p")
    print("=" * 118)
    print(claims.to_string())
    if skipped:
        print("skipped (missing column / too few paired instances):", skipped)
    claims.to_csv(f"{OUT_DIR}/significance_per_claim_ci.csv", index=False)

    rng = np.random.default_rng(0)
    pooled = pooled_cluster_bootstrap(matrices, rng)
    print("\n" + "=" * 118)
    print("  pooled cluster-bootstrap: Instant vs. every other strategy, across all models")
    print("=" * 118)
    print(pooled.to_string())
    pooled.to_csv(f"{OUT_DIR}/significance_pooled_instant_vs_all.csv", index=False)

    print(f"\nsaved -> {OUT_DIR}/significance_per_claim_ci.csv, {OUT_DIR}/significance_pooled_instant_vs_all.csv")


if __name__ == "__main__":
    main()
