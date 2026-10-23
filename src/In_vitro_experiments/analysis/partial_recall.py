"""
Fig. 9: MEQ partial recall, i.e. how many of the expected events the model retrieves per question.

    python src/In_vitro_experiments/analysis/partial_recall.py --runs runs.json --out figures/

Needs the evaluation of training runs (step_2_train_on_SEQ_and_MEQ.py --policy single; paper: Mistral-7B
and Qwen2.5 on scifi/10, default/30, news/100 events of the Sonnet-4.6 narratives). runs.json holds, per run,
the column matching_groundtruth_items_score of the run's evaluation dataframe (one list per question,
one {item: 0 or 1} per expected event):

    {"runs": [{"model": "Mistral-7B", "universe": "scifi", "events": 10,
               "recall": [[{"Anna": 1}, {"Liam": 0}], ...]}, ...]}

Rows are models, columns are runs in the order given. Writes recall_distribution_grid.pdf.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _color(plt, frac):
    return plt.cm.RdYlGn(frac * 0.8 + 0.1)


def plot_run(ax, series, title, show_ylabel):
    """Per bin of expected events (2, 3, 4, >=5): share of questions at each recall level, and mean recall."""
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in series:
        if len(row) > 1:  # MEQ only
            found = sum(1 for d in row if list(d.values())[0] == 1)
            groups[min(len(row), 5)].append(found / len(row))
    bins = sorted(groups)
    width = 0.7 / 6
    for i, b in enumerate(bins):
        fracs = groups[b]
        levels = defaultdict(int)
        for f in fracs:
            levels[f if b < 5 else round(f * 5) / 5] += 1
        mean = sum(fracs) / len(fracs)
        keys = sorted(levels)
        offsets = np.linspace(-width * (len(keys) - 1) / 2, width * (len(keys) - 1) / 2, len(keys))
        ax.bar(i, mean * 100, width=width * len(keys), color=_color(plt, mean), alpha=0.2, edgecolor="none", zorder=1)
        ax.hlines(mean * 100, i - width * len(keys) / 2, i + width * len(keys) / 2, colors="#555555",
                  linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
        for off, f in zip(offsets, keys):
            pct = levels[f] / len(fracs) * 100
            ax.bar(i + off, pct, width=width * 0.85, color=_color(plt, f), edgecolor="white", linewidth=0.3, zorder=3)
            label_y = max(pct, mean * 100) + 6 if abs(pct - mean * 100) < 8 else pct + 1.5
            ax.text(i + off, label_y, f"{int(round(f * min(b, 5)))}/{min(b, 5)}", ha="center", va="bottom",
                    fontsize=5, color="#333333", zorder=4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    for i, b in enumerate(bins):
        mean = sum(groups[b]) / len(groups[b])
        ax.text(i, ax.get_ylim()[1] * 0.92, f"μ={mean:.0%}", ha="center", va="bottom", fontsize=8,
                color="#555555", fontstyle="italic", zorder=5)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{b if b < 5 else '≥5'}\n(n={len(groups[b])})" for b in bins], fontsize=7)
    if show_ylabel:
        ax.set_ylabel("Frequency (%)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.grid(True, linestyle="-", alpha=0.15, color="gray", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="JSON with the per-question recall of each run (see above)")
    ap.add_argument("--out", default=".", help="folder for the PDF")
    args = ap.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    runs = json.loads(Path(args.runs).read_text())["runs"]
    models = list(dict.fromkeys(r["model"] for r in runs))
    rows = [[r for r in runs if r["model"] == m] for m in models]
    n_cols = max(len(r) for r in rows)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.5 * n_cols, 3.0 * len(rows)), squeeze=False,
                             gridspec_kw={"wspace": 0.25, "hspace": 0.4})
    for i, row in enumerate(rows):
        for j, run in enumerate(row):
            title = f"{run['universe']} · {run['events']} events" if i == 0 else ""
            plot_run(axes[i, j], run["recall"], title, show_ylabel=(j == 0))
        axes[i, 0].annotate(models[i], xy=(-0.35, 0.5), xycoords="axes fraction", fontsize=10,
                            fontweight="bold", ha="right", va="center", rotation=90)
    fig.text(0.5, 0.02, "Number of events to retrieve", ha="center", va="center", fontsize=9)
    fig.legend(handles=[Patch(facecolor=_color(plt, f), label=l) for f, l in ((0, "0%"), (0.5, "50%"), (1, "100%"))]
               + [Patch(facecolor="gray", alpha=0.25, label="μ (avg)")],
               title="Recall", loc="upper right", bbox_to_anchor=(0.99, 0.98), frameon=False, fontsize=10)
    fig.suptitle("MEQ Partial Recall", fontsize=11, fontweight="bold", y=0.98)
    out = Path(args.out) / "recall_distribution_grid.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
