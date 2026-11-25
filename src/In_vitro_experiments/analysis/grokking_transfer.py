"""
Fig. 5: extended training (left) and transfer learning (right); Fig. 14: extended training on two narratives.

    python src/In_vitro_experiments/analysis/grokking_transfer.py --out figures/

Reads data/results/grokking_transfer.json (the scores reported in the paper, not the training runs; pass
--scores with your own runs' scores in the same format to plot those instead), which holds the per-run scores behind the figure:
  grokking           Llama-3-8B, 100-event narrative: SEQ / MEQ / CommonsenseQA accuracy vs. epochs
                     (3 runs to 50 epochs; run 1 continued to 500, 3000, 9000 epochs)
  transfer_learning  Mistral-7B: per-bin accuracy on a held-out narrative after training on
                     N in {0, 4, 7, 10, 20, 100} disjoint narratives, per event size and universe
  two_narratives     Llama-3-8B, two 100-event narratives, trained on SEQ1 + SEQ2 + MEQ2 and
                     evaluated on narrative 1: SEQ1 / MEQ1 / CommonsenseQA vs. epochs (one run, up to 500)
The runs themselves come from step_2_sequential_SEQ.py (more epochs) and
step_2_trainsfer_learning_exp.py; MEQ = count-weighted mean over the bins 2, 3-5, 6+.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import REPO_ROOT

DATA = REPO_ROOT / "data" / "results" / "grokking_transfer.json"
UNIVERSES = ["default", "scifi", "news"]


def _style(plt, font_size):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": font_size, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})


def plot_grokking(data, out_pdf, font_size=16):
    import matplotlib.pyplot as plt

    _style(plt, font_size)
    ext = data["run1_extended"]
    epochs = data["epochs"] + [int(e) for e in ext]
    pos = list(range(len(epochs)))
    n = len(data["epochs"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for key, label, color, marker in [("SEQ", "SEQ", "#F2D338", "o"), ("MEQ", "MEQ", "#7B3FF2", "s"),
                                      ("CommonsenseQA", "Commonsense", "#00A087", "^")]:
        for run in data[key][1:]:
            ax.plot(pos[:n], run, color=color, alpha=0.25, linewidth=1.5, marker=marker, markersize=4)
        ax.plot(pos, data[key][0] + [ext[e][key] for e in ext], color=color, linewidth=2.5, marker=marker,
                markersize=6, label=label, markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xlabel("Training(SEQ) epochs")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(-2.5, 100)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(e) for e in data["epochs"]] + [rf"$\mathbf{{{e}}}$" for e in ext])
    ax.set_xlim(-0.3, len(pos) - 0.7)
    ax.tick_params(labelsize=font_size - 2.5, length=5, width=1.5)
    for a, b in zip(pos[n - 1:-1], pos[n:]):
        ax.text((a + b) / 2, -7, "⋯", fontsize=12, ha="center", va="center", color="gray")
    _finish(ax, plt, font_size, out_pdf)


def plot_two_narratives(data, out_pdf, font_size=11):
    import matplotlib.pyplot as plt

    _style(plt, font_size)
    pos = list(range(len(data["epochs"])))
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color, marker in [("SEQ", "SEQ1", "#F2D338", "o"), ("MEQ", "MEQ1", "#7B3FF2", "s"),
                                      ("CommonsenseQA", "Commonsense", "#00A087", "^")]:
        ax.plot(pos, data[key], color=color, linewidth=2.5, marker=marker, markersize=6, label=label,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xlabel("Training(SEQ1 + SEQ2 + MEQ2) epochs")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(-3, 105)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(e) for e in data["epochs"]])
    ax.set_xlim(-0.3, len(pos) - 0.7)
    ax.tick_params(labelsize=font_size, length=5, width=1.5)
    ax.text(pos[-1] - 0.5, -8, "⋯", fontsize=18, ha="center", va="center", color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5, color="gray", axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=font_size - 1, loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=600, facecolor="white")
    plt.close()
    print(f"saved {out_pdf}")


def meq(bins):
    meq_bins = [b for k, b in bins.items() if k != "1"]
    den = sum(b["count"] for b in meq_bins)
    return sum(b["score"] * b["count"] for b in meq_bins) / den if den else np.nan


def plot_transfer(data, out_pdf, font_size=20):
    import matplotlib.pyplot as plt

    _style(plt, font_size)
    res = data["results"]
    levels = sorted(res, key=int)
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for event_size, color, run_style, avg_style, marker in [("10", "#4F6BFF", "--", "--", "s"), ("30", "#5A1FD6", ":", "-", "o")]:
        for universe in UNIVERSES:
            ys = [meq(res[s][event_size][universe]) if universe in res[s].get(event_size, {}) else np.nan for s in levels]
            if not np.all(np.isnan(ys)):
                ax.plot(x, ys, color=color, alpha=0.2, linestyle=run_style, linewidth=1.5, marker="o" if event_size == "10" else "s", markersize=4)
        avg = []
        for s in levels:
            universes = res[s].get(event_size, {})
            bins = [b for u in universes.values() for k, b in u.items() if k != "1"]
            den = sum(b["count"] for b in bins)
            avg.append(sum(b["score"] * b["count"] for b in bins) / den if den else np.nan)
        ax.plot(x, avg, color=color, linestyle=avg_style, linewidth=2.8, marker=marker, markersize=6,
                label=f"MEQ (event size = {event_size})", markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5)
    ax.set_xlabel("Supervision level")
    ax.set_ylabel("Correct Answer (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.tick_params(labelsize=font_size - 2.5, length=5, width=1.5)
    ax.set_ylim(0, 100)
    _finish(ax, plt, font_size, out_pdf)


def _finish(ax, plt, font_size, out_pdf):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, linewidth=0.5, color="gray", axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=font_size - 3.5, loc="upper left", bbox_to_anchor=(0.0, 1.40), frameon=False)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", dpi=600, facecolor="white")
    plt.close()
    print(f"saved {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="folder for the PDFs")
    ap.add_argument("--scores", default=str(DATA), help="scores to plot (default: the paper's, grokking_transfer.json)")
    args = ap.parse_args()
    data = json.loads(Path(args.scores).read_text())
    out = Path(args.out)
    g = data["grokking"]
    print(f"Grokking (run 1): MEQ {g['MEQ'][0][-1]:.1f}% at 50 ep -> " +
          ", ".join(f"{v['MEQ']:.1f}% at {e} ep" for e, v in g["run1_extended"].items()) +
          f"; CommonsenseQA {g['CommonsenseQA'][0][0]:.0f}% -> {g['run1_extended']['9000']['CommonsenseQA']:.0f}%")
    plot_grokking(g, out / "grokking_9k.pdf")
    plot_transfer(data["transfer_learning"], out / "transfer_learning.pdf")
    t = data["two_narratives"]
    print(f"Two narratives: MEQ1 {t['MEQ'][-2]:.1f}% at {t['epochs'][-2]} ep -> {t['MEQ'][-1]:.1f}% at {t['epochs'][-1]} ep")
    plot_two_narratives(t, out / "two_narratives.pdf")


if __name__ == "__main__":
    main()
