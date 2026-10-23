"""
Fig. 7: correct answers of the RSS conditions (Sec. 5) per number of matching events k.

    python src/In_vitro_experiments/analysis/rss_conditions.py --out figures/

Reads data/results/rss_conditions.json (the scores reported in the paper, not the training runs; pass
--scores with your own runs' scores in the same format to plot those instead): Llama-3-8B on the 30-event narratives, % correct answers per
bin of k for each condition, as the mean over three runs (one per universe) plus the per-run values
(the dots). Prints the table and writes barplot.pdf. Left legend: fully automated conditions;
right legend: diagnostic controls that use the ground truth.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import REPO_ROOT

DATA = REPO_ROOT / "data" / "results" / "rss_conditions.json"
COLORS = {"Sequential-SEQ": "#95A5A6", "Sequential-SEQ + O-LoRA": "#3498DB", "RSS": "#2C3E50",
          "RSS-No-Hallucination": "#E74C3C", "RSS-Aggregated": "#1ABC9C", "Coverage": "#F39C12"}
AUTOMATED = ["Sequential-SEQ", "Sequential-SEQ + O-LoRA", "RSS"]
GROUND_TRUTH = ["RSS-No-Hallucination", "RSS-Aggregated", "Coverage"]


def plot(data, out_pdf, legend_size=7.7, n_size=7):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 2.68))
    x = np.arange(len(data["bins"]))
    width = 0.13
    bars = {}
    for i, (name, cond) in enumerate(data["conditions"].items()):
        bars[name] = ax.bar(x + i * width, cond["mean"], width, color=COLORS[name], edgecolor="black", linewidth=0.5, label=name)
        for run in cond["runs"]:
            ax.scatter(x + i * width, run, color=COLORS[name], s=12, edgecolors="black", linewidth=0.3, alpha=0.8, zorder=10)
    ax.set_xlabel("Number of events mentioned", labelpad=10)
    ax.set_ylabel("Correct Answers (%)")
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(data["bins"])
    ax.set_ylim(0, 110)
    left = ax.legend([bars[m] for m in AUTOMATED], AUTOMATED, loc="upper left", bbox_to_anchor=(0.0, 1.61),
                     fontsize=legend_size, frameon=True, edgecolor="black", fancybox=False)
    ax.add_artist(left)
    ax.legend([bars[m] for m in GROUND_TRUTH], GROUND_TRUTH, loc="upper right", bbox_to_anchor=(1.01, 1.61),
              fontsize=legend_size, frameon=True, edgecolor="black", fancybox=False)
    for i, n in enumerate(data["questions"]):
        ax.annotate(f"n={n}", xy=(i + width * 2.5, 0), xytext=(0, -18), textcoords="offset points",
                    ha="center", va="top", fontsize=n_size, color="#666666")
    plt.subplots_adjust(top=0.62, bottom=0.25)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=1000)
    plt.close()
    print(f"saved {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="folder for the PDF")
    ap.add_argument("--scores", default=str(DATA), help="scores to plot (default: the paper's, rss_conditions.json)")
    args = ap.parse_args()
    data = json.loads(Path(args.scores).read_text())
    print(f"{'condition':<25}" + "".join(f"{f'k={b} (n={n})':>14}" for b, n in zip(data["bins"], data["questions"])))
    for name, cond in data["conditions"].items():
        print(f"{name:<25}" + "".join(f"{v:>14.2f}" for v in cond["mean"]))
    plot(data, Path(args.out) / "barplot.pdf")


if __name__ == "__main__":
    main()
