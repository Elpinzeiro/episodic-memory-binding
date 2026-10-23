"""
Fig. 10: correct answers after static multi-epoch Train(SEQ+MEQ), per event size.

    python src/In_vitro_experiments/analysis/train_seq_meq.py --out figures/

Reads data/results/train_seq_meq.json (the scores reported in the paper, not the training runs; pass
--scores with your own runs' scores in the same format to plot those instead): per model and event size (10/30/100), mean and std over the
three universes of the % correct SEQ and MEQ answers. The runs come from
step_2_train_on_SEQ_and_MEQ.py --policy all. Prints the table and writes train_seq_meq_SEQ.pdf and
train_seq_meq_MEQ.pdf (the two panels). Llama3 70B was run once, on the default universe ("n=1").
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import REPO_ROOT

DATA = REPO_ROOT / "data" / "results" / "train_seq_meq.json"
SIZES = ["10", "30", "100"]


def plot_panel(results, metric, out_pdf, show_ylabel, colorbar, font_size=13, std_font_size=8):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import PercentFormatter

    models = list(results)
    avg = np.array([[results[m][s][metric][0] for s in SIZES] for m in models], dtype=float)
    std = np.array([[np.nan if results[m][s][metric][1] is None else results[m][s][metric][1] for s in SIZES] for m in models])
    fig, ax = plt.subplots(figsize=(3.5, 4))
    left, bottom, width, height = 0.15, 0.15, 0.75, 0.7
    ax.set_position([left, bottom, width, height])
    norm = Normalize(vmin=0, vmax=100)
    im = ax.imshow(avg, cmap="viridis", norm=norm, interpolation="nearest", aspect="equal")
    ax.set_xticks(np.arange(len(SIZES)))
    ax.set_xticklabels(SIZES, fontsize=font_size)
    ax.set_xlabel("#Events ", fontsize=font_size)
    ax.set_yticks(np.arange(len(models)))
    if show_ylabel:
        ax.set_yticklabels(models, fontsize=font_size, rotation=15, ha="right", va="center")
    else:
        ax.set_yticklabels([])
    if colorbar:
        cbar = plt.colorbar(im, cax=fig.add_axes([left + width + 0.02, bottom, 0.03, height]))
        cbar.formatter = PercentFormatter(xmax=100)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=font_size - 1)
    for i in range(avg.shape[0]):
        for j in range(avg.shape[1]):
            r, g, b, _ = im.cmap(norm(avg[i, j]))
            color = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "black"
            ax.text(j, i - 0.15, f"{int(round(avg[i, j]))}", ha="center", va="center", color=color, fontsize=font_size + 1)
            ax.text(j, i + 0.27, "n=1" if np.isnan(std[i, j]) else f"({std[i, j]:.1f})", ha="center", va="center",
                    color=color, fontsize=std_font_size)
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=0.5, edgecolor="0.85"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.5, 0.95, metric, ha="center", va="top", fontsize=font_size + 5, transform=fig.transFigure)
    plt.savefig(out_pdf, dpi=500, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close()
    print(f"saved {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="folder for the PDFs")
    ap.add_argument("--scores", default=str(DATA), help="scores to plot (default: the paper's, train_seq_meq.json)")
    args = ap.parse_args()
    results = json.loads(Path(args.scores).read_text())["results"]
    out = Path(args.out)
    print(f"{'model':<14}" + "".join(f"{f'{m} {s}':>14}" for m in ("SEQ", "MEQ") for s in SIZES))
    for model, r in results.items():
        cells = [r[s][m] for m in ("SEQ", "MEQ") for s in SIZES]
        print(f"{model:<14}" + "".join(f"{a:>8.0f} ({'n=1' if sd is None else f'{sd:.1f}'})".rjust(14) for a, sd in cells))
    plot_panel(results, "SEQ", out / "train_seq_meq_SEQ.pdf", show_ylabel=True, colorbar=False)
    plot_panel(results, "MEQ", out / "train_seq_meq_MEQ.pdf", show_ylabel=False, colorbar=False)


if __name__ == "__main__":
    main()
