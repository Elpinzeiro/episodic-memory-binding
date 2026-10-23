"""
Fig. 3: correct answers after static multi-epoch Train(SEQ), per number of events to bind.

    python src/In_vitro_experiments/analysis/binding_heatmaps.py --out figures/

Reads data/results/binding_heatmaps.json (the scores reported in the paper, not the training runs; pass
--scores with your own runs' scores in the same format to plot those instead): per model, event size (10/30/100) and universe, the % of
correct answers in each bin of matching events ('1' = SEQ; '2', '3-5', '6+' = MEQ). The runs come
from step_2_train_on_SEQ_and_MEQ.py --policy single. Each cell of the figure is the mean (and
population std) over the event size x universe entries of that model; prints the table and writes
Llama_GPT_family.pdf (left panel) and Qwen_Mistral_family.pdf (right panel); any other panel of --scores
(e.g. "my_runs" from run_to_scores.py) is drawn as <panel>.pdf.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import REPO_ROOT

DATA = REPO_ROOT / "data" / "results" / "binding_heatmaps.json"
BINS = ["1", "2", "3-5", "6+"]


def aggregate(panel):
    """model -> (means, stds) over all event size x universe entries, one value per bin."""
    table = {}
    for model, sizes in panel.items():
        values = {b: [bins[b] for universes in sizes.values() for bins in universes.values() if b in bins] for b in BINS}
        table[model] = ([np.mean(values[b]) if values[b] else np.nan for b in BINS],
                        [np.std(values[b]) if values[b] else np.nan for b in BINS])
    return table


def plot_heatmap(table, out_pdf, colorbar, font_size=16, std_font_size=14):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import PercentFormatter

    avg = np.array([m for m, _ in table.values()], dtype=float)
    std = np.array([s for _, s in table.values()], dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4.2))
    norm = Normalize(vmin=0, vmax=100)
    im = ax.imshow(avg, cmap="viridis", norm=norm)
    ax.set_xticks(np.arange(len(BINS)))
    ax.set_xticklabels(BINS, fontsize=font_size)
    for j, label in {0: "SEQ", 2: "MEQ"}.items():
        ax.text(j, 1.12, label, ha="center", va="top", fontsize=font_size, transform=ax.get_xaxis_transform())
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(list(table), fontsize=font_size - 2, rotation=30, ha="right", va="center")
    for i in range(avg.shape[0]):
        for j in range(avg.shape[1]):
            if np.isnan(avg[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=font_size + 2)
                continue
            r, g, b, _ = im.cmap(norm(avg[i, j]))
            color = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "black"
            ax.text(j, i - 0.20, f"{int(round(avg[i, j]))}", ha="center", va="center", color=color, fontsize=font_size + 1)
            ax.text(j, i + 0.25, f"({std[i, j]:.1f})", ha="center", va="center", color=color, fontsize=std_font_size)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(y=3.5, color="white", linewidth=3.5)
    ax.set_xlabel("#Ev. to bind", fontsize=14)
    if colorbar:
        fig.canvas.draw()
        box = ax.get_position()
        cbar = plt.colorbar(im, cax=fig.add_axes([box.x0 + box.width + 0.02, box.y0, 0.03, box.height]))
        cbar.formatter = PercentFormatter(xmax=100)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=font_size - 1)
    plt.savefig(out_pdf, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"saved {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="folder for the PDFs")
    ap.add_argument("--scores", default=str(DATA), help="scores to plot (default: the paper's, binding_heatmaps.json)")
    args = ap.parse_args()
    data = json.loads(Path(args.scores).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # the paper's two panels; any other panel (e.g. run_to_scores.py's "my_runs") gets <panel>.pdf
    pdfs = {"llama_gpt": ("Llama_GPT_family.pdf", False), "qwen_mistral": ("Qwen_Mistral_family.pdf", True)}
    panels = [p for p in data if p not in ("note", "std_over_runs")]
    for panel in panels:
        pdf, colorbar = pdfs.get(panel, (f"{panel}.pdf", True))
        print(f"\n{panel}")
        table = aggregate(data[panel])
        # a model with one event size x universe entry (e.g. your own run): show the std over runs from run_to_scores.py
        for model, sizes in data.get("std_over_runs", {}).get(panel, {}).items():
            entries = [bins for universes in sizes.values() for bins in universes.values()]
            if len(entries) == 1:
                table[model] = (table[model][0], [entries[0].get(b, np.nan) for b in BINS])
        print(f"{'model':<14}" + "".join(f"{b:>14}" for b in BINS))
        for model, (means, stds) in table.items():
            print(f"{model:<14}" + "".join(f"{m:>8.0f} ({s:.1f})" if not np.isnan(m) else f"{'—':>14}" for m, s in zip(means, stds)))
        plot_heatmap(table, out / pdf, colorbar)


if __name__ == "__main__":
    main()
