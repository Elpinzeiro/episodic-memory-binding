"""
Tab. 20 and Fig. 4 (= Fig. 12 and 13): logit-lens and forced-prefix results of Llama-3-8B, computed
from the released per-question outputs in data/results/logit_lens/ (written by
experiments/step_5_logit_lens.py).

    python src/In_vitro_experiments/analysis/logit_lens.py            # table only
    python src/In_vitro_experiments/analysis/logit_lens.py --plots    # + top8_retrieval_llama8b.pdf, forced_prefix_llama8b.pdf

Top-8: an entity counts as present if its first token (with or without a leading space) is among
the 8 most probable next tokens at the last prompt position.
Forced prefix: a question answered wrongly counts as recovered if, after forcing the first token or
first word of an omitted entity, any forced continuation contains all ground-truth entities.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import REPO_ROOT, UNIVERSES, EVENT_COUNTS

RESULTS_DIR = REPO_ROOT / "data" / "results" / "logit_lens"
COLORS = {"best": "#0072B2", "mid": "#D95F02", "worst": "#999999"}


def _load(folder):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("q*.json"))]


def _norm(s):
    return re.sub(r"\s+", " ", s or "").lower()


def top8_counts(records):
    counts = [0, 0, 0]  # none, one, both
    for r in records:
        top = set(r["top_ids"])
        found = sum(any(r[t][k] in top for k in ("bare_id", "lead_id")) for t in ("first_token_A", "first_token_B"))
        counts[found] += 1
    return {"both": counts[2], "one": counts[1], "none": counts[0]}


def forced_prefix_counts(records):
    correct = recovered = wrong = 0
    for r in records:
        if not r["missing_entities"]:
            correct += 1
            continue
        gt = r["gt_entities"]
        hit = any(all(_norm(e) in _norm(x[k]["forced_continuation"]) for e in gt)
                  for x in r["forced_prefix_experiments"] for k in ("first_token", "first_word") if k in x)
        recovered += hit
        wrong += not hit
    return {"correct": correct, "recovered": recovered, "wrong": wrong}


def table():
    rows = []
    for nb_events in EVENT_COUNTS:
        for universe in UNIVERSES:
            folder = RESULTS_DIR / f"U{universe}_ev{nb_events}"
            topk, fpc = _load(folder / "topk_first_token"), _load(folder / "forced_prefix_continuation")
            assert [r["question"] for r in topk] == [r["question"] for r in fpc], folder
            rows.append({"events": nb_events, "universe": universe, "N": len(topk),
                         **top8_counts(topk), **forced_prefix_counts(fpc)})
    return rows


def print_table(rows):
    head = f"{'Events':>6} {'Universe':>8} {'N':>4} | {'both':>4} {'one':>4} {'none':>4} {'both%':>6} | {'correct':>7} {'recov.':>6} {'wrong':>5} {'recov.%':>7}"
    print(head + "\n" + "-" * len(head))
    total = {k: 0 for k in ("N", "both", "one", "none", "correct", "recovered", "wrong")}
    for r in rows + [None]:
        if r is None:
            r = {"events": "All", "universe": "", **total}
        else:
            for k in total:
                total[k] += r[k]
        print(f"{r['events']:>6} {r['universe']:>8} {r['N']:>4} | {r['both']:>4} {r['one']:>4} {r['none']:>4} "
              f"{100 * r['both'] / r['N']:>6.1f} | {r['correct']:>7} {r['recovered']:>6} {r['wrong']:>5} "
              f"{100 * r['recovered'] / r['N']:>7.1f}")


def plots(rows, out_dir):
    import matplotlib.pyplot as plt

    labels = [f"{r['universe']}\nev={r['events']}" for r in rows]
    x = range(len(rows))
    edge = dict(edgecolor="black", linewidth=0.5)
    specs = [
        ("top8_retrieval_llama8b.pdf", "Entity retrieval in top-8 first tokens (Llama8B)",
         [("both", "2 entities in top-8", "best"), ("one", "1 entity in top-8", "mid"), ("none", "0 entities in top-8", "worst")]),
        ("forced_prefix_llama8b.pdf", "Forced-prefix binding recovery (Llama8B)",
         [("correct", "Correct", "best"), ("recovered", "Correct with forced prefix", "mid"), ("wrong", "Wrong", "worst")]),
    ]
    for fname, title, stack in specs:
        plt.figure(figsize=(8, 4))
        bottom = [0] * len(rows)
        for key, label, color in stack:
            vals = [r[key] for r in rows]
            plt.bar(x, vals, bottom=bottom, label=label, color=COLORS[color], **edge)
            bottom = [b + v for b, v in zip(bottom, vals)]
        plt.xticks(list(x), labels, fontsize=13)
        plt.ylabel("Count", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / fname, bbox_inches="tight")
        plt.close()
        print(f"saved {out_dir / fname}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--out", default=".", help="folder for the PDFs")
    args = ap.parse_args()
    rows = table()
    print_table(rows)
    if args.plots:
        plots(rows, Path(args.out))


if __name__ == "__main__":
    main()
