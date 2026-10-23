"""
Your own runs -> the binned table of Fig. 3 and a --scores file for binding_heatmaps.py.

    python src/In_vitro_experiments/analysis/run_to_scores.py <run_dir> [<run_dir> ...] --out my_scores.json
    python src/In_vitro_experiments/analysis/binding_heatmaps.py --scores my_scores.json --out figures/

A run folder is the one holding `evaluated_answers/` (0.json, 1.json, ... written by the judge). The
narrative a run was evaluated on (universe, number of events, generator) is found automatically: the
ground-truth items of the judged answers are matched against the released narratives in data/narratives*/.

Scores are the paper's (EvaluationWrapper.get_accuracy_summary_relative_to): on the evaluated questions
(`narratives.evaluated_questions`: get == 'all', non-empty answer), the % of answers with recall = 1, per
bin of matching events k ('1' = SEQ; '2', '3-5', '6+' = MEQ; bins [0,1,2,3,6,inf) as in evaluation_wrapper).
Runs with the same label, event size and universe are averaged (std over runs is printed and stored
under "std_over_runs"). The label is the model folder name (before '_bs_') unless --label is given.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import REPO_ROOT

BINS = ["1", "2", "3-5", "6+"]
BOOK_RE = re.compile(r"U(?P<universe>[^_]+)_Sconcise_seed\d+/books/model_(?P<generator>.+)_itermax_\d+_I\w+?_nbchapters_(?P<events>\d+)_")


def released_narratives():
    """(universe, events, generator, df_qa.parquet, #questions, #questions with a non-empty answer) of every released narrative."""
    out = []
    for path in sorted(REPO_ROOT.glob("data/narratives*/U*_Sconcise_seed*/books/*/df_qa.parquet")):
        m = BOOK_RE.search(path.relative_to(REPO_ROOT).as_posix())
        if m:
            n_items = pd.read_parquet(path, columns=["n_items_correct_answer"])["n_items_correct_answer"]
            out.append((m["universe"], int(m["events"]), m["generator"], path, len(n_items), int((n_items != 0).sum())))
    return out


def load_run(run_dir):
    """The judged answers of one run, in question order (file i = question i)."""
    folder = Path(run_dir) / "evaluated_answers"
    folder = folder if folder.is_dir() else Path(run_dir)  # the evaluated_answers/ folder itself is fine too
    files = sorted((f for f in folder.glob("*.json") if f.stem.isdigit()), key=lambda f: int(f.stem))
    if not files:
        raise SystemExit(f"{run_dir}: no evaluated_answers/<i>.json")
    if [int(f.stem) for f in files] != list(range(len(files))):
        raise SystemExit(f"{run_dir}: evaluated_answers/ is incomplete (expected 0..{len(files) - 1}.json); finish the evaluation first")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def match_narrative(answers, narratives):
    """The released narratives (and question lists) whose ground truth matches the judged answers, best first."""
    gt = [sorted(map(str, a["groundtruth_items"])) for a in answers]
    found = []
    for universe, events, generator, path, n_all, n_nonempty in narratives:
        if len(gt) not in (n_all, n_nonempty):
            continue
        df_qa = pd.read_parquet(path)
        # all questions, or only those with a non-empty answer (not_eval_hall=True, the default of step_2)
        for questions in (df_qa, df_qa[df_qa["n_items_correct_answer"] != 0]):
            if len(questions) != len(gt):
                continue
            # >= 95%, not all: the judge writes a generic ground truth for full-chapter questions ("Full event details")
            share = np.mean([sorted(map(str, c)) == g for c, g in zip(questions["correct_answer"], gt)])
            if share >= 0.95:
                found.append((share, universe, events, generator, questions.reset_index(drop=True)))
    return sorted(found, key=lambda f: -f[0])


def bin_scores(answers, questions):
    """bin -> (% recall = 1, count) on the paper's evaluated questions."""
    df = questions[["get", "n_chapters_correct_answer"]].copy()
    df["recall"] = [a.get("recall", np.nan) for a in answers]
    df = df[(df["get"] == "all") & (df["n_chapters_correct_answer"] > 0)]
    df["bin"] = pd.cut(df["n_chapters_correct_answer"], bins=[0, 1, 2, 3, 6, np.inf],
                       labels=["0", "1", "2", "3-5", "6+"], include_lowest=True, right=False)
    df["perfect_match"] = (df["recall"] == 1.0).astype(int)
    return {b: (100 * g["perfect_match"].mean(), len(g)) for b, g in df.groupby("bin", observed=True) if b in BINS}


def default_label(run_dir):
    path = Path(run_dir).resolve()
    name = path.parent.name if path.name == "evaluated_answers" else path.name
    return name.split("_bs_")[0] if "_bs_" in name else name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (each holding evaluated_answers/), or evaluated_answers/ folders")
    ap.add_argument("--out", default="my_scores.json", help="--scores file for binding_heatmaps.py")
    ap.add_argument("--label", help="model name for all runs (default: the model folder name)")
    ap.add_argument("--panel", default="my_runs", help="panel name in the --scores file (one heatmap per panel)")
    args = ap.parse_args()

    narratives = released_narratives()
    cells = {}  # (label, events, universe) -> list of {bin: (score, count)}
    for run in args.runs:
        answers = load_run(run)
        found = match_narrative(answers, narratives)
        if not found:
            print(f"{run}\n  skipped: its {len(answers)} judged answers match no released narrative in data/narratives*/")
            continue
        _, universe, events, generator, questions = found[0]
        same = sorted({f[3] for f in found if f[1:3] == (universe, events)} - {generator})
        print(f"{run}\n  -> {universe}, {events} events ({' = '.join([generator] + same)}), {len(answers)} judged answers")
        # narratives of two generators can share their events, hence the same questions: then either gives the same scores
        others = {f"{f[1]}/{f[2]} ({f[3]})" for f in found[1:] if bin_scores(answers, f[4]) != bin_scores(answers, questions)}
        if others:
            print(f"  warning: also matches {', '.join(sorted(others))} with other scores; check which narrative the run used")
        label = args.label or default_label(run)
        cells.setdefault((label, events, universe), []).append(bin_scores(answers, questions))

    if not cells:
        raise SystemExit("no run matched a released narrative")
    panel, stds = {}, {}
    print("\n| model | events | universe | runs | " + " | ".join(f"k={b} (n)" for b in BINS) + " |")
    print("|---|---|---|---|" + "---|" * len(BINS))
    for (label, events, universe), runs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        means, sds, row = {}, {}, []
        for b in BINS:
            values = [r[b][0] for r in runs if b in r]
            if not values:
                row.append("—")
                continue
            means[b], sds[b] = round(float(np.mean(values)), 2), round(float(np.std(values)), 2)
            row.append(f"{means[b]:.2f}±{sds[b]:.2f} ({runs[0][b][1]})")
        panel.setdefault(label, {}).setdefault(str(events), {})[universe] = means
        stds.setdefault(label, {}).setdefault(str(events), {})[universe] = sds
        print(f"| {label} | {events} | {universe} | {len(runs)} | " + " | ".join(row) + " |")

    out = {"note": "Written by run_to_scores.py: % of answers with recall = 1 per bin of matching events, "
                   "mean over runs; keys: panel -> model -> event size -> universe -> bin.",
           args.panel: panel, "std_over_runs": {args.panel: stds}}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nsaved {args.out}  (plot: binding_heatmaps.py --scores {args.out})")


if __name__ == "__main__":
    main()
