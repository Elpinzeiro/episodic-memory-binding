"""
Tab. 24 (App. E.1.9): coverage of the RSS training set, i.e. the share of evaluated MEQ tuples
(30-event narratives) that also appear among the Q&A pairs the Recollector/Merger produced.

    python src/In_vitro_experiments/analysis/coverage.py

Inputs, per universe, in `data/narratives/<universe>/books/<30-event book>/coverage/`:
  test_set.json        the 113 evaluated MEQ tuples
  all_chapters.json    the Q&A pairs the RSS pipeline produced over all 30 chapters (training set)
  comparison.json      per test tuple: matched or not, and the matching training tuple
The matching itself was done by an LLM with the prompt of App. E.1.9; this script only counts it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import NARRATIVES_DIR

COLUMNS = [("default", "Default"), ("scifi", "Sci-Fi"), ("news", "News")]


def coverage(universe):
    folder = next((NARRATIVES_DIR / f"U{universe}_Sconcise_seed99" / "books").glob("*_nbchapters_30_*")) / "coverage"
    comparison = json.loads((folder / "comparison.json").read_text())
    test_rows = json.loads((folder / "test_set.json").read_text())["rows"]
    assert len(comparison) == len(test_rows)
    return 100 * sum(item["match"] for item in comparison) / len(comparison), len(comparison)


def main():
    results = {label: coverage(universe) for universe, label in COLUMNS}
    print(f"{'':12}" + "".join(f"{label:>9}" for label in results))
    print(f"{'Match (%)':12}" + "".join(f"{pct:>9.2f}" for pct, _ in results.values()))
    print(f"{'Total MEQs':12}" + "".join(f"{n:>9d}" for _, n in results.values()))


if __name__ == "__main__":
    main()
