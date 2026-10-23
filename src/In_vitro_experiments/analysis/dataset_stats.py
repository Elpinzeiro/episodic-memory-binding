"""
Tab. 2: dataset statistics of the narratives, computed from the released books.

    python src/In_vitro_experiments/analysis/dataset_stats.py

SEQ / MEQ: fine-tuning questions whose answer lies in one / several chapters (per universe:
default / news / scifi). Evaluated: the evaluated subset (`narratives.evaluated_questions`),
split by ground-truth set size k.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import UNIVERSES, EVENT_COUNTS, load_narrative, evaluated_questions

K_BINS = {"1": "k=1", "2": "k=2", "{3,4,5}": "k=3-5", "6+": "k=6+"}


def main():
    print(f"{'Events':>6} | {'SEQ (def/news/scifi)':>22} | {'MEQ (def/news/scifi)':>22} | evaluated k=1 / 2 / 3-5 / 6+")
    for nb_events in EVENT_COUNTS:
        seq, meq, evaluated = [], [], []
        for universe in UNIVERSES:
            bench = load_narrative(universe, nb_events)
            ft = bench.finetuning_questions
            seq.append((ft["n_chapters_correct_answer"] == 1).sum())
            meq.append((ft["n_chapters_correct_answer"] > 1).sum())
            counts = evaluated_questions(bench)["bins_items_correct_answer"].value_counts()
            evaluated.append(" / ".join(str(counts.get(b, 0)) for b in K_BINS))
        dist = evaluated[0] if len(set(evaluated)) == 1 else " | ".join(evaluated)
        print(f"{nb_events:>6} | {' / '.join(map(str, seq)):>22} | {' / '.join(map(str, meq)):>22} | {dist}")


if __name__ == "__main__":
    main()
