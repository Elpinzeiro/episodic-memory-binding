"""
The synthetic narratives of the paper (Sec. 4.1, App. C).

Three universes (default, news, scifi) x three sizes (10, 30, 100 events), seed 99, released in
two versions, one per generator model:
  data/narratives/           Claude 3.5 Sonnet: Llama and GPT-4.1 runs, logit lens, RSS
  data/narratives_sonnet46/  Claude Sonnet 4.6: Qwen and Mistral runs
plus the held-out narratives of the transfer-learning folds (Sonnet 4.6) in data/narratives_transfer/.
`load_narrative` loads them from there, so no API call is made. Generation only happens if a book
is missing (or another seed or universe is asked for), and then needs an Anthropic key in `.env`.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NARRATIVES_DIR = REPO_ROOT / "data" / "narratives"
TRANSFER_DIR = REPO_ROOT / "data" / "narratives_transfer"
ENV_FILE = REPO_ROOT / ".env"

UNIVERSES = ["default", "news", "scifi"]
EVENT_COUNTS = [10, 30, 100]
SEED = 99
GENERATOR_MODEL = "claude-3-5-sonnet-20240620"
# generator model -> folder of its released narratives
FOLDERS = {
    "claude-3-5-sonnet-20240620": NARRATIVES_DIR,
    "claude-sonnet-4-6": REPO_ROOT / "data" / "narratives_sonnet46",
}

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def load_narrative(universe, nb_events, seed=SEED, generator=GENERATOR_MODEL, data_folder=None, env_file=ENV_FILE):
    """Return the BenchmarkGenerationWrapper of one narrative (book, ground truth, df_qa)."""
    data_folder = data_folder or FOLDERS[generator]
    from In_vitro_experiments.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper

    prompt_parameters = {
        "nb_events": nb_events,
        "name_universe": universe,
        "name_styles": "concise",
        "seed": seed,
        "distribution_events": {"name": "geometric", "param": 0.3},
    }
    model_parameters = {"model_name": generator, "max_new_tokens": 4096, "itermax": 10}
    book_parameters = {"indexing": "default", "nb_summaries": 0}
    return BenchmarkGenerationWrapper(prompt_parameters, model_parameters, book_parameters,
                                      Path(data_folder), Path(env_file), verbose=False, rechecking=False)


def evaluated_questions(benchmark):
    """
    The evaluated question set of the paper (Tab. 2, last column; footnote on the 38/113/222 MEQ).

    Not a random sample: every `get == 'all'` question with a non-empty answer. SEQ have their
    answer in one chapter (k = 1), MEQ in several (k > 1).
    """
    df = benchmark.get_df_qa()
    df = df[(df["get"] == "all") & (df["n_chapters_correct_answer"] > 0)].reset_index(drop=True)
    df["kind"] = ["SEQ" if k == 1 else "MEQ" for k in df["n_chapters_correct_answer"]]
    return df
