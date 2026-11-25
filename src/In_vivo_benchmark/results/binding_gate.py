"""Shared known-fact gate for the in-vivo binding benchmark.

Every number in Fig. 2 and App. B.4 (the heatmaps, the threshold sweep, the
refusal/incomplete split, the precision table, the cardinality breakdown, the
bootstrap confidence intervals) is computed over the *same* population: an
entity counts as a verified known fact only if every judged fact-verification
eval for it -- yes/no, forward recall, reverse recall, one letter per
constituent episode -- is at or above a threshold (0.6 = >=3/5, the paper's
rule; App. B.4.1 also uses 0.8 and 1.0 for the strict-threshold robustness
check).

This module is the single source of truth for that population and for the
per-entity accuracy table everything else reads from. Every other script in
this folder imports `raw_table` / `known_population` from here instead of
re-deriving the population on its own.

Why this matters in practice: an earlier iteration of this analysis used a
looser proxy ("this entity has a recorded Instant trial") in several
side-analysis notebooks, on the assumption that it was equivalent to "known".
It silently is not, whenever a fact gets re-verified (and possibly re-scored
downward) after its binding trial has already been run -- an entity can have
a binding trial on file yet still fail the real fact gate. That mismatch
showed up as a real discrepancy between two supposedly-identical model/N
counts after a re-run of the experiment. Keeping exactly one implementation
of the gate, imported everywhere, is what prevents that from recurring.

Usage
-----
    from binding_gate import DATASETS, MODEL_DIRS, MODEL_LABELS, \
        PAPER_STRATEGIES, raw_table, known_population

    df, _ = raw_table("custom", "anthropic_claude-opus-4.6")
    known = known_population("custom", "anthropic_claude-opus-4.6", 0.6)
    failures = df.loc[df.index.isin(known), "binding_raw"] < BIND_THRESH
"""

import ast
import json
import os
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config -- edit this block if you re-ran the experiment under different
# output-dir names or added/removed models. `output_dir` here must match the
# `output_dir` argument passed to run_binding_benchmark.py.
# ---------------------------------------------------------------------------
# The benchmark files shipped with the repo (data/benchmark/).
BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "data" / "benchmark"

# The released outputs, as unpacked by download_invivo_outputs.py. When this
# folder exists it is used; otherwise the folders run_binding_benchmark.py writes.
DOWNLOADED_DIR = Path("invivo_outputs")

DATASETS = {
    "athletes": {
        "base_dir": (str(DOWNLOADED_DIR / "athletes") if DOWNLOADED_DIR.exists()
                     else "./binding_failures_athletes_rebuttal_temp0.3"),
        "json_path": str(BENCHMARK_DIR / "binding_test_cases_athletes.json"),
    },
    "custom": {
        "base_dir": (str(DOWNLOADED_DIR / "custom") if DOWNLOADED_DIR.exists()
                     else "./binding_failures_custom_rebuttal_temp0.3"),
        "json_path": str(BENCHMARK_DIR / "binding_test_cases_custom.json"),
    },
}

MODEL_DIRS = [
    "anthropic_claude-opus-4.6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-20250514",
    "openai_gpt-5.4",
    "x-ai_grok-4.20",
    "google_gemini-3.1-pro-preview",
    "gpt-4o-2024-08-06",
    "gemini-2.0-flash",
    "deepseek-chat",
    "mistralai_mistral-large-2512",
]

MODEL_LABELS = {
    "anthropic_claude-opus-4.6": "Claude Opus 4.6",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "openai_gpt-5.4": "GPT-5.4",
    "x-ai_grok-4.20": "Grok 4.20",
    "google_gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-4o-2024-08-06": "GPT-4o",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "deepseek-chat": "DeepSeek-V3",
    "mistralai_mistral-large-2512": "Mistral Large 25.12",
}

ROWS = [MODEL_LABELS[m] for m in MODEL_DIRS]

# Models with extended/native thinking enabled through the API (the "star"
# rows in Fig. 2 / App. B.4).
STAR_MODELS = {"Claude Opus 4.6", "Claude Opus 4.5", "Claude Sonnet 4",
               "GPT-5.4", "Grok 4.20", "Gemini 3.1 Pro"}

# Paper heatmap column -> directory name. One canonical name per column, on
# purpose: matching Fig. 2 / App. B.4 exactly means matching which directory
# name they treat as authoritative, not silently falling back to an older
# one (e.g. "binding_selfask_modified" without "_2") if it happens to exist.
# `diagnose_stale_directory_names()` below flags it if that fallback would
# ever have mattered for your data.
PAPER_STRATEGIES = {
    "Instant":         "binding_raw",
    "Reason":          "binding_reason",
    "SelfAsk":         "binding_selfask",
    "SelfAsk-Binding": "binding_selfask_modified_2",
    "MC":              "binding_mc_raw",
    "MC+Reason":       "binding_mc_reason",
}
PAPER_COLUMNS = list(PAPER_STRATEGIES)
MC_STRATEGIES = {"MC", "MC+Reason"}
# Judge-graded, list-style strategies: the only ones classify()/precision can
# be computed for (MC variants are rule-based letter matching).
DECOMPOSABLE_STRATEGIES = {"Instant", "Reason", "SelfAsk", "SelfAsk-Binding"}

# All eleven arms evaluated (Fig. 8 / App. B.4), used only to tell the fact
# gate which eval_* directories are BINDING arms rather than facts.
ALL_BINDING_TYPES = [
    "binding_raw", "binding_direct_hinted", "binding_reason",
    "binding_reason_firstletter", "binding_reason_hinted", "binding_no_reason",
    "binding_selfask", "binding_selfask_modified_2", "binding_selfask_modified",
    "binding_mc_raw", "binding_mc_reason", "binding_mc_no_reason",
]

BIND_THRESH = 0.6          # instance fails if accuracy < 0.6 (>=3/5 trials wrong)
FACT_KNOW_THRESH = 0.6     # the paper's "known" rule: every judged fact >=3/5
REQUIRE_FACT_EVALS = True  # an entity with zero judged fact evals is NOT known


# --------------------------------------------------------------- readers -----
def load_json_entities(json_path):
    """Set of `fail_<entity>` directory names the test-case JSON defines."""
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {"fail_" + k.replace(" ", "_").lower() for k in data["test_cases"].keys()}


def read_trials(fail_dir: Path, strategy_dir):
    """Per-trial dicts for one instance + one strategy directory.

    Returns [] if the strategy was never run for this instance -- callers
    should treat that as "no data", not as a failure.
    """
    if strategy_dir is None:
        return []
    eval_dir = fail_dir / f"eval_{strategy_dir}"
    if not eval_dir.exists():
        return []
    trials = []
    for f in sorted(eval_dir.glob("eval_*.json")):
        if f.name.startswith("_debug"):
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        ev = d.get("evaluation", d)  # MC eval files put `correct` at top level
        trials.append({
            "eval_file": f,
            "raw_file": fail_dir / strategy_dir / f.name[len("eval_"):],
            "expected": d.get("expected"),
            "response": d.get("response", ""),
            "prompt": d.get("prompt"),
            "correct": bool(ev.get("correct", False)),
            "found_expected": ev.get("found_expected"),
            "missing": ev.get("missing"),
            "judge_error": ev.get("error"),
        })
    return trials


def parse_expected(expected):
    """`"['AFC Ajax', 'PSG']"` -> `['AFC Ajax', 'PSG']`. `[]` if unparsable."""
    if expected is None:
        return []
    if isinstance(expected, list):
        return expected
    for parser in (ast.literal_eval, json.loads):
        try:
            value = parser(str(expected))
            if isinstance(value, list):
                return value
        except Exception:
            pass
    return []


def iter_instances(tag, model_dir):
    """Yield each `fail_*` instance directory for (dataset, model), filtered
    to the entities the benchmark's own test-case JSON defines."""
    cfg = DATASETS[tag]
    keep = load_json_entities(cfg["json_path"])
    model_path = Path(cfg["base_dir"]) / model_dir
    if not model_path.exists():
        return
    for fail_dir in sorted(d for d in model_path.glob("fail_*") if d.is_dir()):
        if keep is not None and fail_dir.name not in keep:
            continue
        yield fail_dir


# ------------------------------------------------------------ fact gate -----
def compute_accuracy(eval_dir: Path, pattern: str = "eval_*.json") -> float:
    """Fraction of eval files marked correct. Handles both eval schemas: list
    arms nest the verdict under 'evaluation', MC arms put it at the top level."""
    correct = total = 0
    for path in sorted(eval_dir.glob(pattern)):
        if path.name.startswith("_debug"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        total += 1
        verdict = data.get("evaluation")
        is_correct = (bool(verdict.get("correct", False)) if isinstance(verdict, dict)
                      else bool(data.get("correct", False)))
        correct += is_correct
    return correct / total if total > 0 else float("nan")


def summarize_fact_gate(entity_table: pd.DataFrame, fact_know_thresh: float) -> pd.DataFrame:
    """Per-entity fact-gate verdict.

    An entity is known iff it has at least one judged fact eval and every
    judged fact eval is at or above `fact_know_thresh`. This is the single
    definition of the known population, reused by every table.
    """
    columns = ["n_facts", "min_fact", "worst_fact", "known"]
    if not len(entity_table):
        return pd.DataFrame(columns=columns)
    fact_columns = [c for c in entity_table.columns if c not in ALL_BINDING_TYPES]
    records = {}
    for name, row in entity_table.iterrows():
        fact_values = row[fact_columns].dropna() if fact_columns else pd.Series(dtype=float)
        if len(fact_values) == 0:
            records[name] = {"n_facts": 0, "min_fact": float("nan"), "worst_fact": "",
                              "known": not REQUIRE_FACT_EVALS}
            continue
        records[name] = {
            "n_facts": int(len(fact_values)), "min_fact": float(fact_values.min()),
            "worst_fact": str(fact_values.idxmin()),
            "known": bool(fact_values.min() >= fact_know_thresh),
        }
    return pd.DataFrame.from_dict(records, orient="index")[columns]


def _entity_raw_table(model_dir, base_dir, json_path):
    """Per-entity accuracy table: every fact-eval directory (auto-detected by
    name, so it handles the athletes dataset's team/period letters and the
    custom dataset's entity/date letters the same way) plus every one of
    ALL_BINDING_TYPES's eval directories. One row per `fail_*` instance."""
    model_path = Path(base_dir) / model_dir.replace(":", "_")
    if not model_path.exists():
        return pd.DataFrame(), set()

    json_entities = set()
    if json_path and os.path.exists(json_path):
        cases = json.loads(Path(json_path).read_text(encoding="utf-8"))["test_cases"]
        for name in cases:
            json_entities.add(f"fail_{name.replace(' ', '_').lower()}")

    results = {}
    for fail_dir in model_path.glob("fail_*"):
        if not fail_dir.is_dir():
            continue
        if json_entities and fail_dir.name not in json_entities:
            continue

        row = {}
        facts = set()
        for subdir in fail_dir.iterdir():
            if not subdir.is_dir() or not subdir.name.startswith("eval_"):
                continue
            if subdir.name.startswith("eval_binding"):
                continue
            if re.match(r"^eval_[a-z]$", subdir.name):
                facts.add(subdir.name[5:])
            elif re.match(r"^eval_\w+_[a-z]$", subdir.name):
                facts.add(subdir.name.split("_")[-1])

        for fact in sorted(facts):
            for eval_dir in fail_dir.glob(f"eval_*_{fact}"):
                if eval_dir.name.startswith("eval_binding"):
                    continue
                qtype = eval_dir.name.replace(f"_{fact}", "").replace("eval_", "")
                row[f"{qtype}_{fact}"] = compute_accuracy(eval_dir)
            yesno_dir = fail_dir / f"eval_{fact}"
            if yesno_dir.exists():
                row[f"yesno_{fact}"] = compute_accuracy(yesno_dir)

        for binding_type in ALL_BINDING_TYPES:
            binding_dir = fail_dir / f"eval_{binding_type}"
            if binding_dir.exists():
                row[binding_type] = compute_accuracy(binding_dir)

        results[fail_dir.name] = row

    return pd.DataFrame.from_dict(results, orient="index"), json_entities


_RAW_TABLE_CACHE = {}


def raw_table(tag, model_dir):
    """(per-entity accuracy DataFrame, json_entities set) for (dataset, model),
    cached -- this is the one full scan of the eval JSON files; every other
    function in this module and every script in this folder reuses it."""
    key = (tag, model_dir)
    if key not in _RAW_TABLE_CACHE:
        cfg = DATASETS[tag]
        _RAW_TABLE_CACHE[key] = _entity_raw_table(model_dir, cfg["base_dir"], cfg["json_path"])
    return _RAW_TABLE_CACHE[key]


def known_population(tag, model_dir, fact_know_thresh=FACT_KNOW_THRESH, require_binding=True):
    """Set of entity names passing the fact gate at `fact_know_thresh`
    (0.6 = known@3/5, the paper's rule; 0.8 = known@4/5; 1.0 = known@5/5),
    optionally also requiring a recorded Instant (binding_raw) trial --
    Fig. 2 / App. B.4's exact population.
    """
    entity_table, _ = raw_table(tag, model_dir)
    if entity_table.empty:
        return set()
    gate = summarize_fact_gate(entity_table, fact_know_thresh)
    known = set(gate.index[gate["known"]])
    if require_binding and "binding_raw" in entity_table.columns:
        known &= set(entity_table.index[entity_table["binding_raw"].notna()])
    return known


def warm_cache(datasets=DATASETS, model_dirs=MODEL_DIRS):
    """Scan every fact + binding eval file once per (dataset, model). Calling
    this up front turns every later `raw_table`/`known_population` call into
    a cache hit instead of scanning the filesystem again, so it's worth doing
    before running several tables in the same process."""
    for tag in datasets:
        for model_dir in model_dirs:
            raw_table(tag, model_dir)


def diagnose_stale_directory_names(datasets=DATASETS, model_dirs=MODEL_DIRS):
    """Print a warning for any (dataset, model) that only has the older
    'binding_selfask_modified' directory (no '_2'). PAPER_STRATEGIES does not
    fall back to it on purpose (Fig. 2 / App. B.4 don't either), so those
    cells render as N/A rather than silently pulling from a different arm --
    this just makes that visible instead of silent.
    """
    affected = []
    for tag in datasets:
        for model_dir in model_dirs:
            entity_table, _ = raw_table(tag, model_dir)
            if ("binding_selfask_modified" in entity_table.columns
                    and "binding_selfask_modified_2" not in entity_table.columns):
                affected.append((tag, model_dir))
    if affected:
        print(f"NOTE: {affected} only have 'binding_selfask_modified' (no '_2'). "
              "Their SelfAsk-Binding cell will be N/A, matching Fig. 2's own behaviour.")
    return affected
