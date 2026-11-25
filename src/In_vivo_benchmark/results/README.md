# In-vivo results: Fig. 2, Fig. 8 and Tables 7–12

Everything here reads the raw evaluation output of
`experiments/run_binding_benchmark.py` and reproduces one specific figure or
table of the paper. The output is either downloaded (`invivo_outputs/`, see Run
order) or written by your own run (`binding_failures_{athletes,custom}_rebuttal_temp0.3/`);
`binding_gate.py` uses `invivo_outputs/` when it exists. Run every script below
**from the folder that contains the data** (not from here); each script's
docstring has the exact command.

| File | Reproduces |
|---|---|
| `binding_gate.py` | Not runnable on its own. The shared *known-fact gate* (3/5 trials correct on every verification question, Sec. 3.1) that every other file imports, so the population behind every number is defined in exactly one place. |
| `plot_fig2_and_appendix_heatmaps.ipynb` | Fig. 2 (six prompting strategies) and Fig. 8 (App. B.4, all eleven strategies) |
| `table_threshold_sweep.py` | Table 7 (App. B.4.1, 3/5 → 4/5 → 5/5 threshold) and Table 9 (App. B.4.3, per-failure audit of Claude Opus 4.6) |
| `table_cardinality.py` | Table 8 (App. B.4.2, failure rate by answer-set size) |
| `table_refusal_incomplete.py` | Table 10 (App. B.4.4, refusal vs. incomplete failures) |
| `table_precision.py` | Table 11 (App. B.4.5, precision in vivo) |
| `download_invivo_outputs.py` | Not a figure: downloads the released raw outputs of all ten models (see Run order) |
| `significance_bootstrap_ci.py` | Table 12 (App. B.5.1, paired bootstrap CIs, 10,000 resamples) |

Each script prints its table and also writes a CSV to `rebuttal_analysis_arxiv/`
(created next to the data folders, not here).

`evaluation_notebook_complete.ipynb` and `investigate_reasoning_failures.ipynb`
predate `binding_gate.py` and re-derive the known population their own way.
They are kept for qualitative inspection of failures (App. B.6), not for
reproducing the numbers above.

## Run order

1. Get the raw outputs, either way:
   - **Download** them (53 MB, no API key):
     `python src/In_vivo_benchmark/results/download_invivo_outputs.py`.
     It fetches the `invivo/` folder of [FMantovani/do-llms-bind-episodes](https://huggingface.co/datasets/FMantovani/do-llms-bind-episodes),
     checks every archive and unpacks it into `invivo_outputs/`.
   - **Re-run** them: `../experiments/run_binding_benchmark.py` once per model
     (the slow, API-calling step).
2. Any script in this folder, in any order. They all read the same evaluation
   output and do not depend on each other.
