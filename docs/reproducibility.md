# Repository Structure and Path Configuration

## Overview

This repository contains the codebase for multiple experimental components of the paper "Do LLMs Bind Episodes? Probing Cross-Episode Parametric Retrieval Through Shared Cues".  
It has been refactored to support a modular structure aligned with the organization of the associated paper.

The codebase is now divided into independent experimental modules:

- `src/In_vitro_experiments/`
- `src/In_vivo_benchmark/`

Each module represents a logically self-contained subproject with its own data processing, experiment pipeline, and evaluation workflow.

---

## Modular Design

Each experimental module should be considered an independent execution unit.

Typically, each module contains:

- data loading and preprocessing logic  
- experiment or generation scripts  
- evaluation pipelines  
- output directories (results, logs, artifacts)


---

## Paths

All scripts run from the repository root (`python src/...`); they find the repository from their
own location, so no path or environment variable has to be set. The in-vitro code is imported as
the package `In_vitro_experiments` (with `src/` on `sys.path`, which every script and notebook does).

- In-vitro narratives: `data/narratives/` (Claude 3.5 Sonnet: Llama, GPT-4.1) and
  `data/narratives_sonnet46/` (Claude Sonnet 4.6: Qwen, Mistral), each with
  `U<universe>_Sconcise_seed99/` (events, generation cache, books with `df_qa.parquet`);
  `data/narratives_transfer/` holds the held-out narratives of the transfer-learning folds.
  `In_vitro_experiments/narratives.py` loads them; the generator is only called if a book is missing.
  Training outputs go to `data/narratives*/<universe>/answers/` (not in git).
- Transfer-learning training sets: `python src/In_vitro_experiments/download_transfer_sets.py`
  (Hugging Face) writes them to `data/transfer_training_sets/` (not in git).
- In-vitro released outputs: `data/results/` (logit-lens / forced-prefix JSONs, scores behind Fig. 3, 5, 7, 10, 14).
- In-vivo benchmark: `data/benchmark/`; raw outputs on Hugging Face (see `results/README.md`).

---

## About In Vivo Section

The in vivo benchmark evaluates whether large language models can correctly bind multiple temporally distinct episodes that share a common cue. The central hypothesis is that if episodic binding reliably emerges from scale once constituent facts are encoded, then frontier models should demonstrate this capability under controlled retrieval settings.

To test this, we use a two-stage evaluation protocol:

1. **Single-fact verification** — first verify that the model correctly retrieves each individual constituent fact.
2. **Binding evaluation** — then evaluate whether the model can integrate multiple verified facts through a single multi-episode query.

### Dataset construction

The benchmark combines two complementary data sources:

- a filtered subset of TempLAMA containing athlete-related temporal facts across different years
- a newly curated dataset of public-event binding cases spanning entities such as organizations, locations, and event types

Each benchmark instance contains:

- a shared cue
- 2--4 temporally distinct episodes associated with that cue
- single-episode verification questions
- a multi-episode binding question requiring retrieval of the complete set

The restriction to 2--4 episodes is intentional to avoid cue-overload effects, where failures may reflect general retrieval limitations rather than binding-specific errors.

### Dataset generation pipeline

The two benchmark sources are generated independently through dedicated scripts:

- `generation/athletes_dataset_generation/build_athletes_dataset.py`
- `generation/custom_dataset_generation/llm_binding_example_generation.py`

The full verification + binding evaluation protocol is then executed through:

- `experiments/run_binding_benchmark.py`

This design separates dataset construction from evaluation, enabling controlled comparisons across models, prompting strategies, and experimental settings.

### Reproducing the paper's figures and tables

Once `run_binding_benchmark.py` has produced the
`binding_failures_{athletes,custom}_..._temp0.3/` folders, every figure and
table derived from them (Fig. 2, Fig. 8 and Tables 7–12 of App. B.4–B.5) is
reproduced by the scripts in
`results/` — see `results/README.md` for the full mapping and run order.
They all share one module, `results/binding_gate.py`, for the definition of
a "verified known fact" (the 3/5 threshold rule of Sec. 3.1), so that definition
lives in exactly one place instead of being re-derived per script.

## About In Vitro Section:

### RSS Experiments

To ensure modularity and reproducibility, our framework decouples experiment *logic* from experiment *configuration*. In particular, the second-stage pipeline (`step_2_.py` family) primarily targets transfer learning setups involving single-event and multi-event questions (SEQ/MEQ). However, the same execution pipeline can be seamlessly adapted to Recall-based Synthetic Supervision (RSS) experiments without modifying the underlying training or evaluation code.

Instead of introducing a separate pipeline implementation, RSS experiments are defined entirely through modifications of the `answering_parameters` configuration passed to the `EvaluationWrapper`. This design allows all experiment variants to share the same execution backbone while differing only in data policy, supervision structure, and prompt composition.

In this setting, RSS behavior is fully determined by the selected prompting and data-handling strategy, enabling direct comparison between transfer learning and RSS-style supervision under a unified framework.

#### Prompt design

All prompts used in RSS and transfer learning experiments are centrally defined in `generation/prompts.py`, ensuring consistency across experimental conditions and simplifying prompt-level ablations.

The key prompt components are the following. The file also contains additional prompt variants and ablations used during the exploratory phase of development.

- **Question generation prompt**: `generating_question_chapter_prompt_1`  
  Used to generate chapter-level questions from episodic content.

- **Recollection / merging prompt**: `generate_recollection_questions_prompt_generic_with_context_1`  
  Used to merge and structure recollection-based supervision signals across contexts.

- **Hallucination filtering prompt**: `smart_llm_remove_hall`  
  Used to remove or correct hallucinated or low-quality generated questions.

#### Design rationale

This design reflects a separation of concerns:

- `step_2.py` defines the *execution pipeline* (training, evaluation, benchmarking)
- `answering_parameters` define the *experimental regime* (RSS vs SEQ/MEQ, supervision level, data policy)
- `prompts.py` defines the *data construction and transformation layer* (how supervision signals are generated and refined, which RSS components are used)

As a result, RSS experiments are not implemented as a separate pipeline, but instead emerge from a controlled configuration of the same underlying system.

### O-LoRA ablation experiment

For parameter-efficient fine-tuning, we build upon the O-LoRA framework (Orthogonal Low-Rank Adaptation), which introduces orthogonal constraints to reduce interference between learned subspaces during continual adaptation.

We integrate and adapt the open-source implementation provided by the original authors to fit our experimental pipeline and dataset structure.

Repository: https://github.com/cmnfriend/O-LoRA


