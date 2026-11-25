# 📚 In-vivo benchmark: one complete example

The full benchmark (questions and ground-truth answers for 150 Athletes and 105 Custom items) is in [`benchmark/`](benchmark/); the raw outputs of all ten models are released on Hugging Face as [FMantovani/do-llms-bind-episodes](https://huggingface.co/datasets/FMantovani/do-llms-bind-episodes) (folder `invivo/`) (`python src/In_vivo_benchmark/results/download_invivo_outputs.py` fetches them). This folder contains **one complete item, end to end**, so you can see exactly what the code asks, what it stores and how a binding failure is decided, without downloading anything.

The item is **Rio Ferdinand**, evaluated on **GPT-5.4**. GPT-5.4 knows each of his three teams between 2010 and 2015, but when asked for all of them at once it leaves out Queens Park Rangers in 5 of 5 trials: a binding failure as defined in Sec. 2 of the paper.

```
data/examples/
├── benchmark/
│   └── binding_test_cases_athletes_example.json      # the item: input of run_binding_benchmark.py
└── raw_outputs/athletes/openai_gpt-5.4/fail_rio_ferdinand/   # everything the code wrote for it (200 files)
```

---

## 1. The benchmark item

`binding_test_cases_athletes_example.json` has the same format as the full benchmark file, with a single entry:

```
{"test_cases": {"Rio Ferdinand": {"facts": …, "binding": …, "binding_multichoice": …}}}
```

**`facts`**: the verification questions. The item has three **episodes** (`a`, `b`, `c`), one per team, and three question types per episode:

| Key | Question | Ground truth |
|---|---|---|
| `fact_c_yesno` | In 2015, did Rio Ferdinand play for Queens Park Rangers F.C.? Answer yes or no. | Yes |
| `fact_c_team` | Which team did Rio Ferdinand play for in 2015? | Queens Park Rangers F.C. |
| `fact_c_period` | In which period did Rio Ferdinand play for Queens Park Rangers F.C.? | 2015 |

Episode `a` is England (2010) and episode `b` is Manchester United (2010), with the same three questions. In the Custom dataset the three types are `yesno`, `entity` and `date`.

**`binding`**: the binding question and its ground-truth set Y(q):

> *List ALL teams that Rio Ferdinand played for between 2010 and 2015, including both club and national teams. Provide your answer as a JSON list like ["a","b"].*
>
> Y(q) = `["England national association football team", "Manchester United F.C.", "Queens Park Rangers F.C."]`

`prompts.raw` is the plain question (*Instant* in the paper); `reason` and `no_reason` add an instruction at the end.

**`binding_multichoice`**: the same question as multiple choice (each team alone, all three together, or "None of them"), with the correct letter in `expected_answer` (`D`).

---

## 2. How one item is evaluated

Every question is asked **5 times** at temperature 0.3, with a system prompt telling the model to use only its internal knowledge and to say so when it does not know.

```
Step 1  Ask every verification question            3 episodes × 3 types × 5 trials = 45 answers
          a/ b/ c/  ── yesno_0..4, team_0..4, period_0..4

Step 2  Judge them, episode by episode (a, then b, then c)
          a question type passes if ≥ 3 of its 5 answers are correct
          an episode passes if all three of its types pass
          judging stops at the first episode that fails
          eval_a/ eval_team_a/ eval_period_a/  …

Step 3  Gate: is every episode known?
          no  → stop. The item is not part of this model's known set, no binding question is asked.
          yes → continue (Rio Ferdinand passes all 9 checks for GPT-5.4)

Step 4  Ask the binding question under every prompting strategy, 5 trials each
          binding_raw/ binding_reason/ binding_mc_raw/ …  0.json … 4.json

Step 5  Judge every binding answer
          list answer: correct if it contains ALL items of Y(q) (extra items are allowed, i.e. recall = 1)
          eval_binding_raw/ eval_binding_reason/ …  eval_0.json … eval_4.json

Step 6  Binding failure if ≥ 3 of the 5 trials are incorrect
```

Because of the gate in Step 3, each model has its **own set of known items**, and the failure rate in the paper is computed over that set (the N shown in each cell of Fig. 2). For items that stop at the gate, the folder contains the verification answers but no `binding_*` folders, and the `eval_*` folders of the episodes after the failing one are missing. That is expected, not missing data.

How each answer is judged:

| Question | Judge | Stored in `evaluation` |
|---|---|---|
| yes/no | Rule-based: the answer must say "yes" and not "no" | `detected_answer`, `correct` |
| team / period (Custom: entity / date) | LLM judge | `identified_items`, `correct`, `explanation` |
| binding, list format | LLM judge, recall-based | `found_expected`, `missing`, `correct` |
| binding, multiple choice | Rule-based letter match | `correct` |

---

## 3. What each subfolder contains

The folder name is the item name with a `fail_` prefix. The prefix is historical: **every** item has a `fail_` folder, whether or not the model fails it.

**Verification**

| Folder | Content |
|---|---|
| `a/`, `b/`, `c/` | One folder per episode: `yesno_{0..4}.json`, `team_{0..4}.json`, `period_{0..4}.json` |
| `eval_a/` … | Judge verdicts for the yes/no answers of that episode |
| `eval_team_a/` … , `eval_period_a/` … | Judge verdicts for the team and period answers |

**Binding**: one folder per prompting strategy, each with `0.json … 4.json`, and a matching `eval_binding_*` folder with the verdicts:

| Folder | Paper name (Fig. 2 / Fig. 8) | What changes in the prompt | GPT-5.4 correct |
|---|---|---|---|
| `binding_raw` | **Instant** | The plain question | **0/5** |
| `binding_no_reason` | NoReason | + "NO extra text, NO reasoning!" | 0/5 |
| `binding_direct_hinted` | Instant+Hint | "List ALL **3** teams …" | 0/5 |
| `binding_reason` | **Reason** ★ | + "Think about each event independently and reason before answering." Native extended thinking on | **5/5** |
| `binding_reason_hinted` | Reason+Hint | Reason + "List ALL **3** teams …" | 5/5 |
| `binding_reason_firstletter` | Reason+FirstLetter | Reason + "The first letters of the answers are: E, M, Q." | 5/5 |
| `binding_selfask` | **SelfAsk** | Few-shot decomposition examples of Press et al. (2023) | **5/5** |
| `binding_selfask_modified_2` | **SelfAsk-Binding** | Few-shot decomposition examples written for list retrieval | **5/5** |
| `binding_mc_raw` | **MC** | Multiple choice | **5/5** |
| `binding_mc_reason` | **MC+Reason** | Multiple choice + reasoning instruction, thinking on | **5/5** |
| `binding_mc_no_reason` | MC+NoReason | Multiple choice + "Do NOT write anything else" | 5/5 |

Strategies in bold are the six columns of Fig. 2; the other five appear only in Fig. 8 (App. B.4). ★ marks native extended thinking.

This one item shows the paper's main in-vivo findings in miniature. Asked directly, GPT-5.4 answers `["Manchester United","England"]` in all five trials, although it answered every Queens Park Rangers question correctly. With extended thinking or a SelfAsk decomposition it retrieves all three teams, and in multiple choice it always recognizes the full set (`D`).

---

## 4. What each file contains

**Answer file** (e.g. `binding_raw/0.json`):

| Field | Meaning |
|---|---|
| `entity` | Item name |
| `prompt` | Exact text sent to the model |
| `expected` | Ground truth (a list for binding questions, a string otherwise) |
| `retry` | Trial index, 0–4 |
| `response` | The model's final answer |
| `model` | Model identifier as sent to the API |
| `extended_reasoning_enabled` | Whether native thinking was enabled (binding questions only) |
| `reasoning_content`, `reasoning_details` | The thinking trace, when the API returns one |
| `finish_reason`, `usage`, `success` | API metadata |

**Verdict file** (e.g. `eval_binding_raw/eval_0.json`): `expected`, `response` and the `evaluation` described in Section 2. Verdicts of the Reason strategy also record `judge_model` and whether the judge read the thinking trace (`reasoning_content_used`, `found_in`). Example, trial 0 of Instant:

```json
{
  "expected": "['England national association football team', 'Manchester United F.C.', 'Queens Park Rangers F.C.']",
  "response": "[\"Manchester United\",\"England\"]",
  "evaluation": {
    "id": 0,
    "found_expected": ["England national association football team", "Manchester United F.C."],
    "missing": ["Queens Park Rangers F.C."],
    "correct": false
  }
}
```

---

## 5. From folders to the paper's numbers

All analysis code reads these folders through one module, [`src/In_vivo_benchmark/results/binding_gate.py`](../src/In_vivo_benchmark/results/binding_gate.py), which implements Steps 3 and 6 above:

- **Known item**: every verification question type of every episode has ≥ 3/5 correct answers.
- **Binding failure**: among known items, < 3/5 correct binding answers for a given strategy.
- **Failure rate** (each cell of Fig. 2) = binding failures / known items, per model and strategy.

To recompute the paper's figures and tables from the full release, see [`src/In_vivo_benchmark/results/README.md`](../src/In_vivo_benchmark/results/README.md).
