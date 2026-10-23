#!/usr/bin/env python3
"""
Logit-lens and forced-prefix analysis (Sec. 4.2.2, Fig. 4, App. F: Fig. 11-13, Tab. 20).

Input: a Llama-3-8B LoRA adapter trained on the SEQ of one narrative (static training, 40 epochs,
lr 1e-5, bs 16; 50 epochs for 100 events), e.g. produced by step_2_train_on_SEQ_and_MEQ.py (policy
'single'). Questions: every MEQ with k = 2 of that narrative (get == 'all').

    # top-8 first tokens at the last prompt position (Fig. 4 left / Fig. 12)
    python src/In_vitro_experiments/experiments/step_5_logit_lens.py topk --universe default --events 30 --lora <adapter_dir>
    # force the first token / word of each omitted entity (Fig. 4 right / Fig. 13)
    python src/In_vitro_experiments/experiments/step_5_logit_lens.py forced-prefix --universe default --events 30 --lora <adapter_dir>
    # layer-by-layer top-8 heatmap of one question (Fig. 11: default, 10 events, question 0)
    python src/In_vitro_experiments/experiments/step_5_logit_lens.py heatmap --universe default --events 10 --lora <adapter_dir> --question 0

One JSON per question is written to data/results/logit_lens/U<universe>_ev<events>/<topk_first_token|
forced_prefix_continuation>/q<idx>.json (existing files are kept). The released outputs of the paper
are already there; analysis/logit_lens.py turns them into Tab. 20 and Fig. 4.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from In_vitro_experiments.narratives import REPO_ROOT, UNIVERSES, EVENT_COUNTS, load_narrative

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SYSTEM_PROMPT = ("You are an expert in memory tests regarding the fictional book "
                 "\"Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition\".")
RESULTS_DIR = REPO_ROOT / "data" / "results" / "logit_lens"
TOPK = 8
MAX_NEW_FREE, MAX_NEW_FORCED = 200, 40


def two_entity_questions(benchmark):
    """MEQ with exactly two ground-truth entities, in df_qa order."""
    df = benchmark.get_df_qa()
    df = df[(df["bins_items_correct_answer"] == "2") & (df["get"] == "all")]
    return [{"question": q, "GT_entities": list(a)} for q, a in zip(df["question"], df["correct_answer"])
            if len(set(a)) >= 2]


def load_model(lora_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
                                                 attn_implementation="eager")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return tokenizer, model


def prompt_ids(tokenizer, model, benchmark, question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": benchmark.get_user_prompt_for_finetuning(question)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)


def first_token(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return ids[0] if ids else None


def run_topk(questions, benchmark, tokenizer, model, out_dir):
    import torch

    for qi, target in enumerate(questions):
        path = out_dir / f"q{qi:04d}.json"
        if path.exists():
            continue
        ids = prompt_ids(tokenizer, model, benchmark, target["question"])
        with torch.no_grad():
            logits = model(ids, return_dict=True).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        topv, topi = torch.topk(probs, TOPK)

        def pack(entity):
            bare, lead = first_token(tokenizer, entity), first_token(tokenizer, " " + entity)
            return {"bare_id": bare, "bare_str": tokenizer.decode([bare]) if bare is not None else None,
                    "lead_id": lead, "lead_str": tokenizer.decode([lead]) if lead is not None else None}

        ent_a, ent_b = target["GT_entities"][:2]
        report = {"question_idx": qi, "question": target["question"], "GT_entities": target["GT_entities"],
                  "ent_A": ent_a, "ent_B": ent_b, "topk": TOPK, "top_ids": [int(x) for x in topi.tolist()],
                  "top_tokens": [{"id": int(i), "str": tokenizer.decode([int(i)]), "prob": float(p)}
                                 for i, p in zip(topi, topv)],
                  "first_token_A": pack(ent_a), "first_token_B": pack(ent_b)}
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def run_forced_prefix(questions, benchmark, tokenizer, model, out_dir):
    import torch

    def generate(ids, max_new_tokens):
        with torch.no_grad():
            return model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id, use_cache=True)

    def contains(entity, text):
        return entity.lower() in text.lower()

    for qi, target in enumerate(questions):
        path = out_dir / f"q{qi:04d}.json"
        if path.exists():
            continue
        ids = prompt_ids(tokenizer, model, benchmark, target["question"])
        n = ids.shape[1]
        free_text = tokenizer.decode(generate(ids, MAX_NEW_FREE)[0, n:], skip_special_tokens=True)
        missing = [e for e in target["GT_entities"] if not contains(e, free_text)]

        experiments = []
        for entity in missing:
            tok = first_token(tokenizer, entity)
            if tok is None:
                continue
            tok_ids = torch.cat([ids, torch.tensor([[tok]], device=ids.device)], dim=1)
            tok_text = tokenizer.decode(generate(tok_ids, MAX_NEW_FORCED)[0, n:], skip_special_tokens=True)
            word = entity.split()[0]
            word_ids = torch.cat([ids, tokenizer(word, add_special_tokens=False, return_tensors="pt").input_ids.to(ids.device)], dim=1)
            word_text = tokenizer.decode(generate(word_ids, MAX_NEW_FORCED)[0, n:], skip_special_tokens=True)
            experiments.append({
                "missing_entity": entity,
                "first_token": {"id": tok, "str": tokenizer.decode([tok]), "forced_continuation": tok_text,
                                "recovered": contains(entity, tok_text)},
                "first_word": {"str": word, "forced_continuation": word_text, "recovered": contains(entity, word_text)},
            })

        report = {"question_idx": qi, "question": target["question"], "gt_entities": target["GT_entities"],
                  "free_generation": free_text, "missing_entities": missing,
                  "is_binding_failure": len(missing) > 0, "forced_prefix_experiments": experiments}
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def run_heatmap(target, benchmark, tokenizer, model, out_pdf):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    ids = prompt_ids(tokenizer, model, benchmark, target["question"])
    with torch.no_grad():
        hidden = model(ids, output_hidden_states=True, return_dict=True).hidden_states
    norm, lm_head = model.base_model.model.model.norm, model.get_output_embeddings()
    tokens, scores = [], np.zeros((len(hidden), TOPK))
    for layer, h in enumerate(hidden):
        probs = torch.softmax(lm_head(norm(h[0, -1])).float(), dim=-1)
        top_p, top_i = torch.topk(probs, TOPK)
        tokens.append([tokenizer.decode([int(t)]) for t in top_i])
        scores[layer] = top_p.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 16))
    im = ax.imshow(scores, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    for i in range(len(hidden)):
        for j in range(TOPK):
            ax.text(j, i, f"{tokens[i][j][:8]}\n{scores[i][j]:.2f}", ha="center", va="center", fontsize=13,
                    color="white" if scores[i][j] > 0.5 else "black")
    ax.set_yticks(range(len(hidden)))
    ax.set_yticklabels(["emb"] + [f"L{i:02d}" for i in range(len(hidden) - 2)] + ["OUT"], fontsize=13)
    ax.set_xticks(range(TOPK))
    ax.set_xticklabels([f"#{i + 1}" for i in range(TOPK)], fontsize=13)
    ax.set_xlabel("Rank", fontsize=15)
    ax.set_ylabel("Layer", fontsize=15)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Probability", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"saved {out_pdf}  (question: {target['question']} | GT: {target['GT_entities']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["topk", "forced-prefix", "heatmap"])
    ap.add_argument("--universe", choices=UNIVERSES, required=True)
    ap.add_argument("--events", type=int, choices=EVENT_COUNTS, required=True)
    ap.add_argument("--lora", required=True, help="LoRA adapter folder of the SEQ-trained Llama-3-8B")
    ap.add_argument("--question", type=int, default=0, help="heatmap only: index among the two-entity questions")
    ap.add_argument("--out", default="mech_int_llama_heatmap.pdf", help="heatmap only: output PDF")
    args = ap.parse_args()

    benchmark = load_narrative(args.universe, args.events)
    questions = two_entity_questions(benchmark)
    print(f"{len(questions)} two-entity questions ({args.universe}, {args.events} events)")
    tokenizer, model = load_model(args.lora)

    if args.mode == "heatmap":
        run_heatmap(questions[args.question], benchmark, tokenizer, model, args.out)
        return
    sub = "topk_first_token" if args.mode == "topk" else "forced_prefix_continuation"
    out_dir = RESULTS_DIR / f"U{args.universe}_ev{args.events}" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_topk if args.mode == "topk" else run_forced_prefix)(questions, benchmark, tokenizer, model, out_dir)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
