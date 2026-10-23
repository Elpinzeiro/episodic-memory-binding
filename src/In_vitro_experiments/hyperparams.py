"""
Fine-tuning hyperparameters of the paper.

Llama: Tab. 18 (grid search on Sequential-SEQ, App. D.2), kept fixed across all conditions.
Qwen2.5 and Mistral: 40 epochs (60 for 100-event narratives), lr 1e-5, batch size 16.
LoRA r = 32, alpha = 64 on all attention and MLP projections (App. D.1).
"""

# (events, Llama size) -> hyperparameters
BEST_HYPERPARAMS = {
    (10, "3B"): {"epochs": 20, "lr": 1e-4, "bs": 16},
    (10, "8B"): {"epochs": 40, "lr": 1e-5, "bs": 16},
    (10, "13B"): {"epochs": 40, "lr": 1e-5, "bs": 8},
    (10, "70B"): {"epochs": 20, "lr": 1e-5, "bs": 4},
    (30, "3B"): {"epochs": 40, "lr": 1e-4, "bs": 16},
    (30, "8B"): {"epochs": 40, "lr": 1e-5, "bs": 16},
    (30, "13B"): {"epochs": 20, "lr": 1e-5, "bs": 8},
    (30, "70B"): {"epochs": 40, "lr": 1e-5, "bs": 1},
    (100, "3B"): {"epochs": 40, "lr": 1e-5, "bs": 16},
    (100, "8B"): {"epochs": 40, "lr": 1e-5, "bs": 16},
    (100, "13B"): {"epochs": 40, "lr": 1e-5, "bs": 8},
    (100, "70B"): {"epochs": 20, "lr": 1e-5, "bs": 1},
}
LLAMA_SIZE = {
    "meta-llama/Llama-3.2-3B-Instruct": "3B",
    "meta-llama/Meta-Llama-3-8B-Instruct": "8B",
    "meta-llama/Llama-2-13b-chat-hf": "13B",
    "meta-llama/Llama-3.3-70B-Instruct": "70B",
}


def hyperparams_for(model, nb_events, size=None):
    """Hyperparameters of `model` on a narrative of `nb_events` events (size: force a Tab. 18 row)."""
    size = size or LLAMA_SIZE.get(model)
    if size is not None:
        return dict(BEST_HYPERPARAMS[(nb_events, size)])
    if any(name in model.lower() for name in ("qwen", "mistral")):
        return {"epochs": 60 if nb_events == 100 else 40, "lr": 1e-5, "bs": 16}
    raise ValueError(f"no hyperparameters known for {model}: pass --size (Tab. 18 row) or --epochs/--lr/--bs")


def generator_for(model):
    """Narrative set a model was trained on: Qwen and Mistral used the Claude Sonnet 4.6 narratives."""
    return "claude-sonnet-4-6" if any(n in model.lower() for n in ("qwen", "mistral")) else "claude-3-5-sonnet-20240620"
