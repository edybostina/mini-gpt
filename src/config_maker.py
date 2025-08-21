#!/usr/bin/env python3
import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "model": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "vocab_size": 50257,
        "block_size": 1024,
    },
    "training": {
        "encoding": "gpt2",
        "batch_size": 524288,
        "micro_batch_size": 1,
        "seq_length": 1024,
        "nr_return_sequences": 5,
        "max_length": 30,
        "max_learning_rate": 0.0006,
        "min_learning_rate": 0.00006,
        "warmup_steps": 10,
        "max_steps": 1000,
        "seed": 69,
        "weight_decay": 0.1,
        "eval_interval": 50,
        "eval_iters": 50,
        "deterministic": False,
    }
}


def ask(prompt, default=None, cast=str):
    if default is not None:
        q = f"{prompt} [{default}]: "
    else:
        q = f"{prompt}: "
    val = input(q).strip()
    if not val:
        return default
    try:
        return cast(val)
    except Exception:
        print(f"Invalid input, using default {default}")
        return default


def main():
    print("=== GPT Config Maker ===")
    print("Press ENTER to accept defaults, or type a new value.\n")

    cfg = DEFAULT_CONFIG.copy()

    print("Model configuration:")
    cfg["model"]["n_embd"] = ask("Embedding dimension", cfg["model"]["n_embd"], int)
    cfg["model"]["n_layer"] = ask("Number of layers", cfg["model"]["n_layer"], int)
    cfg["model"]["n_head"] = ask("Number of attention heads", cfg["model"]["n_head"], int)
    cfg["model"]["vocab_size"] = ask("Vocab size", cfg["model"]["vocab_size"], int)
    cfg["model"]["block_size"] = ask("Context length (block_size)", cfg["model"]["block_size"], int)

    print("\nTraining configuration:")
    cfg["training"]["batch_size"] = ask("Global batch size", cfg["training"]["batch_size"], int)
    cfg["training"]["micro_batch_size"] = ask("Micro batch size", cfg["training"]["micro_batch_size"], int)
    cfg["training"]["seq_length"] = ask("Sequence length", cfg["training"]["seq_length"], int)
    cfg["training"]["max_steps"] = ask("Max training steps", cfg["training"]["max_steps"], int)
    cfg["training"]["max_learning_rate"] = ask("Max LR", cfg["training"]["max_learning_rate"], float)
    cfg["training"]["min_learning_rate"] = ask("Min LR", cfg["training"]["min_learning_rate"], float)
    cfg["training"]["warmup_steps"] = ask("Warmup steps", cfg["training"]["warmup_steps"], int)
    cfg["training"]["weight_decay"] = ask("Weight decay", cfg["training"]["weight_decay"], float)
    cfg["training"]["seed"] = ask("Random seed", cfg["training"]["seed"], int)
    cfg["training"]["eval_interval"] = ask("Eval interval (steps)", cfg["training"]["eval_interval"], int)
    cfg["training"]["eval_iters"] = ask("Eval iters", cfg["training"]["eval_iters"], int)
    cfg["training"]["deterministic"] = ask("Deterministic", cfg["training"]["deterministic"], bool)

    grad_accum = cfg["training"]["batch_size"] // (cfg["training"]["micro_batch_size"] * cfg["training"]["seq_length"])
    print(f"\nGradient accumulation steps = {grad_accum}")

    out_path = ask("\nSave config to file", "configs/model/custom.yaml", str)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"\nConfig saved to {out_path}")


if __name__ == "__main__":
    main()
