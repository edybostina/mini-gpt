import argparse
import torch

from .models import GPT, Config
from .data import DataLoader
from .train import train_loop, get_optimizer, save_checkpoint
from .utils import load_config, get_logger, get_optimal_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file (YAML/JSON)")
    parser.add_argument("--exp_name", type=str, default="default_exp",
                        help="Experiment name (for logs/checkpoints)")
    parser.add_argument("--input", type=str, default="./data/input.txt",
                        help="Training data (plain text)")
    parser.add_argument("--save_path", type=str, default="checkpoints/last.pt",
                        help="Where to save final checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")


    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda/mps). Defaults to auto-detect")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (micro_batch_size) from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")

    parser.add_argument("--log_interval", type=int, default=50,
                        help="Steps between logging")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Steps between evals")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra debug info")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["training"]["micro_batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["max_learning_rate"] = args.lr

    logger = get_logger(args.exp_name, verbose=args.verbose)

    torch.manual_seed(args.seed)
    device = args.device or get_optimal_device()
    logger.info(f"Using device: {device}, seed={args.seed}")

    model = GPT(Config(**cfg["model"]))
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    dataloader = DataLoader(
        logger,
        encoding=cfg["training"]["encoding"],
        seq_length=cfg["model"]["block_size"],
        batch_size=cfg["training"]["micro_batch_size"],
        filepath=args.input
    )

    optimizer = get_optimizer(
        model,
        logger,
        cfg["training"]["weight_decay"],
        cfg["training"]["max_learning_rate"],
        device=device
    )

    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "step" in checkpoint:
            start_step = checkpoint["step"]
        logger.info(f"Resumed from checkpoint {args.resume} at step {start_step}")

    train_loop(model, dataloader, optimizer, device, cfg["training"], logger, args.save_path, start_step=start_step)

    save_checkpoint(args.save_path, model, optimizer, cfg["training"]["max_steps"])
    logger.info(f"Training complete. Final checkpoint saved at {args.save_path}")

    print(model.generate_text(
        "Once upon a time",
        encoding=cfg["training"]["encoding"],
        max_new_tokens=60,
        temperature=0.9,
        top_p=0.95,
        device=device
    ))


if __name__ == "__main__":
    main()
