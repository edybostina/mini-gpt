from models import GPT, Config
from data import DataLoader
from train import train_loop, get_optimizer, save_checkpoint
from utils import load_config, get_logger, get_optimal_device

import torch

def main():
    # Load config (YAML/JSON)
    cfg = load_config("../configs/model/gpt2-small.yaml")

    # Logger
    logger = get_logger("train")

    # Model
    model = GPT(Config(**cfg["model"]))
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Data
    #dataset = OpenWebTextDataset(cfg["data"]["path"], block_size=cfg["model"]["block_size"])
    dataloader = DataLoader(logger, seq_length=cfg["model"]["block_size"], batch_size=cfg["training"]["micro_batch_size"], filepath="../input.txt")
    device = get_optimal_device()
    logger.info(f"Using device: {device}")
    # Optimizer
    optimizer = get_optimizer(model, logger, cfg["training"]["weight_decay"], cfg["training"]["max_learning_rate"], device=device)

    # Train
    train_loop(model, dataloader, optimizer, device, cfg["training"], logger)

    # Save
    save_checkpoint(model, optimizer, "checkpoints/last.pt")
    logger.info("Training complete.")

if __name__ == "__main__":
    main()