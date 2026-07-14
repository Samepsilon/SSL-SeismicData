import os
import math
import torch
from Config.Pretraining_Models_Config import simCLR


def load_optimizer( model: torch.nn.Module):
    """
    Build an optimizer (and optional LR scheduler) from the central config dict.

    Supported values of cfg["optimizer"]:
        "Adam"   — plain Adam, constant lr
        "AdamW"  — AdamW with cosine LR decay and linear warmup
    """
    scheduler = None

    if simCLR["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=simCLR["lr"])

    elif simCLR["optimizer"] == "AdamW":
        # Scale lr by batch size relative to reference batch of 256
        scaled_lr = simCLR["lr"] * simCLR["batch_size"] / 256
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=scaled_lr,
            betas=(0.9, 0.95),
            weight_decay=simCLR["weight_decay"],
        )
        # Cosine decay with linear warmup
        def lr_func(epoch):
            return min(
                (epoch + 1) / (simCLR["warmup_epoch"] + 1e-8),
                0.5 * (math.cos(epoch / simCLR["epochs"] * math.pi) + 1),
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    else:
        raise NotImplementedError(f"Unknown optimizer: {simCLR['optimizer']}")

    return optimizer, scheduler

