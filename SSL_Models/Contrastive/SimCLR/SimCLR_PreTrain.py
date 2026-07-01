"""
SimCLR_pretrain.py  —  self-supervised pretraining loop.
"""

import time
import numpy as np
import torch
from tqdm import tqdm

#function from other file
from config import CFG
from model import load_optimizer, save_model
from build_dataset import CustomTensorDataset
from Dataset_STEAD import train_set

#module
from simCLR.simCLR import SimCLR_Transformer
from simCLR.module.nt_xent import NT_Xent
from simCLR.module.transformation.TS_transformation import Jittering



def train_one_epoch(cfg, loader, model, criterion, optimizer):
    model.train()
    losses = []
    for step, (x_i, x_j, _) in enumerate(tqdm(loader, desc="SSL")):
        x_i = x_i.to(cfg["device"])
        x_j = x_j.to(cfg["device"])

        _, _, z_i, z_j = model(x_i, x_j)
        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  step [{step}/{len(loader)}]  loss {loss.item():.4f}")

        losses.append(loss.item())

    return sum(losses) / len(losses)


def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CFG["device"] = device
    print(f"Using device: {device}")

    # --- data ---
    train_x, train_y = train_set()
    train_dataset = CustomTensorDataset(
        data=(train_x, train_y),
        transform_A=Jittering(0, 0.1),
        # transform_A=Scaling(),
        # transform_B=Flipping(),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # --- model ---
    model = SimCLR_Transformer(
        projection_dim=CFG["projection_dim"],
        n_channel=CFG["n_channel"],
        n_length=CFG["n_length"],
    ).to(device)

    optimizer, scheduler = load_optimizer(CFG, model)
    criterion = NT_Xent(CFG["batch_size"], CFG["temperature"])

    # --- training loop ---
    print("Pretraining started.")
    lowest_loss = float("inf")

    for epoch in range(CFG["epochs"]):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        mean_loss = train_one_epoch(CFG, train_loader, model, criterion, optimizer)
        scheduler.step()

        if mean_loss < lowest_loss:
            print(f"  ↓ loss improved {lowest_loss:.4f} → {mean_loss:.4f}  — saving model")
            save_model(CFG, model)
            lowest_loss = mean_loss

        print(
            f"Epoch [{epoch+1}/{CFG['epochs']}]  "
            f"loss {mean_loss:.4f}  lr {lr:.2e}  "
            f"({time.time()-t0:.1f}s)"
        )


if __name__ == "__main__":
    main()