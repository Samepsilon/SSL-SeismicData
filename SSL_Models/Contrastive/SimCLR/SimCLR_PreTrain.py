"""
SimCLR_pretrain.py  —  self-supervised pretraining loop.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

#function from other file
from config import CFG
from model import load_optimizer, save_model
from build_dataset import CustomTensorDataset
from Dataset_STEAD import train_set

#module
from simCLR.simCLR import SimCLR_Transformer
from simCLR.module.nt_xent import NT_Xent
from simCLR.module.transformation.TS_transformation import Jittering, Scaling

#mlflow
import mlflow




def train_one_epoch(cfg, loader, model, criterion, optimizer):
    model.train()
    losses = []
    for step, (x_i, x_j, _) in enumerate(tqdm(loader, desc="pretraining")):
        x_i = x_i.to(cfg["device"])
        x_j = x_j.to(cfg["device"])

        h_i, _, z_i, z_j = model(x_i, x_j)
        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  step [{step}/{len(loader)}]  loss {loss.item():.4f}")

        losses.append(loss.item())
    check_collapse(h_i.detach())



    return sum(losses) / len(losses)

def check_collapse(h: torch.Tensor, label: str = "encoder output") -> dict:
    """
    Checks whether the encoder has collapsed by measuring the variance
    of its output representations.

    A collapsed model produces near-identical embeddings for all inputs,
    meaning variance → 0 and all features become constant.

    Args:
        h     : encoder output [batch, n_features]
        label : string printed in the report (e.g. "epoch 5 / pretrain")

    Returns:
        dict of scalar metrics — also prints a human-readable report.
    """
    with torch.no_grad():
        # variance per feature dimension, averaged across the batch
        # shape: [n_features] → scalar
        per_feature_var  = h.var(dim=0)             # variance across samples
        mean_var         = per_feature_var.mean().item()
        min_var          = per_feature_var.min().item()
        dead_dims        = (per_feature_var < 1e-6).sum().item()
        total_dims       = h.shape[1]
        dead_pct         = 100 * dead_dims / total_dims

        # std of the per-feature variances — high std means some dims are
        # active, others dead (partial collapse)
        std_of_var       = per_feature_var.std().item()

        # cosine similarity between all pairs in the batch — collapse means
        # all pairs → 1.0
        h_norm           = F.normalize(h, dim=1)
        cos_sim_matrix   = h_norm @ h_norm.T
        # exclude diagonal (self-similarity = 1 always)
        mask             = ~torch.eye(h.shape[0], dtype=torch.bool, device=h.device)
        mean_cos_sim     = cos_sim_matrix[mask].mean().item()

        # collapse verdict
        if mean_var < 1e-4 or dead_pct > 90:
            status = "🔴 COLLAPSED — all embeddings are identical"
        elif dead_pct > 50 or mean_cos_sim > 0.95:
            status = "🟠 PARTIAL COLLAPSE — many dead dimensions"
        elif mean_var < 0.01 or mean_cos_sim > 0.85:
            status = "🟡 WARNING — low variance, monitor closely"
        else:
            status = "🟢 OK — representations look diverse"

        print(f"\n── Collapse check [{label}] ──────────────────────")
        print(f"  status           : {status}")
        print(f"  mean feature var : {mean_var:.6f}   (want >> 0)")
        print(f"  min feature var  : {min_var:.6f}   (want >> 0)")
        print(f"  dead dims        : {dead_dims}/{total_dims} ({dead_pct:.1f}%)  (want ~0%)")
        print(f"  std of var       : {std_of_var:.6f}  (very high = uneven dims)")
        print(f"  mean cosine sim  : {mean_cos_sim:.4f}    (want << 1.0)")
        print(f"────────────────────────────────────────────────────\n")

        return {
            "mean_var":      mean_var,
            "min_var":       min_var,
            "dead_dims":     dead_dims,
            "dead_pct":      dead_pct,
            "std_of_var":    std_of_var,
            "mean_cos_sim":  mean_cos_sim,
        }

def mainWmlflow():

    mlflow.set_tracking_uri(r"sqlite:///D:\Desktop\Intership IT\SSL&SeismicData\SSL_PT_FT_MLflow.db")
    mlflow.set_experiment("SSL_Pretraining_SimCLR_Label_Ratio_Importance")
    # Start MLflow run for Pre-training
    with mlflow.start_run(run_name="1_PreTraining_SimCLR"):

        torch.manual_seed(CFG["seed"])
        np.random.seed(CFG["seed"])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        CFG["device"] = device
        print(f"Using device: {device}")

        # --- data ---
        train_x, train_y = train_set()

        # Log Dataset Size & Parameters manually
        mlflow.log_params({"pretrain_dataset_size": len(train_y),"dataset_size": CFG["n_length"],"ratio":(len(train_y)/CFG["n_length"])})
        mlflow.log_params({
            "seed": CFG["seed"],
            "batch_size": CFG["batch_size"],
            "projection_dim": CFG["projection_dim"],
            "temperature": CFG["temperature"],
            "epochs": CFG["epochs"],
            "model_architecture": "SimCLR_Transformer",
            "jittering_std": CFG["jittering_std"],
        })

        train_dataset = CustomTensorDataset(
            data=(train_x, train_y),
            transform_A=Jittering(0, CFG["jittering_std"]),
        )
        mlflow.log_params({
            "seed": CFG["seed"],
            "batch_size": CFG["batch_size"],
            "projection_dim": CFG["projection_dim"],
        })

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

        mlflow.log_params({
            "n_layers": CFG["n_layers"],
            "n_hid": CFG["n_hid"],
            "n_head": CFG["n_head"],
        })

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

            # Log Epoch metrics to MLflow
            mlflow.log_metrics({
                "mean_loss": mean_loss,
                "learning_rate": lr
            }, step=epoch)

            if mean_loss < lowest_loss:
                print(f"  ↓ loss improved {lowest_loss:.4f} → {mean_loss:.4f}  — saving model")
                save_model(CFG, model)
                lowest_loss = mean_loss

            print(
                f"Epoch [{epoch + 1}/{CFG['epochs']}]  "
                f"loss {mean_loss:.4f}  lr {lr:.2e}  "
                f"({time.time() - t0:.1f}s)"
            )

        # Log the final best loss
        mlflow.log_metric("best_pretrain_loss", lowest_loss)


if __name__ == "__main__":
    mainWmlflow()