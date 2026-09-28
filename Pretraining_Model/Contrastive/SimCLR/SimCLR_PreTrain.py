import time

# mlflow
import mlflow
import numpy as np
import torch
from tqdm import tqdm

# function from other file
from Config.Pretraining_Models_Config import simCLR
from Config.dataset_config import extractedSTEAD
from Pretraining_Model.utility.Dataset.Dataset_STEAD import train_set
from SimCLR.build_dataset import CustomTensorDataset
from model import load_optimizer
from save_model_simCLR import save_model
from simCLR.module.nt_xent import NT_Xent
from simCLR.module.transformation.TS_transformation import CustomAugmentation
# module
from simCLR.simCLR import SimCLR_Transformer

"""
SimCLR pretraining implementation
SSL, self supervised learning,  

"""


def train_one_epoch(device, loader, model, criterion, optimizer):
    model.train()
    losses = []
    for step, (x_i, x_j, _) in enumerate(tqdm(loader, desc="pretraining")):
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        _, _, z_i, z_j = model(x_i, x_j)
        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  step [{step}/{len(loader)}]  loss {loss.item():.4f}")

        losses.append(loss.item())

    return sum(losses) / len(losses)


def mainWmlflow():
    """
    Implementation of the training loop for SimCLR, with mlflow logging
    """
    # URI for the sqlite database used by MLflow
    mlflow.set_tracking_uri(r"sqlite:///D:\Desktop\Intership IT\SSL&SeismicData\SSL_PT_FT_MLflow.db")
    # Name for the experiment in the MLflow system
    mlflow.set_experiment("SSL_Pretraining_SimCLR_Tuning_v2")

    early_stop_counter = 0

    # Start MLflow run for Pre-training, name of the run for mlflow system
    with mlflow.start_run(run_name="run45"):

        torch.manual_seed(simCLR["seed"])
        np.random.seed(simCLR["seed"])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # data loading
        train_x, train_y = train_set()

        # mlflow logging of training parameters
        mlflow.log_params({
            "seed": simCLR["seed"],
            "temperature": simCLR["temperature"],
            "epochs": simCLR["epochs"],
            "warmup_epoch": simCLR["warmup_epoch"],
        })

        # mlflow logging of SimCLR parameters
        mlflow.log_params({
            "batch_size": simCLR["batch_size"],
            "projection_dim": simCLR["projection_dim"],
            "n_layers": simCLR["n_layers"],
            "n_hid": simCLR["n_hid"],
            "n_head": simCLR["n_head"],
        })

        train_dataset = CustomTensorDataset(
            data=(train_x, train_y),
            transform_A=CustomAugmentation(simCLR["n_speed_change"], simCLR["max_speed_ratio"], simCLR["noise_scale"],
                                           simCLR["probability_for_dropout"]),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=simCLR["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        #  model creation
        model = SimCLR_Transformer(
            projection_dim=simCLR["projection_dim"],
            n_channel=extractedSTEAD["n_channel"],
            n_length=extractedSTEAD["n_length"],
            n_layers=simCLR["n_layers"],
            n_hid=simCLR["n_hid"],
            n_head=simCLR["n_head"],
        ).to(device)

        optimizer, scheduler = load_optimizer(model)
        criterion = NT_Xent(simCLR["batch_size"], simCLR["temperature"])

        # training loop
        print("Pretraining started.")
        lowest_loss = float("inf")

        for epoch in range(simCLR["epochs"]):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]

            mean_loss = train_one_epoch(device, train_loader, model, criterion, optimizer)  # loss calculation
            scheduler.step()

            # Log epoch metrics to MLflow
            mlflow.log_metrics({
                "mean_loss": mean_loss,
                "learning_rate": lr
            }, step=epoch)

            if mean_loss < lowest_loss:  # Model save if loss improve
                print(f"  ↓ loss improved {lowest_loss:.4f} → {mean_loss:.4f}  — saving model")
                save_model(model, "simCLR", extractedSTEAD["name"], simCLR["n_layers"], simCLR["n_hid"],
                           simCLR["n_head"], simCLR["projection_dim"])
                lowest_loss = mean_loss
                early_stop_counter = 0

            print(
                f"Epoch [{epoch + 1}/{simCLR['epochs']}]  "
                f"loss {mean_loss:.4f}  lr {lr:.2e}  "
                f"({time.time() - t0:.1f}s)"
            )
            # Early stopping
            if mean_loss == lowest_loss:
                if early_stop_counter == simCLR["early_stopping_patience"]:
                    print(f"Early stopping: Loss not improved for {simCLR["early_stopping_patience"]} step")
                    break
                else:
                    early_stop_counter += 1

        # Log the final best loss
        mlflow.log_metric("best_pretrain_loss", lowest_loss)


if __name__ == "__main__":
    mainWmlflow()
