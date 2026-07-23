import argparse
import math
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import *
import torch.utils.data as Data
from build_dataset import CustomTensorDataset
import numpy as np

from mae.MAE import *
from save_model_MAE import save_model
import random
from Config.Pretraining_Models_Config import MAE
from Config.dataset_config import extractedSTEAD
from utility.Dataset.Dataset_STEAD import train_set

import mlflow
from tqdm import tqdm

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():

    setup_seed(MAE["seed"])

    batch_size = MAE["batch_size"]
    load_batch_size = min(MAE["max_device_batch_size"],MAE["batch_size"])

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # Load dataset
    print("Dataset:", extractedSTEAD["name"])
    dataset_name = globals()['dataset_' + extractedSTEAD["name"]]

    train_x, train_y = train_set()

    train_dataset = CustomTensorDataset(
        data=(train_x, train_y)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=load_batch_size,
        shuffle=True
    )

    model = MAE_ViT(
        sample_shape=[extractedSTEAD["n_channel"], extractedSTEAD["n_length"] ],
        patch_size=(extractedSTEAD["n_channel"]),
        mask_ratio=MAE["mask_ratio"]
    ).to(device)

    model = torch.compile(model)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=MAE["base_learning_rate"] * MAE["batch_size"] / 256,
        betas=(0.9, 0.95),
        weight_decay=MAE["weight_decay"]
    )

    lr_func = lambda epoch: min((epoch + 1) / (MAE["warmup_epoch"] + 1e-8),
                                0.5 * (math.cos(epoch / MAE["total_epoch"] * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)


    step_count = 0
    optim.zero_grad()
    min_loss = 100
    for e in range(MAE["total_epoch"]):
        model.train()
        print('===== start training ======')
        losses = []
        for sample, label in tqdm(iter(train_loader)):
            step_count += 1
            sample = sample.to(device)
            predicted_sample, mask = model(sample)
            loss = torch.mean((predicted_sample - sample) ** 2 * mask) / MAE["mask_ratio"]
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' save pre-trained model '''
        if avg_loss < min_loss:
            min_loss = avg_loss
            save_model(model,"MAE", extractedSTEAD["name"], MAE["mask_ratio"])
            print("Model update with loss {}.".format(min_loss))

def mainWmlflow():
    mlflow.set_tracking_uri(r"sqlite:///D:\Desktop\Intership IT\SSL&SeismicData\SSL_PT_FT_MLflow.db")
    mlflow.set_experiment("SSL_Pretraining_MAE_6*512_Tuning")

    with mlflow.start_run(run_name="Baseline_20_0.07 "):

        mlflow.log_params({
            "seed": MAE["seed"],
            "batch_size": MAE["batch_size"],
            "base_learning_rate": MAE["base_learning_rate"],
            "weight_decay": MAE["weight_decay"],
            "mask_ratio": MAE["mask_ratio"],
            "total_epoch": MAE["total_epoch"],
            "warmup_epoch": MAE["warmup_epoch"],
            "patch_size": MAE["patch_size"],
            "emb_dim": MAE["emb_dim"],
            "model_architecture": "MAE_Transformer",
        })

        setup_seed(MAE["seed"])

        batch_size = MAE["batch_size"]
        load_batch_size = min(MAE["max_device_batch_size"], MAE["batch_size"])

        assert batch_size % load_batch_size == 0
        steps_per_update = batch_size // load_batch_size

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device:", device)

        # Load dataset
        print("Dataset:", extractedSTEAD["name"])

        train_x, train_y = train_set()

        train_dataset = CustomTensorDataset(
            data=(train_x, train_y)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=load_batch_size,
            shuffle=True
        )

        model = MAE_ViT(
            sample_shape=[extractedSTEAD["n_channel"], extractedSTEAD["n_length"]],
            patch_size=MAE["patch_size"],
            mask_ratio=MAE["mask_ratio"],
            emb_dim=MAE["emb_dim"],
            encoder_layer=MAE["encoder_layer"],
            decoder_layer=MAE["decoder_layer"],
            encoder_head=MAE["attention_head"],
            decoder_head=MAE["attention_head"],
        ).to(device)


        optim = torch.optim.AdamW(
            model.parameters(),
            lr=MAE["base_learning_rate"] * MAE["batch_size"] / 256,
            betas=(0.9, 0.95),
            weight_decay=MAE["weight_decay"]
        )

        lr_func = lambda epoch: min((epoch + 1) / (MAE["warmup_epoch"] + 1e-8),
                                    0.5 * (math.cos(epoch / MAE["total_epoch"] * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

        step_count = 0
        optim.zero_grad()
        min_loss = 100
        for epoch in range(MAE["total_epoch"]):
            lr = optim.param_groups[0]["lr"]



            model.train()
            losses = []
            for sample, label in tqdm(iter(train_loader)):
                step_count += 1
                sample = sample.to(device)
                predicted_sample, mask = model(sample)
                loss = torch.mean((predicted_sample - sample) ** 2 * mask) / MAE["mask_ratio"]
                loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                losses.append(loss.item())
            lr_scheduler.step()
            avg_loss = sum(losses) / len(losses)
            print(f'In epoch {epoch}, average traning loss is {avg_loss}.')

            mlflow.log_metrics({
                "average_loss": avg_loss,
                "learning_rate": lr
            }, step=epoch)

            ''' save pre-trained model '''
            if avg_loss < min_loss:
                min_loss = avg_loss
                save_model(model, "MAE", extractedSTEAD["name"], MAE["mask_ratio"],MAE["patch_size"],MAE["emb_dim"],MAE["encoder_layer"],MAE["decoder_layer"],MAE["attention_head"])
                print("Model update with loss {}.".format(min_loss))

        mlflow.log_metric("best_pretrain_loss", min_loss)


if __name__ == '__main__':
    mainWmlflow()



