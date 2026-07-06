"""
SimCLR_finetune_test.py  —  fine-tuning and evaluation loop.

All hyperparameters live in config.py.
External modules that are not part of this repo are marked with
# EXTERNAL MODULE — install from the original SimCLR repo.
"""

import os
import random
import warnings

import mlflow
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)
from tqdm import tqdm

from config import CFG
from simCLR.simCLR import SimCLR_Transformer
from build_dataset import CustomTensorDataset
from Dataset_STEAD import train_set, test_set, validation_set

from simCLR.module.logistic_regression import MLP_Classifier


warnings.filterwarnings("ignore")


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def _compute_metrics(cfg, y_true, logits):
    """
    y_true : 1-D LongTensor  [B]
    logits : 2-D FloatTensor [B, n_class]
    Returns dict of scalar metrics.
    """
    predicted  = logits.argmax(1)
    one_hot    = F.one_hot(y_true, num_classes=cfg["n_class"])

    y_np  = y_true.detach().numpy()
    p_np  = predicted.detach().numpy()
    oh_np = one_hot.detach().numpy()
    lg_np = logits.detach().numpy()

    try:
        auc = roc_auc_score(oh_np, lg_np, average="macro", multi_class="ovr")
    except ValueError:
        auc = 0.0

    return {
        "acc":       accuracy_score(y_np, p_np),
        "precision": precision_score(y_np, p_np, average="macro"),
        "recall":    recall_score(y_np, p_np, average="macro"),
        "f1":        f1_score(y_np, p_np, average="macro"),
        "auc":       auc,
        "prc":       average_precision_score(oh_np, lg_np, average="macro"),
    }


def _mean_metrics(records: list[dict]) -> dict:
    keys = records[0].keys()
    return {k: sum(r[k] for r in records) / len(records) for k in keys}


def finetune_epoch(cfg, loader, encoder, classifier, criterion, optimizer):
    encoder.train()
    classifier.train()

    losses, metrics_list = [], []

    # Wrap your loader with tqdm
    loop = tqdm(loader, desc="Finetune Epoch", leave=False)

    for x, _, y in loop:
        x = x.to(cfg["device"])
        y = y.squeeze(-1).long().to(cfg["device"])

        if CFG["fullfinetune"]:
            h, _, _, _ = encoder(x, x)
        else:
            with torch.no_grad():
                h, _, _, _ = encoder(x, x)

        logits = classifier(h)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        metrics_list.append(_compute_metrics(cfg, y.cpu(), logits.cpu()))

        # Update the progress bar with the current loss
        loop.set_postfix(loss=loss.item())

    return sum(losses) / len(losses), _mean_metrics(metrics_list)


def eval_epoch(cfg, loader, encoder, classifier, criterion):
    encoder.eval()
    classifier.eval()

    losses, metrics_list = [], []

    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(cfg["device"])
            y = y.squeeze(-1).long().to(cfg["device"])

            h, _, _, _ = encoder(x, x)
            logits = classifier(h)
            loss = criterion(logits, y)

            losses.append(loss.item())
            metrics_list.append(_compute_metrics(cfg, y.cpu(), logits.cpu()))

    return sum(losses) / len(losses), _mean_metrics(metrics_list)



def mainWmlflow():

    mlflow.set_tracking_uri(r"sqlite:///D:\Desktop\Intership IT\SSL&SeismicData\SSL_PT_FT_MLflow.db")
    mlflow.set_experiment("SSL_Finetuning_Label_Ratio_Importance")
    if mlflow.active_run():
        mlflow.end_run()

    # Start MLflow run for Fine-tuning
    with mlflow.start_run(run_name="1_FineTuning_Dummy"):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        CFG["device"] = device
        print(f"Using device: {device}")

        setup_seed(CFG["finetune_seed"])
        labelled_ratio = CFG["labelled_ratio"]

        train_X, train_Y = train_set()
        test_X, test_Y = test_set()
        val_X, val_Y = validation_set()

        # Log overarching dataset sizes
        mlflow.log_params({
            "original_train_size": len(train_X),
            "val_size": len(val_X),
            "test_size": len(test_X),
            "labelled_ratio": labelled_ratio,
            "finetune_mode": CFG["finetune_mode"],
            "logistic_epochs": CFG["logistic_epochs"],
            "logistic_batch_size": CFG["logistic_batch_size"]
        })

        # --- build a balanced, label-ratio-constrained training set ---
        fea, lab = [], []
        for i in range(CFG["n_class"]):
            mask = train_Y == i
            idx = np.where(mask)[0]
            take = max(1, int(len(idx) * labelled_ratio))
            fea.append(train_X[idx[:take]])
            lab.append(train_Y[idx[:take]])
            print(f"  class {i}: {(mask).sum()} total → {take} used")

        train_x = np.concatenate(fea)
        train_y = np.concatenate(lab)
        perm = np.random.permutation(len(train_y))
        train_x, train_y = train_x[perm], train_y[perm]
        print(f"Fine-tune set: {len(train_y)} samples  (ratio={labelled_ratio})")

        # Log the final constrained dataset size
        mlflow.log_param("constrained_train_size", len(train_y))

        def make_loader(x, y, shuffle=True):
            ds = CustomTensorDataset(data=(x, y))
            return torch.utils.data.DataLoader(
                ds,
                batch_size=CFG["logistic_batch_size"],
                shuffle=shuffle,
                drop_last=True,
            )

        train_loader = make_loader(train_x, train_y)
        val_loader = make_loader(val_X, val_Y, shuffle=False)
        test_loader = make_loader(test_X, test_Y, shuffle=False)

        # --- model ---
        encoder = SimCLR_Transformer(
            projection_dim=CFG["projection_dim"],
            n_channel=CFG["n_channel"],
            n_length=CFG["n_length"],
        )
        
        if CFG["pretraining"]:
            ckpt = os.path.join(
                CFG["model_path"],
                f"Pretrained_{CFG['dataset']}_{CFG['lr']}_{CFG['projection_dim']}.tar",
            )
            encoder.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"Loaded pretrained weights from {ckpt}")
            arch = CFG["dataset"]

        encoder = encoder.to(device)
        classifier = MLP_Classifier(encoder.n_features, CFG["n_class"]).to(device)

        if CFG["full_finetune"]:
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + list(classifier.parameters()), lr=3e-4
            )
        else:
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-4)

        criterion = torch.nn.CrossEntropyLoss()

        save_dir = CFG["save_dir"]
        model_ckpt = os.path.join(save_dir, f"{arch}_{labelled_ratio}_model.pt")
        clf_ckpt = os.path.join(save_dir, f"{arch}_{labelled_ratio}_classifier.pt")
        os.makedirs(save_dir, exist_ok=True)

        highest_f1 = 0.0

        for epoch in range(CFG["logistic_epochs"]):
            train_loss, train_m = finetune_epoch(
                CFG, train_loader, encoder, classifier, criterion, optimizer
            )
            val_loss, val_m = eval_epoch(CFG, val_loader, encoder, classifier, criterion)

            # Log training and validation metrics per epoch
            mlflow.log_metrics({
                "finetune_train_loss": train_loss,
                "finetune_val_loss": val_loss,
                "val_f1": val_m["f1"],
                "val_acc": val_m["acc"]
            }, step=epoch)

            if val_m["f1"] > highest_f1:
                highest_f1 = val_m["f1"]
                torch.save(encoder.state_dict(), model_ckpt)
                torch.save(classifier.state_dict(), clf_ckpt)
            print(f"  [epoch {epoch + 1}] ↑ val F1 {highest_f1:.4f}  — checkpoint saved")

            if epoch % 10 == 0:
                encoder.load_state_dict(torch.load(model_ckpt, map_location=device))
                classifier.load_state_dict(torch.load(clf_ckpt, map_location=device))

                test_loss, test_m = eval_epoch(CFG, test_loader, encoder, classifier, criterion)

                # Log periodic test metrics to MLflow
                mlflow.log_metrics({
                    "test_loss": test_loss,
                    "test_acc": test_m["acc"],
                    "test_f1": test_m["f1"],
                    "test_auc": test_m["auc"],
                    "test_prc": test_m["prc"]
                }, step=epoch)

                print(
                    f"Epoch [{epoch + 1}/{CFG['logistic_epochs']}]  "
                    f"test loss {test_loss:.4f}  "
                    f"acc {test_m['acc']:.4f}  f1 {test_m['f1']:.4f}  "
                    f"auc {test_m['auc']:.4f}  prc {test_m['prc']:.4f}"
                )

if __name__ == "__main__":
    mainWmlflow()