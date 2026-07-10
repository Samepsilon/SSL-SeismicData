"""
SimCLR_finetune_test.py  —  fine-tuning and evaluation loop.

All hyperparameters live in config.py.
External modules that are not part of this repo are marked with
# EXTERNAL MODULE — install from the original SimCLR repo.
"""

import os
from pathlib import Path
import warnings

import mlflow
import numpy as np
import torch


from Config.Downstream_Task_Config import ANN
from Config.dataset_config import extractedSTEAD
from MultiLayerPerceptron.logistic_regression import MLP_Classifier

from simCLR.simCLR import SimCLR_Transformer
from utility.build_dataset import CustomTensorDataset
from utility.encoder_loader import encoder_loader_path
from utility.Dataset.Dataset_STEAD import train_set, test_set, validation_set
from utility.model import setup_seed, finetune_epoch, eval_epoch

warnings.filterwarnings("ignore")





def mainWmlflow():
    mlflow.set_tracking_uri(r"sqlite:///D:\Desktop\Intership IT\SSL&SeismicData\SSL_PT_FT_MLflow.db")
    mlflow.set_experiment("SSL_Finetuning_Label_Ratio_Importance")
    if mlflow.active_run():
        mlflow.end_run()

    # Start MLflow run for Fine-tuning
    with mlflow.start_run(run_name="1_Test_w_DummyClassifier"):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        setup_seed(ANN["finetune_seed"])

        train_X, train_Y = train_set()
        test_X, test_Y = test_set()
        val_X, val_Y = validation_set()

        # Log overarching dataset sizes
        mlflow.log_params({
            "original_train_size": len(train_X),
            "val_size": len(val_X),
            "test_size": len(test_X),
            "labelled_ratio": ANN["labelled_ratio"],
            "full_finetune": ANN["full_finetune"],
            "logistic_epochs": ANN["logistic_epochs"],
            "learning_rate": ANN["learning_rate"],
            "logistic_batch_size": ANN["logistic_batch_size"]
        })

        # --- build a balanced, label-ratio-constrained training set ---
        fea, lab = [], []
        for i in range(extractedSTEAD["n_class"]):
            mask = train_Y == i
            idx = np.where(mask)[0]
            take = max(1, int(len(idx) * ANN["labelled_ratio"]))
            fea.append(train_X[idx[:take]])
            lab.append(train_Y[idx[:take]])
            print(f"  class {i}: {(mask).sum()} total → {take} used")

        train_x = np.concatenate(fea)
        train_y = np.concatenate(lab)
        perm = np.random.permutation(len(train_y))
        train_x, train_y = train_x[perm], train_y[perm]
        print(f"Fine-tune set: {len(train_y)} samples  (ratio={ANN["labelled_ratio"]})")

        # Log the final constrained dataset size
        mlflow.log_param("constrained_train_size", len(train_y))

        def make_loader(x, y, shuffle=True):
            ds = CustomTensorDataset(data=(x, y))
            return torch.utils.data.DataLoader(
                ds,
                batch_size=ANN["logistic_batch_size"],
                shuffle=shuffle,
                drop_last=True,
            )

        train_loader = make_loader(train_x, train_y)
        val_loader = make_loader(val_X, val_Y, shuffle=False)
        test_loader = make_loader(test_X, test_Y, shuffle=False)

        if ANN["pretrained"]:
            ckpt = encoder_loader_path()
            projection_dim = Path(ckpt).stem.split('_')[-1]
            print(projection_dim)
            print(f"Loaded pretrained weights from {ckpt}")


        # --- model ---
        encoder = SimCLR_Transformer(
            projection_dim=projection_dim,
            n_channel=extractedSTEAD["n_channel"],
            n_length=extractedSTEAD["n_length"],
        )

        if ANN["pretrained"]:
            encoder.load_state_dict(torch.load(ckpt, map_location=device))



        encoder = encoder.to(device)
        classifier = MLP_Classifier(encoder.n_features, extractedSTEAD["n_class"]).to(device)

        if ANN["full_finetune"]:
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + list(classifier.parameters()), lr=ANN["learning_rate"]
            )
        else:
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=ANN["learning_rate"])

        criterion = torch.nn.CrossEntropyLoss()

        model_ckpt = os.path.join(ANN["SAVE_DIR"], f"{extractedSTEAD["name"]}_{ANN["labelled_ratio"]}_model.pt")
        clf_ckpt = os.path.join(ANN["SAVE_DIR"], f"{extractedSTEAD["name"]}_{ANN["labelled_ratio"]}_classifier.pt")
        os.makedirs(ANN["SAVE_DIR"], exist_ok=True)

        highest_f1 = 0.0

        for epoch in range(ANN["logistic_epochs"]):
            train_loss, train_m = finetune_epoch(
                device, train_loader, encoder, classifier, criterion, optimizer
            )
            val_loss, val_m = eval_epoch(device, val_loader, encoder, classifier, criterion)

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

                test_loss, test_m = eval_epoch(device, test_loader, encoder, classifier, criterion)

                # Log periodic test metrics to MLflow
                mlflow.log_metrics({
                    "test_loss": test_loss,
                    "test_acc": test_m["acc"],
                    "test_f1": test_m["f1"],
                    "test_auc": test_m["auc"],
                    "test_prc": test_m["prc"]
                }, step=epoch)

                print(
                    f"Epoch [{epoch + 1}/{ANN['logistic_epochs']}]  "
                    f"test loss {test_loss:.4f}  "
                    f"acc {test_m['acc']:.4f}  f1 {test_m['f1']:.4f}  "
                    f"auc {test_m['auc']:.4f}  prc {test_m['prc']:.4f}"
                )

if __name__ == "__main__":
    mainWmlflow()