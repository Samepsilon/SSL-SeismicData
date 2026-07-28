"""
implementing the ANN classifier for Contrastive view
"""

#when loading weight from pretrained, the last number on the file is the projection dimension for the encoder


import os
from pathlib import Path
import warnings
import copy

import mlflow
import numpy as np
import torch

from Config.Downstream_Task_Config import ANN
from Config.dataset_config import extractedSTEAD
from CustomANN.ANN_Model import CustomANN

from Pretraining_Model.Generative_Encoder.MAE.mae.MAE import MAE_ViT
from utility.build_dataset import CustomTensorDataset
from utility.encoder_loader import encoder_loader_path
from utility.Dataset.Dataset_STEAD import train_set, test_set, validation_set
from utility.model_MAE import setup_seed, finetune_epoch, eval_epoch

warnings.filterwarnings("ignore")




def mainWmlflow():
    mlflow.set_tracking_uri(r"sqlite:///D:\Desktop\Intership IT\SSL&SeismicData\SSL_PT_FT_MLflow.db")
    mlflow.set_experiment("SSL_MAE_ANN_Classifier_Experiment")
    if mlflow.active_run():
        mlflow.end_run()

    # Start MLflow run for Fine-tuning
    with mlflow.start_run(run_name="Pretraining_1.0_45"):

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
            "logistic_epochs": ANN["logistic_epochs"],
            "learning_rate": ANN["learning_rate"],
            "logistic_batch_size": ANN["logistic_batch_size"],
            "internal_structure": ANN["hidden_layer_layout"],
        })

        fea, lab = [], []
        rng = np.random.default_rng(ANN["finetune_seed"])
        for i in range(extractedSTEAD["n_class"]):
            mask = train_Y == i
            idx = np.where(mask)[0]
            rng.shuffle(idx)
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

        def make_loader(x, y, shuffle=True,drop_last=False):
            ds = CustomTensorDataset(data=(x, y))
            return torch.utils.data.DataLoader(
                ds,
                batch_size=ANN["logistic_batch_size"],
                shuffle=shuffle,
                drop_last=drop_last,
            )

        attention_head = 16
        decoder_layer = 2
        encoder_layer = 6
        emb_dim = 512
        patch_size = 100
        mask_ratio = 0.75

        train_loader = make_loader(train_x, train_y,drop_last=True)
        val_loader = make_loader(val_X, val_Y, shuffle=False)
        test_loader = make_loader(test_X, test_Y, shuffle=False)

        if ANN["pretrained"]:
            CKPT_PATH = encoder_loader_path()
            splited_info = Path(CKPT_PATH).stem.split('_')
            attention_head = int(splited_info[-1])
            decoder_layer = int(splited_info[-2])
            encoder_layer = int(splited_info[-3])
            emb_dim = int(splited_info[-4])
            patch_size = int(splited_info[-5])
            mask_ratio = float(splited_info[-6])

        # Log Hyperparameters to MLflow
        mlflow.log_params({
            "patch_size": patch_size,
            "mask_ratio": mask_ratio,
            "emb_dim": emb_dim,
            "encoder_layer": encoder_layer,
            "decoder_layer": decoder_layer,
            "encoder_head": attention_head,
            "decoder_head": attention_head
        })

        fullmodel = MAE_ViT(
            sample_shape=[extractedSTEAD["n_channel"], extractedSTEAD["n_length"]],
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            emb_dim=emb_dim,
            encoder_layer=encoder_layer,
            decoder_layer=decoder_layer,
            encoder_head=attention_head,
            decoder_head=attention_head,
        )

        if ANN["pretrained"]:
            fullmodel.load_state_dict(torch.load(CKPT_PATH, map_location=device))
            print(f"Loaded checkpoint: {CKPT_PATH}")

        encoder = fullmodel.encoder
        encoder = encoder.to(device)

        classifier = CustomANN(
            input_dim=emb_dim,
            num_classes=extractedSTEAD["n_class"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()), lr=ANN["learning_rate"]
        )

        criterion = torch.nn.CrossEntropyLoss()

        model_ckpt = os.path.join(ANN["SAVE_DIR"], f"ANN_{extractedSTEAD["name"]}_{ANN["labelled_ratio"]}_model.pt")
        clf_ckpt = os.path.join(ANN["SAVE_DIR"], f"ANN_{extractedSTEAD["name"]}_{ANN["labelled_ratio"]}_classifier.pt")
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
            }, step=epoch)

            if val_m["f1"] > highest_f1:
                highest_f1 = val_m["f1"]
                torch.save(encoder.state_dict(), model_ckpt)
                torch.save(classifier.state_dict(), clf_ckpt)
                print(f"  [epoch {epoch + 1}] ↑ val F1 {highest_f1:.4f}  — checkpoint saved")

            if epoch % 10 == 0:
                eval_encoder = copy.deepcopy(encoder)
                eval_classifier = copy.deepcopy(classifier)

                eval_encoder.load_state_dict(torch.load(model_ckpt, map_location=device))
                eval_classifier.load_state_dict(torch.load(clf_ckpt, map_location=device))

                test_loss, test_m = eval_epoch(device, test_loader, eval_encoder, eval_classifier, criterion)

                # Log periodic test metrics to MLflow
                mlflow.log_metrics({
                    "test accuracy": test_m["acc"],
                    "test precision": test_m["precision"],
                    "test recall": test_m["recall"],
                    "test f1 score": test_m["f1"],
                    "test ROC AUC": test_m["auc"],
                    "test average precision score": test_m["prc"],
                }, step=epoch)

                print(
                    f"Epoch [{epoch + 1}/{ANN['logistic_epochs']}]  "
                    f"test loss {test_loss:.4f}  "
                    f"acc {test_m['acc']:.4f}  f1 {test_m['f1']:.4f}  "
                    f"auc {test_m['auc']:.4f}  prc {test_m['prc']:.4f}"
                )


if __name__ == "__main__":
    mainWmlflow()