from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR / "Models"

ANN = {
    "hidden_layer_layout":   [256,128,64,32],          # name used to build file paths / module names

    #  fine-tuning
    "finetune_seed": 42,
    "logistic_batch_size": 256,
    "logistic_epochs": 200,
    "learning_rate": 3e-4,
    "labelled_ratio": 0.3,  # fraction of labelled train data to use (e.g. 0.1 = 10 %)
    "pretrained": False,
    "SAVE_DIR" : SAVE_DIR / "ANN"
}

MLP = {
    #  fine-tuning
    "finetune_seed": 42,
    "logistic_batch_size": 64,
    "logistic_epochs": 200,
    "learning_rate": 3e-4,
    "labelled_ratio": 0.1,  # fraction of labelled train data to use (e.g. 0.1 = 10 %)
    "full_finetune" : True,
    "pretrained": True,
    "SAVE_DIR" : SAVE_DIR / "MLP"
}

