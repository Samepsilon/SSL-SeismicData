from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR.parent /"Models"/"simCLR"
SSL = MODELS_DIR /"pretraining"
FINETUNE = MODELS_DIR /"finetunemodel"


CFG = {
    # simCLR config
    "n_layers" : 4,
    "n_hid"    : 1024,
    "n_head"   : 4,   # must divide n_length evenly

    #  dataset
    "dataset":          "STEAD",          # name used to build file paths / module names
    "n_class":          2,
    "n_channel":        3,              # number of sensor channels
    "n_length":         15000,            # timesteps per sample

    #  pretraining
    "seed":             42,
    "batch_size":       256,
    "epochs":           50,
    "optimizer":        "AdamW",        # "Adam" or "AdamW"
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "warmup_epoch":     10,
    "temperature":      0.2,
    "projection_dim":   256,
    "model_path":       SSL,
    "jittering_std" : 0.3,

    #  fine-tuning
    "finetune_seed":    42,
    "logistic_batch_size": 128,
    "logistic_epochs":  60,
    "labelled_ratio":   0.1,            # fraction of labelled train data to use (e.g. 0.1 = 10 %)
    "pretraining":         True,
    "finetune_mode": "Full",
    "save_dir": FINETUNE
}