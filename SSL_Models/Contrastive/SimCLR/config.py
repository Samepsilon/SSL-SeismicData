from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR.parent /"Models"
SSL = MODELS_DIR /"SSL"
FINETUNE = MODELS_DIR /"finetunemodel"


CFG = {
    # --- dataset ---
    "dataset":          "STEAD",          # name used to build file paths / module names
    "n_class":          2,
    "n_channel":        3,              # number of sensor channels
    "n_length":         6000,            # timesteps per sample

    # --- pretraining ---
    "seed":             42,
    "batch_size":       128,
    "epochs":           60,
    "optimizer":        "AdamW",        # "Adam" or "AdamW"
    "lr":               3e-4,
    "weight_decay":     1e-4,
    "warmup_epoch":     10,
    "temperature":      0.5,
    "projection_dim":   128,
    "model_path":       SSL,

    # --- fine-tuning ---
    "finetune_seed":    42,
    "logistic_batch_size": 64,
    "logistic_epochs":  50,
    "labelled_ratio":   0.1,            # fraction of labelled train data to use (e.g. 0.1 = 10 %)
    "SSL":         True,
    "finetune_mode": "Full",
    "save_dir": FINETUNE
}