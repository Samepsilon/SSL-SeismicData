CFG = {
    # simCLR config
    "n_layers" : 8,
    "n_hid"    : 1024,
    "n_head"   : 8,   # must divide sample evenly

    #  pretraining
    "seed":             42,
    "batch_size":       512,
    "epochs":           200,
    "optimizer":        "AdamW",        # "Adam" or "AdamW"
    "lr":               3e-4,
    "weight_decay":     1e-4,
    "warmup_epoch":     25,
    "temperature":      0.2,
    "projection_dim":   256,
    "early_stopping_patience": 5,

    #custom augmenter parameter
    "n_speed_change": 4,
    "max_speed_ratio": 2.0,
    "noise_scale": 0.01,
    "probaility_for_dropout": 0.1,
}