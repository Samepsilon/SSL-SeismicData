simCLR = {
    # simCLR config
    "n_layers" : 6,
    "n_hid"    : 512,
    "n_head"   : 16,   # must divide the number of sample evenly

    #  pretraining
    "seed":             42,
    "batch_size":       512,
    "epochs":           50,
    "optimizer":        "AdamW",        # "Adam" or "AdamW"
    "lr":               3e-4,
    "weight_decay":     1e-4,
    "warmup_epoch":     5,
    "temperature":      0.2,
    "projection_dim":   512,
    "early_stopping_patience": 5,

    #custom augmenter parameter
    "n_speed_change": 4,
    "max_speed_ratio": 2.0,
    "noise_scale": 0.01,
    "probability_for_dropout": 0.1,
}

MAE = {
    #simCLR Config
    "emb_dim":512,
    "encoder_layer":8,
    "attention_head": 16,
    "decoder_layer":2,


    #  pretraining
    "seed" : 42,
    "batch_size" : 512,
    "patch_size" : 20,
    "max_device_batch_size" : 512,
    "base_learning_rate" : 3e-4,
    "weight_decay": 0.05,
    "mask_ratio": 0.1,
    "total_epoch": 50,
    "warmup_epoch": 5,
}