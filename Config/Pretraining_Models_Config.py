simCLR = {
    # simCLR config
    "n_layers": 6,
    "n_hid": 512,
    "n_head": 16,  # must divide the number of sample evenly

    #  pretraining
    "seed": 45,
    "batch_size": 512,
    "epochs": 50,
    "optimizer": "AdamW",  # "Adam" or "AdamW"
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epoch": 10,
    "temperature": 0.5,
    "projection_dim": 512,
    "early_stopping_patience": 5,

    # custom augmenter parameter
    "n_speed_change": 1,
    "max_speed_ratio": 2.0,
    "noise_scale": 0.02,
}

MAE = {
    # MAE Config
    "emb_dim": 512,
    "encoder_layer": 6,
    "attention_head": 16,
    "decoder_layer": 2,

    #  pretraining
    "seed": 45,
    "batch_size": 512,
    "patch_size": 100,
    "max_device_batch_size": 512,
    "base_learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "mask_ratio": 0.75,
    "total_epoch": 200,
    "warmup_epoch": 25,
}
