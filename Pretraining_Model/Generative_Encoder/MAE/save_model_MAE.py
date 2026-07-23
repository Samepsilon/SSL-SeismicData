import os
from pathlib import Path
import torch
import torchvision

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SAVE_DIR = BASE_DIR / "Models"



def save_model(
        model: torch.nn.Module,
        model_name : str,
        dataset_name : str ,
        mask_ratio : str ,
        patch_size : str ,
        emb_dim : int,
        encoder_layer : int,
        decoder_layer : int,
        attention_head : int,
):
    """Save encoder weights """
    os.makedirs(SAVE_DIR / model_name , exist_ok=True)
    out = os.path.join(
        SAVE_DIR / model_name / f"{model_name}_{dataset_name}_{mask_ratio}_{patch_size}_{emb_dim}_{encoder_layer}_{decoder_layer}_{attention_head}.tar",
    )
    state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state, out)
    print(f"Model saved → {out}")

if __name__ == '__main__':
    print(SAVE_DIR)


