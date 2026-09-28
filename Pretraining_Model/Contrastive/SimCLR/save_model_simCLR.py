import os
from pathlib import Path

import torch

# Directory config to save the models
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SAVE_DIR = BASE_DIR / "Models"


def save_model(
        model: torch.nn.Module,
        model_name: str,
        dataset_name: str,
        n_layer: str,
        n_hidden: str,
        n_head: str,
        projection_dim: str,
):
    """
    Save encoder weights at designated location
    Take in argument different value of configuration for continuity in the fine-tuning phase

    :param model: an instance of the SimCLR model (torch.nn.Module)
    :param model_name: the name of the model
    :param dataset_name: the name of the dataset
    :param n_layer: the number of layers for the encoder
    :param n_hidden:  the number of hidden dimension inside each layer
    :param n_head: the number of attention heads (transformer architecture)
    :param projection_dim: the size of the projection dimension for the MLP projector inside the model

    """
    os.makedirs(SAVE_DIR / model_name, exist_ok=True)
    out = os.path.join(
        SAVE_DIR / model_name / f"{model_name}_{dataset_name}_{n_layer}_{n_hidden}_{n_head}_{projection_dim}.tar",
    )
    state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state, out)
    print(f"Model saved → {out}")


if __name__ == '__main__':
    print(SAVE_DIR)
