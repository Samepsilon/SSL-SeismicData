import os
import random
import warnings

import mlflow
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)
from tqdm import tqdm

from Config.Downstream_Task_Config import ANN
from Config.dataset_config import extractedSTEAD

def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def _compute_metrics(n_class, y_true, logits):
    """
    y_true : 1-D LongTensor  [B]
    logits : 2-D FloatTensor [B, n_class]
    Returns dict of scalar metrics.
    """
    predicted  = logits.argmax(1)
    one_hot    = F.one_hot(y_true, num_classes=n_class)

    y_np  = y_true.detach().numpy()
    p_np  = predicted.detach().numpy()
    oh_np = one_hot.detach().numpy()
    lg_np = logits.detach().numpy()

    try:
        auc = roc_auc_score(oh_np, lg_np, average="macro", multi_class="ovr")
    except ValueError:
        auc = 0.0

    return {
        "acc":       accuracy_score(y_np, p_np),
        "precision": precision_score(y_np, p_np, average="macro"),
        "recall":    recall_score(y_np, p_np, average="macro"),
        "f1":        f1_score(y_np, p_np, average="macro"),
        "auc":       auc,
        "prc":       average_precision_score(oh_np, lg_np, average="macro"),
    }

def _mean_metrics(records: list[dict]) -> dict:
    keys = records[0].keys()
    return {k: sum(r[k] for r in records) / len(records) for k in keys}



def finetune_epoch(device, loader, encoder, classifier, criterion, optimizer):
    encoder.train()
    classifier.train()

    losses, metrics_list = [], []

    # Wrap your loader with tqdm
    loop = tqdm(loader, desc="Finetune Epoch", leave=False)

    for x, _, y in loop:
        x = x.to(device)
        y = y.squeeze(-1).long().to(device)

        h, _, _, _ = encoder(x, x)

        logits = classifier(h)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(classifier.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        losses.append(loss.item())
        metrics_list.append(_compute_metrics(extractedSTEAD["n_class"], y.cpu(), logits.cpu()))

        # Update the progress bar with the current loss
        loop.set_postfix(loss=loss.item())

    return sum(losses) / len(losses), _mean_metrics(metrics_list)

def eval_epoch(device, loader, encoder, classifier, criterion):
    encoder.eval()
    classifier.eval()

    losses, metrics_list = [], []

    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(device)
            y = y.squeeze(-1).long().to(device)

            h, _, _, _ = encoder(x, x)
            logits = classifier(h)
            loss = criterion(logits, y)

            losses.append(loss.item())
            metrics_list.append(_compute_metrics(extractedSTEAD["n_class"], y.cpu(), logits.cpu()))

    return sum(losses) / len(losses), _mean_metrics(metrics_list)