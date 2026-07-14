"""
clustering_analysis.py  —  evaluate SimCLR pretrained encoder via clustering.

Loads a dataset (X, y), extracts encoder representations h, then:
  1. Clusters h with K-Means (k = n_class)
  2. Evaluates cluster quality with unsupervised + supervised metrics
  3. Plots UMAP / t-SNE colored by true label and by cluster assignment
  4. Plots a confusion matrix between clusters and true labels

Usage:
    python clustering_analysis.py

All paths and hyperparameters are set in the CONFIG block below.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import umap

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

from Config.dataset_config import extractedSTEAD
from simCLR.simCLR  import SimCLR_Transformer
from utility.Dataset.Dataset_STEAD import test_set

from sklearn.manifold import TSNE

from utility.encoder_loader import encoder_loader_path

# CONFIG — edit these to point at your data and model checkpoint


BATCH_SIZE  = 128
N_CLASS     = extractedSTEAD["n_class"]     # 2 for your seismic dataset
SAVE_DIR    = "plots_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# Helpers

def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_class: int) -> float:
    """
    Match cluster IDs to true labels optimally via the Hungarian algorithm,
    then compute accuracy. Needed because K-Means cluster indices are arbitrary.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_class)))
    row_ind, col_ind = linear_sum_assignment(-cm)   # maximise match
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_remapped = np.array([mapping.get(c, c) for c in y_pred])
    return accuracy_score(y_true, y_remapped), y_remapped


def extract_embeddings(model, X: np.ndarray, device, batch_size: int) -> np.ndarray:
    """Run encoder on X in batches, return h [N, n_features]."""
    model.eval()
    embeddings = []
    n = X.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.tensor(
                X[start:start + batch_size], dtype=torch.float32
            ).to(device)
            h, _, _, _ = model(batch, batch)
            embeddings.append(h.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def main():

    # Load data
    X, y = test_set()

    # Load pretrained encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    CKPT_PATH = encoder_loader_path()
    projection_dim = int(Path(CKPT_PATH).stem.split('_')[-1])
    print(projection_dim)

    encoder = SimCLR_Transformer(
        projection_dim=projection_dim,
        n_channel=extractedSTEAD["n_channel"],
        n_length=extractedSTEAD["n_length"],
    )



    encoder.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    encoder = encoder.to(device)
    print(f"Loaded checkpoint: {CKPT_PATH}")

    # Extract representations
    h = extract_embeddings(encoder, X, device, BATCH_SIZE)

    # L2-normalise before clustering (standard practice for contrastive models)
    h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)

    # K-Means clustering
    kmeans = KMeans(n_clusters=N_CLASS, n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(h_norm)

    # Metrics
    acc, cluster_labels_remapped = hungarian_accuracy(y, cluster_labels, N_CLASS)
    ari = adjusted_rand_score(y, cluster_labels)
    nmi = normalized_mutual_info_score(y, cluster_labels)
    sil = silhouette_score(h_norm, cluster_labels, sample_size=min(5000, len(y)))

    print("\n Clustering metrics")
    print(f"  Clustering accuracy (Hungarian): {acc:.4f}   (random = {1 / N_CLASS:.4f})")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}   (random ≈ 0, perfect = 1)")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}   (random ≈ 0, perfect = 1)")
    print(f"  Silhouette score: {sil:.4f}   (range -1 → 1, higher = better)")

    # 2-D projections (UMAP + t-SNE)

    coords = {}

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    coords["UMAP"] = reducer.fit_transform(h_norm)

    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
    coords["t-SNE"] = tsne.fit_transform(h_norm)

    # Plotting
    n_methods = len(coords)
    class_cmap = plt.get_cmap("tab10", N_CLASS)

    fig = plt.figure(figsize=(7 * n_methods, 12))
    gs = gridspec.GridSpec(3, n_methods, figure=fig, hspace=0.4, wspace=0.3)

    for col, (method_name, xy) in enumerate(coords.items()):

        # ── row 0: coloured by true label ──
        ax = fig.add_subplot(gs[0, col])
        for cls in range(N_CLASS):
            mask = y == cls
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=[class_cmap(cls)], label=f"class {cls}",
                       s=6, alpha=0.6, linewidths=0)
        ax.set_title(f"{method_name} — true labels", fontsize=11)
        ax.legend(markerscale=2, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # ── row 1: coloured by cluster assignment ──
        ax = fig.add_subplot(gs[1, col])
        for clu in range(N_CLASS):
            mask = cluster_labels == clu
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=[class_cmap(clu)], label=f"cluster {clu}",
                       s=6, alpha=0.6, linewidths=0)
        ax.set_title(f"{method_name} — K-Means clusters", fontsize=11)
        ax.legend(markerscale=2, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # ── row 2: confusion matrix (clusters vs true labels, after Hungarian matching) ──
    ax_cm = fig.add_subplot(gs[2, :])
    cm = confusion_matrix(y, cluster_labels_remapped, labels=list(range(N_CLASS)))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax_cm, fraction=0.03)
    ax_cm.set_xlabel("Predicted cluster (remapped)", fontsize=10)
    ax_cm.set_ylabel("True label", fontsize=10)
    ax_cm.set_title(
        f"Cluster vs true label  |  acc {acc:.3f}  ARI {ari:.3f}  NMI {nmi:.3f}  sil {sil:.3f}",
        fontsize=11
    )
    ax_cm.set_xticks(range(N_CLASS))
    ax_cm.set_yticks(range(N_CLASS))
    for i in range(N_CLASS):
        for j in range(N_CLASS):
            ax_cm.text(j, i, str(cm[i, j]),
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontsize=10)


    out_path = os.path.join(SAVE_DIR, "clustering_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    for cls in range(2):
        mask = y == cls
        mean_signal = X[mask].mean(axis=0)  # [n_channel, n_length]
        for ch in range(mean_signal.shape[0]):
            plt.plot(mean_signal[ch], label=f"class {cls} ch {ch}", alpha=0.7)

    plt.legend()
    plt.title("Mean waveform per class")
    out_path = os.path.join(SAVE_DIR, "mean_waveforms.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()



if __name__ == '__main__':
    main()
