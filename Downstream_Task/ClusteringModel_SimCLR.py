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

import mlflow
import pandas as pd
import plotly.express as px

# CONFIG — edit these to point at your data and model checkpoint


BATCH_SIZE  = 128
N_CLASS     = extractedSTEAD["n_class"]     # 2 for your seismic dataset



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
    # Start MLflow Run
    mlflow.set_tracking_uri(r"sqlite:///D:\Desktop\Intership IT\SSL&SeismicData\SSL_PT_FT_MLflow.db")
    mlflow.set_experiment("SSL_Pretraining_SimCLR_6*512_Clustering")

    with mlflow.start_run(run_name="Baseline"):
        # Load data
        X, y = test_set()

        # Load pretrained encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        CKPT_PATH = encoder_loader_path()
        splited_info = Path(CKPT_PATH).stem.split('_')
        projection_dim = int(splited_info[-1])
        n_layers = int(splited_info[-4])
        n_hid = int(splited_info[-3])
        n_head = int(splited_info[-2])

        # Log Hyperparameters to MLflow
        mlflow.log_params({
            "projection_dim": projection_dim,
            "n_layers": n_layers,
            "n_hid": n_hid,
            "n_head": n_head,
            "checkpoint_path": CKPT_PATH
        })

        encoder = SimCLR_Transformer(
            projection_dim=projection_dim,
            n_channel=extractedSTEAD["n_channel"],
            n_length=extractedSTEAD["n_length"],
            n_layers=n_layers,
            n_hid=n_hid,
            n_head=n_head,
        )

        encoder.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        encoder = encoder.to(device)
        print(f"Loaded checkpoint: {CKPT_PATH}")

        # Extract representations
        h = extract_embeddings(encoder, X, device, BATCH_SIZE)

        # L2-normalise before clustering
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)

        # K-Means clustering
        kmeans = KMeans(n_clusters=N_CLASS, n_init=20, random_state=42)
        cluster_labels = kmeans.fit_predict(h_norm)

        # Metrics
        acc, cluster_labels_remapped = hungarian_accuracy(y, cluster_labels, N_CLASS)
        ari = adjusted_rand_score(y, cluster_labels)
        nmi = normalized_mutual_info_score(y, cluster_labels)
        sil = silhouette_score(h_norm, cluster_labels, sample_size=min(5000, len(y)))

        # Log Metrics to MLflow
        mlflow.log_metrics({
            "accuracy": acc,
            "ARI": ari,
            "NMI": nmi,
            "silhouette": sil
        })

        # 2-D projections (UMAP + t-SNE)
        coords = {}

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        coords["UMAP"] = reducer.fit_transform(h_norm)

        tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
        coords["t-SNE"] = tsne.fit_transform(h_norm)

        # ─── Log Interactive Projections to MLflow ───
        for method_name, xy in coords.items():
            df_plot = pd.DataFrame({
                "x": xy[:, 0],
                "y": xy[:, 1],
                "true_label": y.astype(str),
                "cluster": cluster_labels.astype(str)
            })

            # Interactive plot colored by True Label
            fig_true = px.scatter(
                df_plot, x="x", y="y", color="true_label",
                title=f"{method_name} (True Labels)",
                hover_data=["cluster"]
            )
            mlflow.log_figure(fig_true, f"interactive_plots/{method_name.lower()}_true_labels.html")

            # Interactive plot colored by Cluster Assignment
            fig_cluster = px.scatter(
                df_plot, x="x", y="y", color="cluster",
                title=f"{method_name} (K-Means Clusters)",
                hover_data=["true_label"]
            )
            mlflow.log_figure(fig_cluster, f"interactive_plots/{method_name.lower()}_clusters.html")



if __name__ == '__main__':
    main()
