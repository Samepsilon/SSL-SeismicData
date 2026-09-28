# Self-Supervised Learning (SSL) for Seismic Signal Classification & Representation Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![SeisBench](https://img.shields.io/badge/SeisBench-Seismic%20Data-orange.svg)](https://github.com/seisbench/seisbench)

A comprehensive deep learning framework for Self-Supervised Learning (SSL)  on a 3-channel continuous seismic waveforms. The experiment as for goal, whether pretraining on unlabelled seismic data improves earthquake and earthquake-like waveform classification performance, especially under different level of label scarcity.

---

## Table of Contents

- [Program Overview & Core Function](#program-overview--core-function)
- [System Architecture & Workflow](#system-architecture--workflow)
  - [1. Data Extraction & Preprocessing](#1-data-extraction--preprocessing)
  - [2. Self-Supervised Pretraining Paradigms](#2-self-supervised-pretraining-paradigms)
    - [A. Contrastive Learning (SimCLR)](#a-contrastive-learning-simclr)
    - [B. Masked Autoencoder (MAE)](#b-masked-autoencoder-mae)
  - [3. Downstream Tasks](#3-downstream-tasks)
    - [A. Supervised Fine-Tuning (Custom ANN)](#a-supervised-fine-tuning-custom-ann)
    - [B. Latent Space Evaluation & Clustering](#b-latent-space-evaluation--clustering)
  - [4. Experiment Tracking with MLflow](#4-experiment-tracking-with-mlflow)
- [Repository File Structure](#repository-file-structure)
- [Configuration Reference](#configuration-reference)
- [Getting Started & Usage](#getting-started--usage)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Execution Guide](#step-by-step-execution-guide)
- [Evaluation Metrics](#evaluation-metrics)

---

## Program Overview & Core Function


This program provides an end-to-end framework to:
1. **Pre-process and standardize** multi-channel seismic waveforms (Z, N, E components) from raw HDF5 using SeisBench.
2. **Pre-train deep neural feature extractors (Encoders)** on unlabelled seismic data :
   - **Contrastive Learning (SimCLR)**
   - **Generative Masked Modeling (Masked Autoencoder - MAE)**
3. **Fine-tune and benchmark** on a downstream  classification task across multiple low-data regimes (99% to 0% label scarcity).
4. **Evaluate representation quality** with K-Means clustering.
5. **Track all parameters, loss curves, classification metrics, and interactive HTML scatter plots** inside **MLflow** .

---



### 1. Data Extraction & Preprocessing

- **Source Code**: [`Data_PreProcessing/STEAD_Data_Extraction_V2.py`](Data_PreProcessing/STEAD_Data_Extraction_V2.py)
- **Input**: STEAD waveforms stored in `.seisbench/STEAD/`.
- **Filtering Criteria for binary classification**:
  - `earthquake_local`: Source magnitude $M > 2.5$, epicentral distance $\le 21\text{ km}$.
  - `noise`: Ambient background seismic noise traces.
- **Window Generation**:
  - Fixed length: $N = 2000$ samples (20 seconds at 100 Hz sampling rate).
  - Channels: 3 components ($Z$, $N$, $E$).
  - For earthquakes, `sbg.WindowAroundSample` windows around the `trace_p_arrival_sample`.
  - For noise, `sbg.RandomWindow` extracts random 2000-sample segments.
- **Normalization**:
  - Sequence-wise, channel-wise Z-score standardization:

- **Splits**: Stratified split into Train ($80\%$), Validation ($10\%$), and Test ($10\%$). Saved as `x.npy` and `y.npy` in `data/train/`, `data/validation/`, and `data/test/`.

---

### 2. Self-Supervised Pretraining Paradigms

#### A. Contrastive Learning (SimCLR)
- **Scripts**: [`Pretraining_Model/Contrastive/SimCLR/SimCLR_PreTrain.py`](Pretraining_Model/Contrastive/SimCLR/SimCLR_PreTrain.py)
- **Augmentation Pipeline** ([`TS_transformation.py`](Pretraining_Model/Contrastive/SimCLR/simCLR/module/transformation/TS_transformation.py)):
  - Utilizes `tsaug` library to produce two correlated views $(x_i, x_j)$ per sample:
    - **TimeWarp**: Non-linear stretching and compression of the time axis.
    - **Gaussian Jitter / Noise**: Adds scaled random noise .
    - **Dropout**: Randomly zeros out continuous temporal segments.
- **Model Architecture** ([`simCLR.py`](Pretraining_Model/Contrastive/SimCLR/simCLR/simCLR.py)):
  - **Encoder**: PyTorch `TransformerEncoder` with multi-head self-attention.
  - **Projector**: Non-linear MLP ($d_{in} \to d_{in}/2 \to d_{proj}$) mapping hidden states into latent space.
- **Objective Function**:
  - **NT-Xent** (Normalized Temperature-scaled Cross Entropy) loss ([`nt_xent.py`](Pretraining_Model/Contrastive/SimCLR/simCLR/module/nt_xent.py)) computed over positive view pairs against in-batch negative pairs.
- **Checkpointing**: Checkpoint saved when validation/epoch loss reaches a new minimum.

#### B. Masked Autoencoder (MAE)
- **Scripts**: [`Pretraining_Model/Generative_Encoder/MAE/MAE_PreTrain.py`](Pretraining_Model/Generative_Encoder/MAE/MAE_PreTrain.py)
- **Model Architecture** ([`MAE.py`](Pretraining_Model/Generative_Encoder/MAE/mae/MAE.py)):
  - **Patchify**: 1D convolution layer dividing 2000-sample traces into non-overlapping patches (e.g., `patch_size = 100`, yielding 20 patches per trace).
  - **Random Masking**: 75% of patches are masked; only 25% visible patches are fed to the encoder.
  - **Encoder**: ViT Transformer Blocks operating exclusively on unmasked tokens with 1D positional embeddings.
  - **Decoder**: Lightweight ViT Transformer Blocks receiving encoded visible tokens plus learnable `[MASK]` tokens restored to their original temporal sequence positions.
- **Objective Function**:
  - Mean Squared Error (MSE) computed strictly on the reconstructed masked patches:
    $$\mathcal{L}_{\text{MAE}} = \frac{1}{\text{mask\_ratio}} \cdot \frac{1}{N_{\text{masked}}} \sum (x_{\text{pred}} - x_{\text{orig}})^2 \odot \text{mask}$$

---

### 3. Downstream Tasks

#### A. Supervised Fine-Tuning with an custom ANN
- **Scripts**:
  - Contrastive: [`Downstream_Task/CustomANN/ANNclassifier_Contrastive.py`](Downstream_Task/CustomANN/ANNclassifier_Contrastive.py)
  - Generative: [`Downstream_Task/CustomANN/ANNclassifier_Generative.py`](Downstream_Task/CustomANN/ANNclassifier_Generative.py)
- **Classifier Architecture** ([`ANN_Model.py`](Downstream_Task/CustomANN/ANN_Model.py)):
  - 4-layer fully connected network: `Linear -> BatchNorm1d -> ReLU -> Dropout(0.2)` across hidden dimensions `[256, 128, 64, 32] -> Linear(32, 2)`.
- **Label Scarcity Simulation**:
  - Subsets training data to fractions such as `labelled_ratio = 0.01` (1%), `0.1` (10%), `0.3` (30%), `0.5` (50%), or `1.0` (100%).
- **Modes**:
  - **Fine-Tuning Pretrained**: Pretrained encoder weights loaded from checkpoint; end-to-end backpropagation updates both encoder and classifier.
  - **Baseline (From Scratch)**: Randomly initialized encoder trained without prior self-supervised weights.

#### B. Latent Space Evaluation & Clustering
- **Scripts**:
  - SimCLR evaluation: [`Downstream_Task/ClusteringModel_SimCLR.py`](Downstream_Task/ClusteringModel_SimCLR.py)
  - MAE evaluation: [`Downstream_Task/ClusteringModel_MAE.py`](Downstream_Task/ClusteringModel_MAE.py)
- **Methodology**:
  1. Extracts representation vectors $h$ on the test set using the frozen pretrained encoder.
  2. Applies $L_2$-normalization to embeddings: $h_{\text{norm}} = h / (\|h\|_2 + \epsilon)$.
  3. Clusters vectors using **K-Means** ($k = 2$).
  4. Resolves label assignment ambiguity via the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) against ground-truth labels.
  5. Projects latent representations to 2D using **UMAP** and **t-SNE**.
  6. Generates interactive Plotly scatter figures colored by true label and cluster label, logged directly as HTML artifacts into MLflow.

---

### 4. Experiment Tracking with MLflow

All pretraining and downstream runs are recorded in a local SQLite database (`SSL_PT_FT_MLflow.db`). Tracked entities include:
- **Parameters**: Random seed, learning rate, weight decay, epochs, batch sizes, patch sizes, mask ratios, etc.
- **Epoch-level Metrics**: Pretraining loss, learning rate schedule, finetune train loss, validation loss, validation F1-score.
- **Test Metrics**: Accuracy, Precision, Recall, Macro F1, ROC-AUC, PR-AUC, etc.
- **Artifacts**:  interactive UMAP/t-SNE HTML plots.

---

## Repository File Structure

```text
SSL&SeismicData/
│
├── Config/                                   # Central configuration modules
│   ├── dataset_config.py                     # Dataset specifications (channels, length, classes)
│   ├── Pretraining_Models_Config.py          # SimCLR & MAE pretraining hyperparameters
│   ├── Downstream_Task_Config.py             # ANN classifier layers, learning rate, label ratio
│   └── dataPrepration_config.py              # Data preparation configuration
│
├── Data_PreProcessing/                       # Raw data extraction & preprocessing
│   └── STEAD_Data_Extraction_V2.py           # SeisBench STEAD filtering, windowing, Z-score, split
│
├── data/                                     # Processed seismic arrays (.npy) [Git-ignored]
│   ├── train/                                # Training waveforms (x.npy) and labels (y.npy)
│   ├── validation/                           # Validation waveforms (x.npy) and labels (y.npy)
│   ├── test/                                 # Test waveforms (x.npy) and labels (y.npy)
│   ├── x.npy                                 # Full combined waveform array (6000, 3, 2000)
│   └── y.npy                                 # Full combined label array (6000,)
│
├── Pretraining_Model/                        # Self-supervised learning pipelines
│   ├── Contrastive/                          # Contrastive learning paradigm
│   │   └── SimCLR/
│   │       ├── SimCLR_PreTrain.py            # SimCLR pretraining script with MLflow tracking
│   │       ├── build_dataset.py              # PyTorch Dataset returning dual augmented views
│   │       ├── model.py                      # Optimizer & cosine warmup scheduler builder
│   │       ├── save_model_simCLR.py          # Serializes SimCLR checkpoint with architecture tags
│   │       └── simCLR/
│   │           ├── simCLR.py                 # SimCLR Transformer encoder & projection head
│   │           └── module/
│   │               ├── nt_xent.py            # Normalized Temperature-scaled Cross Entropy loss
│   │               ├── gather.py             # Distributed tensor gather helper
│   │               └── transformation/
│   │                   └── TS_transformation.py # Time series augmentations (Warp, Jitter, Dropout)
│   │
│   ├── Generative_Encoder/                   # Generative masked modeling paradigm
│   │   └── MAE/
│   │       ├── MAE_PreTrain.py               # MAE pretraining script with MLflow tracking
│   │       ├── build_dataset.py              # PyTorch Dataset loader for MAE
│   │       ├── model.py                      # Optimizer & learning rate schedule builder
│   │       ├── save_model_MAE.py             # Serializes MAE checkpoint with architecture tags
│   │       └── mae/
│   │           └── MAE.py                    # 1D PatchShuffle, MAE_Encoder, MAE_Decoder, ViT_Classifier
│   │
│   ├── Generative_Adversarial/               # GAN exploration directory
│   │   └── GAN/
│   │
│   └── utility/
│       └── Dataset/
│           └── Dataset_STEAD.py              # Helper loading train/val/test .npy arrays
│
├── Downstream_Task/                          # Transfer learning & evaluation
│   ├── CustomANN/                            # Supervised classification
│   │   ├── ANN_Model.py                      # 4-layer PyTorch MLP with BatchNorm & Dropout
│   │   ├── ANNclassifier_Contrastive.py      # Fine-tunes SimCLR encoder + ANN on constrained labels
│   │   └── ANNclassifier_Generative.py       # Fine-tunes MAE encoder + ANN on constrained labels
│   │
│   ├── ClusteringModel_SimCLR.py             # K-Means, Hungarian matching, UMAP/t-SNE for SimCLR
│   ├── ClusteringModel_MAE.py                # K-Means, Hungarian matching, UMAP/t-SNE for MAE
│   │
│   └── utility/                              # Downstream utilities
│       ├── encoder_loader.py                 # Interactive terminal selector for model checkpoints
│       ├── build_dataset.py                  # PyTorch TensorDataset builder
│       ├── model.py                          # Training & evaluation loops, multi-metric calculator
│       ├── model_MAE.py                      # Fine-tuning & evaluation routines for MAE
│       └── Dataset/
│           └── Dataset_STEAD.py              # Dataset loader for downstream tasks
│
├── Exploratory_Data_Analysis/                # Data exploration and visualization
│   ├── STEAD_Dataset_EDA.ipynb               # EDA on raw STEAD metadata and catalog
│   ├── Extracted_STEAD_Dataset_EDA.ipynb     # EDA on extracted and normalized waveforms
│   └── PLOTS_EDA/                            # Exported visualization figures
│
├── notebook/                                 # Interactive Jupyter Notebook pipeline
│   ├── 01_Data_Preparation_and_EDA.ipynb    # Data prep, 3-component waveforms & spectral EDA
│   ├── 02_Pretraining_MAE_and_Clustering.ipynb  # MAE pretraining, reconstruction & clustering
│   ├── 03_Pretraining_SimCLR_and_Clustering.ipynb # SimCLR augmentations, pretraining & clustering
│   └── 04_Downstream_Classification_ANN.ipynb # Downstream ANN (Pretrained vs Scratch benchmark)
│
├── Models/                                   # Saved weights and checkpoints [Git-ignored]
│   ├── simCLR/                               # Pretrained SimCLR checkpoints (.tar)
│   ├── MAE/                                  # Pretrained MAE checkpoints (.tar)
│   ├── ANN/                                  # Fine-tuned ANN classifiers and encoders (.pt)
│   └── Good Performing Model/                # Best-performing benchmark checkpoints
│
├── SSL_PT_FT_MLflow.db                       # SQLite backend database for MLflow experiments
├── README.md                                 # Project documentation
└── .gitignore                                # Git ignore rules for datasets, models, and virtualenvs
```

---

## Configuration Reference

Hyperparameters are decoupled into dedicated configuration files inside the [`Config/`](Config/) directory:

### 1. Dataset Configuration ([`Config/dataset_config.py`](Config/dataset_config.py))
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `name` | `"extractedSTEAD"` | Identifier used to name checkpoints and experiment runs |
| `n_class` | `2` | Number of target classes (`0`: Noise, `1`: Earthquake) |
| `n_channel` | `3` | Number of seismic channels ($Z, N, E$) |
| `n_length` | `2000` | Number of temporal samples per window (20 s @ 100 Hz) |

### 2. Pretraining Models Configuration ([`Config/Pretraining_Models_Config.py`](Config/Pretraining_Models_Config.py))

#### SimCLR Configuration:
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `n_layers` | `6` | Number of Transformer encoder layers |
| `n_hid` | `512` | Hidden dimension of Transformer layers |
| `n_head` | `16` | Number of attention heads (must divide input length evenly) |
| `batch_size` | `512` | Mini-batch size |
| `epochs` | `200` | Total pretraining epochs |
| `lr` | `3e-4` | Initial learning rate (AdamW optimizer) |
| `temperature` | `0.5` | Temperature parameter in NT-Xent loss |
| `projection_dim` | `512` | Output dimension of the projection head |
| `n_speed_change` | `4` | Number of speed warp points in time warping |
| `max_speed_ratio` | `2.0` | Maximum speed ratio for time warping |
| `noise_scale` | `0.01` | Standard deviation for Gaussian noise augmentation |
| `probability_for_dropout` | `0.1` | Probability of temporal dropout |

#### MAE Configuration:
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `emb_dim` | `512` | Latent embedding dimension |
| `encoder_layer` | `6` | Number of ViT encoder Transformer blocks |
| `decoder_layer` | `2` | Number of lightweight ViT decoder blocks |
| `attention_head` | `16` | Number of multi-head self-attention heads |
| `patch_size` | `100` | Number of timesteps per 1D patch (20 patches total) |
| `mask_ratio` | `0.75` | Fraction of patches masked during pretraining ($75\%$) |
| `base_learning_rate`| `2e-4` | Base learning rate scaled by batch size |
| `total_epoch` | `200` | Total pretraining epochs |
| `warmup_epoch` | `25` | Cosine decay warmup epochs |

### 3. Downstream Classifier Configuration ([`Config/Downstream_Task_Config.py`](Config/Downstream_Task_Config.py))
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `hidden_layer_layout`| `[256, 128, 64, 32]` | Hidden layer dimensions for `CustomANN` |
| `logistic_epochs` | `100` | Number of fine-tuning epochs |
| `logistic_batch_size`| `256` | Batch size for downstream classification |
| `learning_rate` | `3e-4` | Fine-tuning learning rate (AdamW) |
| `labelled_ratio` | `0.3` | Proportion of training labels to use ($0.01, 0.1, 0.3, 0.5, 1.0$) |
| `pretrained` | `False` | `True` to load pretrained weights via loader; `False` for scratch |

---

## Getting Started & Usage

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for Transformer pretraining)
- Key Python dependencies:
  ```bash
  pip install torch torchvision torchaudio seisbench scikit-learn scipy numpy pandas plotly umap-learn tsaug timm einops tqdm mlflow
  ```

---

### Step-by-Step Execution Guide

#### Step 1: Pre-process the STEAD Dataset
Ensure the raw STEAD dataset files (HDF5 and CSV metadata) are placed under `.seisbench/STEAD/`. Run the extraction script:
```bash
python Data_PreProcessing/STEAD_Data_Extraction_V2.py
```
*Output*: Generates stratified `x.npy` and `y.npy` files inside `data/train/`, `data/validation/`, and `data/test/`.

#### Step 2: Self-Supervised Pretraining

- **To pretrain SimCLR (Contrastive Learning)**:
  ```bash
  python Pretraining_Model/Contrastive/SimCLR/SimCLR_PreTrain.py
  ```
- **To pretrain MAE (Generative Masked Autoencoding)**:
  ```bash
  python Pretraining_Model/Generative_Encoder/MAE/MAE_PreTrain.py
  ```
*Output*: Model checkpoints are saved under `Models/simCLR/` or `Models/MAE/`. Training progress is logged to MLflow under the corresponding experiment.

#### Step 3: Downstream Classification (Fine-Tuning)
Configure `labelled_ratio` and `pretrained` in [`Config/Downstream_Task_Config.py`](Config/Downstream_Task_Config.py).

- **For Contrastive (SimCLR) Pretrained Classifier**:
  ```bash
  python Downstream_Task/CustomANN/ANNclassifier_Contrastive.py
  ```
- **For Generative (MAE) Pretrained Classifier**:
  ```bash
  python Downstream_Task/CustomANN/ANNclassifier_Generative.py
  ```
*Interactive Console Prompt*: If `pretrained = True`, the script prompts you to select the checkpoint folder and file:

#### Step 4: Representation Clustering & Latent Projections
To evaluate how well the pretrained encoder clusters earthquake vs. noise waveforms without any fine-tuning:

- **For SimCLR**:
  ```bash
  python Downstream_Task/ClusteringModel_SimCLR.py
  ```
- **For MAE**:
  ```bash
  python Downstream_Task/ClusteringModel_MAE.py
  ```
*Output*: Computes unsupervised metrics (Silhouette, ARI, NMI, Hungarian Accuracy) and logs interactive Plotly HTML UMAP and t-SNE projections to MLflow.

#### Step 5: Launch the MLflow Dashboard
View all runs, metrics, hyperparameters, and interactive plots in your browser:
```bash
mlflow server --backend-store-uri "sqlite:///D:\Desktop\Intership IT\SSL&SeismicData\SSL_PT_FT_MLflow.db" --port 5000
```
Then navigate to: `http://localhost:5000`

---

## Evaluation Metrics

| Category | Metric | Purpose |
| :--- | :--- | :--- |
| **Supervised Classification** | **Accuracy (Acc)** | Proportion of correct predictions across test samples |
| | **Macro F1-Score** | Harmonic mean of precision and recall (handles class balance) |
| | **ROC-AUC** | Area under the Receiver Operating Characteristic curve |
| | **PR-AUC** | Area under the Precision-Recall curve |
| **Unsupervised Clustering** | **Hungarian Accuracy** | Optimal bijective mapping between cluster assignments and true labels |
| | **Adjusted Rand Index (ARI)** | Similarity between predicted clusters and ground-truth classes |
| | **Normalized Mutual Info (NMI)** | Information shared between clusters and true labels ($[0, 1]$) |
| | **Silhouette Score** | Measure of cluster compactness and separation ($[-1, 1]$) |
| **Manifold Visualizations** | **UMAP & t-SNE (2D)** | Dimensionality reduction plots colored by true label and cluster |