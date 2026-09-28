from pathlib import Path

import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""
Extract and Prepare the Data from the STEAD Dataset
The dataset should be in the .seisbench Folder, under the Seisbench format with both, hdf5 and csv file
The extracted data is save as 6 numpy file, between training, test and validation 
"""

# Configuration
seed_default = 42
sampling_rate_default = 100  #
component_order_default = "ZNE"  #
n_samples_default = 3000  # control the number of sample taken for each class, in this case noise and earthquake_local, so the total number of sample is 2*3000
n_windows_default = 2000  # control
# parameter for earthquake selection
magnitude_default = 2.5
distance_default = 21  # distance from recording station
# Train, test and validation split values
train_test_split_default = 0.2
validation_test_split_default = 0.5

# File Path
save_dataset = True  # save the numpy splitted dataset
BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR.parent / ".seisbench" / "STEAD"
output_path = BASE_DIR.parent / ".seisbench" / "extractedSTEAD"  # save directory for the new seisbench (not implemented for now)

TEST_DIR = BASE_DIR.parent / "data" / "test"
X_test_path = TEST_DIR / "x.npy"
y_test_path = TEST_DIR / "y.npy"

TRAIN_DIR = BASE_DIR.parent / "data" / "train"
X_train_path = TRAIN_DIR / "x.npy"
y_train_path = TRAIN_DIR / "y.npy"

VALIDATION_DIR = BASE_DIR.parent / "data" / "validation"
X_val_path = VALIDATION_DIR / "x.npy"
y_val_path = VALIDATION_DIR / "y.npy"

MAIN_DIR = BASE_DIR.parent / "data"
X_path = MAIN_DIR / "x.npy"
y_path = MAIN_DIR / "y.npy"


def dataBuilder():
    stead = sbd.WaveformDataset(input_path, sampling_rate=sampling_rate_default,
                                component_order=component_order_default)  # import the dataset

    mask_eq = (
            (stead.metadata["source_magnitude"] > magnitude_default) &
            (stead.metadata["source_distance_km"] <= distance_default) &
            (stead.metadata["trace_category"] == "earthquake_local")
    )  # mask for earthquake selection
    mask_no = (stead.metadata["trace_category"] == "noise")  # mask for noise selection

    stead_EQ = stead.copy()
    stead_EQ.filter(mask_eq)

    stead_NO = stead.copy()
    stead_NO.filter(mask_no)

    EarthquakeGenerator = sbg.GenericGenerator(stead_EQ)  # Creation of the seisbench earthquake generator
    EarthquakeGenerator.augmentation(
        sbg.WindowAroundSample(
            stead_EQ.metadata["trace_p_arrival_sample"],
            windowlen=n_windows_default,
            samples_before=0,
            selection="random",
            strategy="variable",
        )
    )

    NoiseGenerator = sbg.GenericGenerator(stead_NO)  # Creation of the seisbench noise generator
    NoiseGenerator.augmentation(
        sbg.RandomWindow(
            low=0,
            high=6000,
            windowlen=n_windows_default,
            strategy="pad",
        ),
    )

    N_total = n_samples_default * 2
    X = np.empty((N_total, 3, n_windows_default), dtype=np.float32)
    y = np.empty((N_total,), dtype=np.int32)

    Earthquake_samples = []
    for i in range(n_samples_default):  # generation of the final earthquake sample with windowing
        sample = EarthquakeGenerator[i]
        Earthquake_samples.append([sample['X'], 1])

    Noise_samples = []
    for i in range(n_samples_default):  # generation of the final noise sample with windowing
        sample = NoiseGenerator[i]
        Noise_samples.append([sample['X'], 0])

    Chosen_samples = Noise_samples + Earthquake_samples
    for i in range(N_total):
        X[i] = Chosen_samples[i][0]
        y[i] = Chosen_samples[i][1]

    X_Norm = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            # Z-score standardization per sequence, per channel
            mean_val = np.mean(X[i, ch, :])
            std_val = np.std(X[i, ch, :])

            # Add a small epsilon to prevent division by zero on dead channels
            X_Norm[i, ch, :] = (X[i, ch, :] - mean_val) / (std_val + 1e-6)

    X_shuffled, y_shuffled = shuffle(X_Norm, y, random_state=seed_default)  # shuffling of the final samples

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_shuffled, y_shuffled, test_size=train_test_split_default, random_state=seed_default, stratify=y
    )  # split between train and other, with stratification, to get an even number sample between all divided dataset
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=validation_test_split_default, random_state=seed_default, stratify=y_temp
    )  # split between validation and test, from the temp split, from the previous step

    print(X_shuffled.shape)
    print(y_shuffled.shape)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)

    if save_dataset:
        np.save(X_test_path, X_test)
        np.save(y_test_path, y_test)

        np.save(X_train_path, X_train)
        np.save(y_train_path, y_train)

        np.save(X_val_path, X_val)
        np.save(y_val_path, y_val)

        np.save(X_path, X_shuffled)
        np.save(y_path, y_shuffled)


if __name__ == '__main__':
    dataBuilder()
