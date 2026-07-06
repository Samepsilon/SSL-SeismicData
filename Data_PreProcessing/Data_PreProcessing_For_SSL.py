from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pathlib

from STEAD_Data_Extraction import *

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR.parent /"data"/ "test"
X_test_path = TEST_DIR /"x.npy"
y_test_path = TEST_DIR /"y.npy"

TRAIN_DIR = BASE_DIR.parent /"data"/ "train"
X_train_path = TRAIN_DIR /"x.npy"
y_train_path = TRAIN_DIR /"y.npy"

VALIDATION_DIR = BASE_DIR.parent /"data"/ "validation"
X_val_path = VALIDATION_DIR /"x.npy"
y_val_path = VALIDATION_DIR /"y.npy"


#Default Value for input
seed_default = 42
frequency_default = 100
lowcut_default = 2
highcut_default = 49
train_test_split_default = 0.2
validation_test_split_default = 0.5
sample_length_default = frequency_default * 60



def BandPass_Filter_and_Normalization(X, fs=frequency_default, lowcut=lowcut_default, highcut=highcut_default):
    nyq  = 0.5 * fs
    b, a = butter(N=4, Wn=[lowcut / nyq, highcut / nyq], btype="band")

    X_filt = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            filtered = filtfilt(b, a, X[i, ch, :])
            max_abs  = np.max(np.abs(filtered))
            X_filt[i, ch, :] = filtered / max_abs if max_abs > 0 else filtered

    return X_filt

def PreProcessing(dataset = dataBuilder(), save_dataset = True):
    N = len(dataset)

    X = np.empty((N, 3, sample_length_default), dtype=np.float32)
    y_mark = np.array(dataset.metadata["trace_category"])
    y = np.empty((N,), dtype=np.int32)

    for i in range(N):
        if y_mark[i] == "earthquake_local":
            y[i] = 1
        else:
            y[i] = 0


    #random shuffle of the dataset
    Xshuffled, ylabel = shuffle(X, y, random_state=seed_default)

    Xfiltered = BandPass_Filter_and_Normalization(Xshuffled)

    X_train, X_temp, y_train, y_temp = train_test_split(
        Xfiltered, y, test_size=train_test_split_default, random_state=seed_default, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=validation_test_split_default, random_state=seed_default, stratify=y_temp
    )

    if save_dataset:
        np.save(X_test_path, X_test)
        np.save(y_test_path, y_test)

        np.save(X_train_path, X_train)
        np.save(y_train_path, y_train)

        np.save(X_val_path, X_val)
        np.save(y_val_path, y_val)


    return Xfiltered, ylabel




if __name__ == '__main__':
    X,y = PreProcessing()
