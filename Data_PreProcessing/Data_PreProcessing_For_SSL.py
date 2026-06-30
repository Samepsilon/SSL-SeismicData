from scipy.signal import butter, filtfilt
from sklearn.utils import shuffle
import numpy as np
import pathlib

from STEAD_Data_Extraction import *

BASE_DIR = Path(__file__).resolve().parent
X_output_path = BASE_DIR.parent / "data"/"x.npy"
y_output_path = BASE_DIR.parent / "data"/"y.npy"



#Default Value for input
frequency_default = 100
lowcut_default = 2
highcut_default = 49



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

    X = np.empty((N, 3, 6000), dtype=np.float32)
    y = np.array(dataset.metadata["trace_category"])

    #random shuffle of the dataset
    Xshuffled, ylabel = shuffle(X, y, random_state=42)

    Xfiltered = BandPass_Filter_and_Normalization(Xshuffled)

    if save_dataset:
        np.save(X_output_path, Xfiltered)
        np.save(y_output_path, y)


    return Xfiltered, ylabel




if __name__ == '__main__':
    X,y = PreProcessing()
    print(X.shape, y.shape)
