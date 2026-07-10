"""
dataset_STEAD.py  —  loads the STEAD dataset to numpy arrays.

    train_x, train_y
    val_x,   val_y
    test_x,  test_y

Shape convention:  x  →  [N number of sample, n_channel, n_length]
                   y  →  [N number of sample]   (integer class labels)
Labels :
    1 for earthquake
    0 for noise
"""

import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

TEST_DIR = BASE_DIR.parent /"data"/ "test"
X_test_path = TEST_DIR /"x.npy"
y_test_path = TEST_DIR /"y.npy"

TRAIN_DIR = BASE_DIR.parent /"data"/ "train"
X_train_path = TRAIN_DIR /"x.npy"
y_train_path = TRAIN_DIR /"y.npy"

VALIDATION_DIR = BASE_DIR.parent /"data"/ "validation"
X_val_path = VALIDATION_DIR /"x.npy"
y_val_path = VALIDATION_DIR /"y.npy"


def train_set():
    train_x = np.load(X_train_path, allow_pickle=True)
    train_y = np.load(y_train_path, allow_pickle=True)
    return train_x, train_y

def test_set():
    test_x = np.load(X_test_path, allow_pickle=True)
    test_y = np.load(y_test_path, allow_pickle=True)
    return test_x, test_y
def validation_set():
    val_x = np.load(X_val_path, allow_pickle=True)
    val_y = np.load(y_val_path, allow_pickle=True)
    return val_x, val_y

if __name__ == '__main__':
    train_x, train_y = train_set()
    print(train_x.shape, train_y.shape)
    test_x, test_y = test_set()
    print(test_x.shape, test_y.shape)
    val_x, val_y = validation_set()
    print(val_x.shape, val_y.shape)

