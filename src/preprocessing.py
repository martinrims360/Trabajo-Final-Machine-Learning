import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalización
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Añadir canal (para CNN)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, X_test, y_train, y_test


def flatten_data(X_train, X_test):
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    return X_train_flat, X_test_flat