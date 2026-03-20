import numpy as np

def dummy_scaler(X : np.array):
    # Dummy Scaler
    return X / 255.


def apply_scaler(X: dict) -> dict:
    result = {}
    for tag, arr in X.items():
        if isinstance(arr, np.ndarray):
            result[tag] = dummy_scaler(X=arr)
        else:
            result[tag] = arr
    return result