import numpy as np
import pandas as pd

def save_polygon(poly: np.ndarray, path: str):
    df = pd.DataFrame(poly)
    df.to_csv(path, index=False, header=False)

def load_polygon(path):
    df = pd.read_csv(path, header=None, index_col=False)
    return df.to_numpy()