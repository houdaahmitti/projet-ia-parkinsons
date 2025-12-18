import numpy as np
import pandas as pd

def load_parkinsons_data(path='./data/telemonitoring/parkinsons_updrs.data'):

    dataset = pd.read_csv(path)
    X = dataset.drop(columns=['total_UPDRS']).values
    y = dataset['total_UPDRS'].values
    return X, y