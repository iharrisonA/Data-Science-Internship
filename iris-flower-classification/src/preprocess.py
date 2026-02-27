"""
Iris Flower Classification — Preprocessing
===========================================
Handles data loading, cleaning, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder

FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
TARGET   = 'species'

SPECIES_MAP = {
    'Iris-setosa':     'Setosa',
    'Iris-versicolor': 'Versicolor',
    'Iris-virginica':  'Virginica',
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].map(SPECIES_MAP).fillna(df[TARGET])
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Returns X_train, X_test, y_train, y_test (scaled), plus the fitted scaler & encoder.
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, le
