"""
Titanic Survival Prediction — Preprocessing & Feature Engineering
=================================================================
All data cleaning and feature engineering logic in one place.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


FEATURES = [
    'Pclass', 'Sex_enc', 'AgeBand', 'FareBand',
    'Embarked_enc', 'FamilySize', 'IsAlone', 'HasCabin', 'Title_enc'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a raw Titanic dataframe.

    Steps:
    - Extract title from passenger name
    - Compute family size and alone flag
    - Impute missing Age using median per title group
    - Bin Age and Fare into ordinal bands
    - Flag whether cabin info is available
    - Fill missing Embarked with mode
    - Label-encode categorical columns

    Returns the enriched dataframe (copy).
    """
    df = df.copy()

    # ── Title ──────────────────────────────────────────────────────────────
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
    df['Title'] = df['Title'].map(title_map).fillna('Other')

    # ── Family Size ─────────────────────────────────────────────────────────
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # ── Age Imputation ──────────────────────────────────────────────────────
    age_medians = df.groupby('Title')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_medians).fillna(df['Age'].median())

    # ── Age Band ────────────────────────────────────────────────────────────
    df['AgeBand'] = pd.cut(
        df['Age'], bins=[0, 12, 18, 35, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    # ── Fare ────────────────────────────────────────────────────────────────
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3]).astype(float)

    # ── Cabin ────────────────────────────────────────────────────────────────
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # ── Embarked ─────────────────────────────────────────────────────────────
    df['Embarked'] = df['Embarked'].fillna('S')

    # ── Label Encoding ───────────────────────────────────────────────────────
    le = LabelEncoder()
    df['Sex_enc']      = le.fit_transform(df['Sex'])
    df['Embarked_enc'] = le.fit_transform(df['Embarked'])
    df['Title_enc']    = le.fit_transform(df['Title'])

    return df


def get_feature_display_names() -> dict:
    """Human-readable names for feature columns."""
    return {
        'Pclass':       'Passenger Class',
        'Sex_enc':      'Sex',
        'AgeBand':      'Age Band',
        'FareBand':     'Fare Band',
        'Embarked_enc': 'Embarkation Port',
        'FamilySize':   'Family Size',
        'IsAlone':      'Travelling Alone',
        'HasCabin':     'Has Cabin Info',
        'Title_enc':    'Title (Mr/Mrs/…)',
    }
