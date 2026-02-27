"""
Movie Rating Prediction — Preprocessing & Feature Engineering
=============================================================
Handles all data cleaning, encoding, and feature construction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Top-N cutoffs for frequency encoding
TOP_DIRECTORS = 50
TOP_ACTORS    = 50

FEATURES = [
    'Year_clean', 'Duration_min',
    'Votes_log', 'Genre_count',
    # Genre flags (top genres)
    'genre_Drama', 'genre_Action', 'genre_Romance', 'genre_Comedy',
    'genre_Thriller', 'genre_Crime', 'genre_Horror', 'genre_Family',
    'genre_Musical', 'genre_Adventure', 'genre_Mystery', 'genre_Biography',
    # Encoded
    'Director_enc', 'Actor1_enc', 'Actor2_enc', 'Actor3_enc',
    # Director / actor avg rating (target encoding)
    'Director_avg_rating', 'Actor1_avg_rating',
]

TARGET = 'Rating'


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw IMDb India dataframe."""
    df = df.copy()

    # Year: "(2019)" → 2019
    df['Year_clean'] = (
        df['Year'].astype(str)
        .str.extract(r'(\d{4})', expand=False)
        .astype(float)
    )

    # Duration: "109 min" → 109
    df['Duration_min'] = (
        df['Duration'].astype(str)
        .str.extract(r'(\d+)', expand=False)
        .astype(float)
    )

    # Votes: strip commas, convert
    df['Votes'] = (
        df['Votes'].astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    df['Votes_log'] = np.log1p(df['Votes'].fillna(0))

    # Genre count
    df['Genre'] = df['Genre'].fillna('Unknown')
    df['Genre_count'] = df['Genre'].str.split(',').apply(len)

    # Genre one-hot flags
    for g in ['Drama','Action','Romance','Comedy','Thriller','Crime',
              'Horror','Family','Musical','Adventure','Mystery','Biography']:
        df[f'genre_{g}'] = df['Genre'].str.contains(g, na=False).astype(int)

    # Fill string NaNs
    for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
        df[col] = df[col].fillna('Unknown').str.strip()

    return df


def fit_encoders(df_train: pd.DataFrame):
    """
    Fit label encoders + target-mean encoders on training data.
    Returns a dict of encoder artefacts.
    """
    artefacts = {}

    # Label encode director & actors (map rare → 'Other')
    for col, enc_key, top_n in [
        ('Director', 'dir_le', TOP_DIRECTORS),
        ('Actor 1',  'a1_le',  TOP_ACTORS),
        ('Actor 2',  'a2_le',  TOP_ACTORS),
        ('Actor 3',  'a3_le',  TOP_ACTORS),
    ]:
        top_vals = df_train[col].value_counts().nlargest(top_n).index.tolist()
        artefacts[f'{enc_key}_top'] = top_vals
        series = df_train[col].apply(lambda x: x if x in top_vals else 'Other')
        le = LabelEncoder()
        le.fit(series)
        artefacts[enc_key] = le

    # Target-mean encoding: Director & Actor 1 average rating
    for col, key in [('Director', 'dir_mean'), ('Actor 1', 'a1_mean')]:
        global_mean = df_train[TARGET].mean()
        mean_map = df_train.groupby(col)[TARGET].mean().to_dict()
        artefacts[key] = mean_map
        artefacts[f'{key}_global'] = global_mean

    return artefacts


def apply_encoders(df: pd.DataFrame, artefacts: dict) -> pd.DataFrame:
    """Apply fitted encoders to any split."""
    df = df.copy()

    for col, enc_key in [
        ('Director', 'dir_le'),
        ('Actor 1',  'a1_le'),
        ('Actor 2',  'a2_le'),
        ('Actor 3',  'a3_le'),
    ]:
        top_vals = artefacts[f'{enc_key}_top']
        series   = df[col].apply(lambda x: x if x in top_vals else 'Other')
        le       = artefacts[enc_key]
        # Handle unseen labels gracefully
        series = series.apply(lambda x: x if x in le.classes_ else 'Other')
        df[f'{col.replace(" ","")}_enc'] = le.transform(series)

    df.rename(columns={
        'Director_enc': 'Director_enc',
        'Actor1_enc':   'Actor1_enc',
        'Actor2_enc':   'Actor2_enc',
        'Actor3_enc':   'Actor3_enc',
    }, inplace=True)

    for col, key, out_col in [
        ('Director', 'dir_mean', 'Director_avg_rating'),
        ('Actor 1',  'a1_mean',  'Actor1_avg_rating'),
    ]:
        mean_map    = artefacts[key]
        global_mean = artefacts[f'{key}_global']
        df[out_col] = df[col].map(mean_map).fillna(global_mean)

    return df


def full_pipeline(df: pd.DataFrame, artefacts: dict = None, fit: bool = False):
    """
    Complete pipeline: clean → encode.
    If fit=True, fits encoders on df and returns (X, y, artefacts).
    If fit=False, applies existing artefacts and returns (X, y).
    """
    df = clean(df)
    df = df.dropna(subset=[TARGET])

    if fit:
        artefacts = fit_encoders(df)

    df = apply_encoders(df, artefacts)

    # Rename encoded actor columns to match FEATURES list
    rename = {
        'Actor 1_enc': 'Actor1_enc',
        'Actor 2_enc': 'Actor2_enc',
        'Actor 3_enc': 'Actor3_enc',
    }
    df.rename(columns=rename, inplace=True)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Fill any remaining NaN in features
    X = X.fillna(X.median(numeric_only=True))

    if fit:
        return X, y, artefacts
    return X, y
