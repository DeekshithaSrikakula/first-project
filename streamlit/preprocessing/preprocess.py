import pandas as pd
import numpy as np

def load_and_clean_data(path):
    df = pd.read_csv(path)

    target_col = "Nutrient_Quality_Score"

    df = df.dropna(subset=[target_col])

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    df = pd.get_dummies(df, columns=["Soil_Texture"], drop_first=True)

    return df
