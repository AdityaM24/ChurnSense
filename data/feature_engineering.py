import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr']
    )

    df['charge_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)

    df['streaming_both'] = (
        (df['StreamingTV'] == 'Yes') & (df['StreamingMovies'] == 'Yes')
    ).astype(int)

    return df