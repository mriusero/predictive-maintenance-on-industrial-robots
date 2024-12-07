from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sksurv.util import Surv


def prepare_data(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    columns_to_include: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepares the data for training and prediction.
    """
    df = df.sort_values(by=['item_id', 'time (months)']).reset_index(drop=True)
    if reference_df is not None:
        df = df.reindex(columns=reference_df.columns, fill_value=0)

    df['label'] = df['label'].astype(bool)
    df['time (months)'] = pd.to_numeric(df['time (months)'], errors='coerce')
    df.dropna(subset=['time (months)', 'label'], inplace=True)

    if columns_to_include:
        x_prepared = df[columns_to_include]
    else:
        x_prepared = df.copy()

    try:
        y = Surv.from_dataframe('label', 'time (months)', x_prepared)
    except ValueError as e:
        print(f"Error creating survival object: {e}")
        raise e

    return x_prepared, y