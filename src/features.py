import pandas as pd

import os
import sys
sys.path.append(os.path.abspath(".."))

from config.columns import Columns

def derived_variable(data: pd.DataFrame):

    df = data.copy()

    df['SPIRAL_TUBE_TEMP_MAX'] = df[Columns.spiral_tube_temp_cols.value].max(axis=1)
    df['SPIRAL_TUBE_TEMP_MEAN'] = df[Columns.spiral_tube_temp_cols.value].mean(axis=1)
    df['LT_TEMP'] = df[[Columns.LT_TEMP1.value, Columns.LT_TEMP2.value]].mean(axis=1)

    origin_cols = []
    for c in Columns:
        val = c.value
        if isinstance(val, (list, tuple)):
            origin_cols.extend(val)
        else:
            origin_cols.append(val)

    origin_cols = (set(origin_cols)
                   - set(Columns.spiral_tube_temp_cols.value)
                   - set([Columns.LT_TEMP1.value, Columns.LT_TEMP2.value])
                   - set([Columns.WET_DRY_MODE.value]))

    derived_cols = ['SPIRAL_TUBE_TEMP_MAX', 'SPIRAL_TUBE_TEMP_MEAN', 'LT_TEMP']

    feature_cols = list(origin_cols) + derived_cols
    
    return df, feature_cols


# =================================================================
# 파생변수 생성
# =================================================================
def add_features(data: pd.DataFrame, lag_steps: int, feature_cols: list, groupby_col: str = 'caseID'):
    df = data.copy().reset_index(drop=True)
    new_cols = {}

    if groupby_col in df.columns:
        grouped = df.groupby(groupby_col)
    else:
        grouped = None

    for col in feature_cols:
        group_col = grouped[col] if grouped else df[col]

        for i in range(5, lag_steps + 1, 5):
            new_cols[f'{col}_lag_{i}'] = group_col.shift(i)
            new_cols[f'{col}_diff_{i}'] = group_col.diff(i).fillna(0.0)
            
            if i > 1:
                new_cols[f'{col}_rolling_mean_{i}'] = (
                    group_col.transform(lambda x: x.shift(1).rolling(window=i, min_periods=1).mean())
                    .fillna(0.0)
                )

                new_cols[f'{col}_rolling_std_{i}'] = (
                    group_col.transform(lambda x: x.shift(1).rolling(window=i, min_periods=1).std())
                    .fillna(0.0)
                )

    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    df = df.dropna().reset_index(drop=True)

    feature_cols += list(new_cols)

    return df, sorted(feature_cols)

# =================================================================
# 타겟변수 생성
# =================================================================

def create_target_deltas(data: pd.DataFrame, target_col: str, horizons: list, group_col: str = 'caseID'):
    df = data.copy()
    target_delta_cols = []

    for h in horizons:
        col_name = f"target_delta_{h}"
        future_val = df.groupby(group_col)[target_col].shift(-h)
        df[col_name] = future_val - df[target_col]
        target_delta_cols.append(col_name)

    df = df.dropna(subset=target_delta_cols).reset_index(drop=True)

    return df, target_delta_cols