import os
from pathlib import Path
import pandas as pd

import sys
sys.path.append(os.path.abspath(".."))

from config.columns import Columns

# =================================================================
# Train, Test 데이터 분할
# =================================================================

def split_time_series_data(
        data: pd.DataFrame,
        output_dir: str = '../data',
        group_col: str = 'caseID',
        test_ratio: float = 0.3,
):
    os.makedirs(output_dir, exist_ok=True)
    df = data.copy()

    case_summary = df.groupby(group_col)['NUM'].first().reset_index()

    train_cases = []
    test_cases = []

    for _, group in case_summary.groupby('NUM'):
        unique_cases = group[group_col].values
        
        split_idx = int(len(unique_cases) * (1 - test_ratio))

        train_cases.extend(unique_cases[:split_idx])
        test_cases.extend(unique_cases[split_idx:])

    train_df = df[df[group_col].isin(train_cases)]
    test_df = df[df[group_col].isin(test_cases)]

    return train_df, test_df


# =================================================================
# 데이터 전처리 실행
# =================================================================
DATA_DIR = Path(r'..\data')

df = pd.read_csv(DATA_DIR / 'data.csv')

train_df, val_test_df = split_time_series_data(
    df, group_col='caseID', test_ratio=0.3
)
val_df, test_df = split_time_series_data(
    val_test_df, group_col='caseID', test_ratio=0.6
)

train_df.to_csv(DATA_DIR / 'train_data.csv', index=False)
val_df.to_csv(DATA_DIR / 'val_data.csv', index=False)
test_df.to_csv(DATA_DIR / 'test_data.csv', index=False)