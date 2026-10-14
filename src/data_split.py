import os
from pathlib import Path
import pandas as pd

import sys
sys.path.append(os.path.abspath(".."))

from config.columns import Columns

# =================================================================
# Wet 모드 추출
# =================================================================
def WET_MODE(data: pd.DataFrame, BLR_MASTER_cond: int):
    cond_blr = data[Columns.BLR_MASTER.value] >= BLR_MASTER_cond

    # 1분동안 조건 지속
    is_60m_continuous = (
        data.groupby('caseID')[Columns.BLR_MASTER.value]
        .transform(lambda x: (x >= BLR_MASTER_cond).rolling(60).sum() == 60)
    )

    blr_start_mask = is_60m_continuous.groupby(data['caseID']).cummax()
    df_blr_filterd = data[blr_start_mask]

    # 종료 시점: DRY 모드 전까지
    first_val = df_blr_filterd.groupby('caseID')[Columns.WET_DRY_MODE.value].transform('first')
    is_two = df_blr_filterd[Columns.WET_DRY_MODE.value] == 2
    two_appeared = is_two.groupby(df_blr_filterd['caseID']).cummax()

    cond_valid_1 = (first_val == 1) & (~two_appeared)
    end_mask = (first_val != 2) & (cond_valid_1)

    df_filtered = df_blr_filterd[end_mask].reset_index(drop=True)

    return df_filtered

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

# 1) WET MODE 데이터 분할
df = pd.read_csv(DATA_DIR / 'data.csv')
wet_df = WET_MODE(df, BLR_MASTER_cond=22)
wet_df = wet_df.dropna(axis=1)
wet_df.to_csv(DATA_DIR / 'wet_data.csv', index=False)

# 2) Train / Validation / Test 분할
wet_df = pd.read_csv(DATA_DIR / 'wet_data.csv')

train_df, val_test_df = split_time_series_data(
    wet_df, group_col='caseID', test_ratio=0.3
)
val_df, test_df = split_time_series_data(
    val_test_df, group_col='caseID', test_ratio=0.6
)

train_df.to_csv(DATA_DIR / 'train_df.csv', index=False)
val_df.to_csv(DATA_DIR / 'val_df.csv', index=False)
test_df.to_csv(DATA_DIR / 'test_df.csv', index=False)