from pathlib import Path
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(".."))

from config.columns import Columns


DATA_DIR = Path(r'../data')
IMPUTATION_FILE_NAME = 'data_imputation.csv'

def WET_MODE(BLR_MASTER_cond: int):
    
    df = pd.read_csv(DATA_DIR / IMPUTATION_FILE_NAME)
    df['time'] = pd.to_datetime(df['time'])

    # 1분동안 조건 지속
    is_60m_continuous = (
        df.groupby('caseID')[Columns.BLR_MASTER.value]
        .transform(lambda x: (x >= BLR_MASTER_cond).rolling(60).sum() == 60)
    )

    blr_start_mask = is_60m_continuous.groupby(df['caseID']).cummax()
    df_blr_filterd = df[blr_start_mask]

    # 종료 시점: DRY 모드 전까지
    first_val = df_blr_filterd.groupby('caseID')[Columns.WET_DRY_MODE.value].transform('first')
    is_two = df_blr_filterd[Columns.WET_DRY_MODE.value] == 2
    two_appeared = is_two.groupby(df_blr_filterd['caseID']).cummax()

    cond_valid_1 = (first_val == 1) & (~two_appeared)
    end_mask = (first_val != 2) & (cond_valid_1)

    df_filtered = df_blr_filterd[end_mask].reset_index(drop=True)
    df_filtered.to_csv('../data/data.csv', index=False)

    return None


if __name__ == '__main__':
    WET_MODE(BLR_MASTER_cond=22) # WET 구간 데이터 전처리