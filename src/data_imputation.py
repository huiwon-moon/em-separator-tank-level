import pandas as pd

import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from config.columns import Columns

ffill_cols = [
                Columns.WET_DRY_MODE.value, Columns.BLR_MASTER.value, Columns.FW_MSTR.value, 
                Columns.FUEL_MSTR.value,Columns.WR_RATIO.value, Columns.ZR_RATIO.value, 
                Columns.XLIM_FW_DMD.value, Columns.LL_PCT.value, Columns.LL.value, 
                Columns.BRP_DISCHARGE_VLV_PCT.value, Columns.BRP_DISCHARGE_VLV.value, 
                Columns.SEP_STORAGE_TNK_LVL_CTRL_VLV.value,
            ]

linear_cols = [
                Columns.LT_TEMP1.value, Columns.LT_TEMP2.value, Columns.SEP_TANK_LVL.value, 
                Columns.ECO_IN_FW_FLOW.value, Columns.FW_FLOW.value, Columns.FW_PRESSURE.value, 
                Columns.STEAM_FLOW.value, Columns.TP.value, Columns.MST.value, Columns.RH_STEAM_TEMP.value, 
                Columns.SUPER_HEAT.value, Columns.LT_RETURN_LEG_LEVEL.value, Columns.LT_RETURN_LEG_TEMP.value, 
                Columns.ECON_IN_FW_TEMP.value, Columns.BRP_DIFF_PRESSURE.value, Columns.FUEL_FW_RATIO.value, 
                Columns.BRP_MTR_AMPS.value, Columns.SEP_WET_LEG_LVL.value, Columns.SEP_WET_LEG_LVL_PCT.value,
            ] + Columns.spiral_tube_temp_cols.value



def imputation():
    df = pd.read_csv(r'data\origin_data\origin_data.csv')
    df.columns = df.columns.str.replace(':', '_')
    
    grouped = df.groupby('caseID')
    for col in ffill_cols:
        df[col] = grouped[col].ffill()

    for col in linear_cols:
        df[col] = grouped[col].transform(
            lambda group: group.interpolate(method='linear')
        )

    nan_case_ids = df[df[ffill_cols + linear_cols].isna().any(axis=1)]['caseID'].unique()
    cleaned_df = df[~df['caseID'].isin(nan_case_ids)]
    cleaned_df = cleaned_df.dropna(axis=1, how='any')

    cleaned_df.to_csv('data/data_imputation.csv', index=False)

if __name__ == "__main__":
  imputation()