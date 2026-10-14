import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
import sys
sys.path.append(os.path.abspath(".."))

from config.columns import Columns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_restored_predictions(
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        case_id: str = None,
        HORIZONS: list = None,
        save_dir: str = None,
):
    test_df = test_df.reset_index(drop=True)

    if case_id is None or case_id == 'all':
        target_case_ids = test_df['caseID'].unique()
    else:
        target_case_ids = [case_id]

    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    for tc in target_case_ids:
        mask = (test_df['caseID'] == tc).values
        chunk_df = test_df[mask].copy().reset_index(drop=True)

        if len(chunk_df) == 0:
            continue

        num_id = str(chunk_df['NUM'].iloc[0])
        time_series = pd.to_datetime(chunk_df['time'])
        n_horizons = len(HORIZONS)

        chunk_y_true = y_test_arr[mask]
        chunk_y_pred = y_pred_arr[mask]

        target_col_name = Columns.SEP_TANK_LVL.value
        base_val = chunk_df[target_col_name].values.reshape(-1, 1)
        chunk_y_naive = np.tile(base_val, (1, n_horizons))

        fig, axes = plt.subplots(nrows=n_horizons, ncols=1, figsize=(15, 3.5 * n_horizons), sharex=True)

        for h_idx in range(n_horizons):
            ax = axes[h_idx]
            h_step = HORIZONS[h_idx]

            y_true_plot = chunk_y_true[:, h_idx]
            y_pred_plot = chunk_y_pred[:, h_idx]
            y_naive_plot = chunk_y_naive[:, h_idx]

            mae = np.mean(np.abs(y_true_plot - y_pred_plot))
            naive_mae = np.mean(np.abs(y_true_plot - y_naive_plot))

            ax.set_facecolor('white')

            ax.plot(time_series, y_true_plot, 'k-', linewidth=2, label=f'Actual (t+{h_step})')
            ax.plot(time_series, y_pred_plot, 'r--', linewidth=1.8, label=f'Pred (t+{h_step})')
            ax.plot(time_series, y_naive_plot, color='gray', linestyle='-.', linewidth=1.5, label=f'Naive (t+{h_step})')

            ax.set_title(f"Horizon +{h_step} Step Ahead Prediction | MAE: {mae:.4f} vs Naive MAE: {naive_mae:.4f}",
                         fontsize=12, fontweight='bold', loc='left')
            ax.set_ylabel(f'Separator Tank Level', fontsize=11)
            ax.legend(loc = 'upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=30)

        axes[-1].set_xlabel('Time', fontsize=12)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

        plt.suptitle(f"[{num_id}호기] CaseID: {tc} | Multi-Horizon (+1 ~ +{n_horizons}) Prediction",
                         fontsize=14, fontweight='bold', y=1.01)

        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f'case_{tc}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

    print(f"총 {len(target_case_ids)}개 CaseID에 대한 그래프 생성 및 저장 완료.")