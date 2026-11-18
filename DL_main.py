import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from config.columns import Columns
from src.features import create_target_deltas, derived_variable
from src.DL_model import TimeSeriesDataset, LSTMModel, GRUModel, Conv1DModel
from src.evaluation import performance_evaluate, save_performance_matrix
from src.visualization import plot_restored_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Seperator Tank Level Pipline")

    # 데이터 경로 지정
    parser.add_argument('--data_dir', type=str, default='data', help='데이터 디렉토리 경로')
    parser.add_argument('--model_dir', type=str, default='models', help='학습된 모델 저장 디렉토리 경로')

    # 파생변수 파라미터
    parser.add_argument('--lag_steps', type=int, default=180, help='lag 값')

    # DL 하이퍼파라미터
    parser.add_argument('--lookback', type=int, default=180, help='과거 입력 시퀀스 길이')
    parser.add_argument('--step_size', type=int, default=10, help='슬라이딩 윈도우 이동 간격 (데이터 샘플링 스텝)')
    parser.add_argument('--batch_size', type=int, default=64, choices=[64, 128, 256], help='미니 배치 크기 (한번에 학습할 수 있는 샘플 수)')
    parser.add_argument('--hidden_dim', type=int, default=512, help='은닉 상태의 차원 크기')
    parser.add_argument('--num_layers', type=int, default=2, help='레이어의 개수')
    parser.add_argument('--dropout', type=float, default=0.2, help='과적합 방지를 위해 무작위로 뉴런을 끄는 Dropout 비율 (0.0 ~ 1.0)')
    
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='과적합 방지를 위한 가중치 감쇠')
    parser.add_argument('--epochs', type=int, default=100, help='최대 학습 Epoch 횟수')
    parser.add_argument('--patience', type=int, default=10, help='조기 종료를 위한 인내 Epoch 수')
    parser.add_argument('--random_state', type=int, default=42, help='seed')

    # 모델 정보
    parser.add_argument('--model_name', type=str, default='lstm', choices=['lstm', 'gru', '1d_cnn'], help='모델 이름')
    parser.add_argument('--output_dir', type=str, default='results', help='성능 평가 저장 경로')
    parser.add_argument('--figure_dir', type=str, default='fig', help='그래프 저장 경로')

    args, _ = parser.parse_known_args()

    return args

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    set_seed(args.random_state)
    args.model_name = '1d_cnn'

    print("=" * 65)
    print(f"[Train Pipeline Start] Model: {args.model_name}")
    print("=" * 65)

    # 1. 데이터 로드
    print("\n 1. Loading Pre-split CSV Datasets...")
    train_path = os.path.join(args.data_dir, f'train_data.csv')
    val_path = os.path.join(args.data_dir, f'val_data.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found at: {train_path}")

    if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation dataset not found at: {val_path}")

    train_df = pd.read_csv(train_path, index_col=False)
    val_df = pd.read_csv(val_path, index_col=False)

    target_col = Columns.SEP_TANK_LVL.value
    horizons = [10, 20, 30, 40, 50, 60]
    
    train_df, target_cols_list = create_target_deltas(train_df, target_col, horizons=horizons)
    val_df, _ = create_target_deltas(val_df, target_col, horizons=horizons)

    # Feature Engineering
    train_df, feature_cols = derived_variable(train_df)
    val_df, feature_cols = derived_variable(val_df)

    # Scaling
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()

    x_scaler = StandardScaler()
    train_df_scaled[feature_cols] = x_scaler.fit_transform(train_df_scaled[feature_cols])
    val_df_scaled[feature_cols] = x_scaler.transform(val_df_scaled[feature_cols])

    y_scaler = StandardScaler()
    train_df_scaled[target_cols_list] = y_scaler.fit_transform(train_df_scaled[target_cols_list])
    val_df_scaled[target_cols_list] = y_scaler.transform(val_df_scaled[target_cols_list])

    # 1. x_scaler 속성 추출 후 저장
    x_params = {
        'mean': x_scaler.mean_.tolist(),
        'scale': x_scaler.scale_.tolist(),
    }
    with open(os.path.join(args.model_dir, 'x_scaler_params.json'), 'w') as f:
        json.dump(x_params, f)

    # 2. y_scaler 속성 추출 후 저장 (타겟이 단일 값이거나 다중 값일 때 모두 대응 가능)
    y_params = {
        'mean': (
            y_scaler.mean_.tolist()
            if hasattr(y_scaler.mean_, 'tolist')
            else [y_scaler.mean_]
        ),
        'scale': (
            y_scaler.scale_.tolist()
            if hasattr(y_scaler.scale_, 'tolist')
            else [y_scaler.scale_]
        ),
    }
    with open(os.path.join(args.model_dir, 'y_scaler_params.json'), 'w') as f:
        json.dump(y_params, f)

    print("모델과 스케일러가 각각 안전하게 저장되었습니다.")

    # Dataset
    train_dataset = TimeSeriesDataset(
        raw_data=train_df,
        data=train_df_scaled,
        feature_cols=feature_cols,
        target_cols=target_cols_list,
        group_col='caseID',
        lookback=args.lookback,
        step_size=args.step_size,
    )

    val_dataset = TimeSeriesDataset(
        raw_data=val_df,
        data=val_df_scaled,
        feature_cols=feature_cols,
        target_cols=target_cols_list,
        group_col='caseID',
        lookback=args.lookback,
        step_size=args.step_size,
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. 모델 학습
    os.makedirs(args.model_dir, exist_ok=True)

    if args.model_name == 'lstm':
        print("\n 2. Training LSTM Model...")
        model = LSTMModel(in_channels=len(feature_cols), 
                          num_horizons=len(horizons), 
                          hidden_dim=args.hidden_dim, 
                          num_layers=args.num_layers, 
                          dropout=args.dropout,
                          ).to(device)
    elif args.model_name == 'gru':
        print("\n 2. Training GRU Model...")
        model = GRUModel(in_channels=len(feature_cols),
                         num_horizons=len(horizons),
                         hidden_dim=args.hidden_dim,
                         num_layers=args.num_layers,
                         dropout=args.dropout,
                         ).to(device)
    elif args.model_name == '1d_cnn':
        print("\n 2. Training 1D CNN Model...")
        model = Conv1DModel(in_channels=len(feature_cols),
                            num_horizons=len(horizons),
                            ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item()
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.model_dir, f"best_{args.model_name}.pt")
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("=" * 65)
    print(f"[Inference Pipeline Start] Model: {args.model_name}")
    print("=" * 65)

    # 4. Test 데이터 전처리 및 파이프라인 적용
    print("\n 4. Loading & Preprocessing Test Dataset...")
    test_path = os.path.join(args.data_dir, f'test_data.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")

    test_df = pd.read_csv(test_path, index_col=False)

    target_col = Columns.SEP_TANK_LVL.value
    horizons = [10, 20, 30, 40, 50, 60]
    test_df, target_cols_list = create_target_deltas(test_df, target_col, horizons=horizons)

    # Feature Engineering
    test_df, feature_cols = derived_variable(test_df)

    # Scaler
    # x_scaler 복원
    with open(os.path.join(args.model_dir, 'x_scaler_params.json'), 'r') as f:
        x_params = json.load(f)

    x_scaler = StandardScaler()
    x_scaler.mean_ = np.array(x_params['mean'])
    x_scaler.scale_ = np.array(x_params['scale'])

    # y_scaler 복원
    with open(os.path.join(args.model_dir, 'y_scaler_params.json'), 'r') as f:
        y_params = json.load(f)

    y_scaler = StandardScaler()
    y_scaler.mean_ = np.array(y_params['mean'])
    y_scaler.scale_ = np.array(y_params['scale'])

    print('스케일러가 JSON 파라미터로부터 성공적으로 복원되었습니다.')

    test_df_scaled = test_df.copy()
    test_df_scaled[feature_cols] = x_scaler.transform(test_df_scaled[feature_cols])
    test_df_scaled[target_cols_list] = y_scaler.transform(test_df_scaled[target_cols_list])

    # Dataset
    test_dataset = TimeSeriesDataset(
        raw_data=test_df,
        data=test_df_scaled,
        feature_cols=feature_cols,
        target_cols=target_cols_list,
        group_col='caseID',
        lookback=args.lookback,
        step_size=args.step_size,
    )

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. 추론 진행
    print("\n 5. Running Inference...")
    model_path = os.path.join(args.model_dir, f'best_{args.model_name}.pt')

    if args.model_name == 'lstm':
        model = LSTMModel(in_channels=len(feature_cols), 
                        num_horizons=len(horizons), 
                        hidden_dim=args.hidden_dim, 
                        num_layers=args.num_layers, 
                        dropout=args.dropout
                        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    elif args.model_name == 'gru':
        model = GRUModel(in_channels=len(feature_cols),
                         num_horizons=len(horizons),
                         hidden_dim=args.hidden_dim,
                         num_layers=args.num_layers,
                         dropout=args.dropout,
                         ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    elif args.model_name == '1d_cnn':
        model = Conv1DModel(in_channels=len(feature_cols),
                            num_horizons=len(horizons),
                            ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    preds_scaled = []
    targets_scaled = []
    target_y = []
    with torch.no_grad():
        for bx, by, by_true in test_loader:
            bx = bx.to(device)
            pred = model(bx)

            preds_scaled.append(pred.cpu().numpy())
            targets_scaled.append(by.numpy())
            target_y.append(by_true.numpy())

    preds_scaled = np.vstack(preds_scaled)
    targets_scaled = np.vstack(targets_scaled)
    target_y = np.vstack(target_y)

    real_preds = y_scaler.inverse_transform(preds_scaled)
    real_targets = y_scaler.inverse_transform(targets_scaled)

    y_pred = target_y + real_preds
    y_true = target_y + real_targets

    eval_test_df = test_df.iloc[test_dataset.target_indices].copy()

    # 6. 성능 평가 및 결과 저장
    file_name = f'performance_matrix'
    print("\n 6. Evaluating Predictions...")

    summary_df = performance_evaluate(eval_test_df, y_true, y_pred, model_name=args.model_name, target_cols_list=target_cols_list)
    save_performance_matrix(summary_df, file_name, args.output_dir)

    # 7. 시각화 및 예측 결과 저장
    print("\n 7. Saving Results & Generating Plot...")
    plot_restored_predictions(
        eval_test_df,
        y_true,
        y_pred,
        case_id=None,
        HORIZONS=horizons,
        save_dir=os.path.join(args.figure_dir, args.model_name)
    )

    print("\n" + "=" * 65)
    print("[Inference Pipeline Finished Successfully]")
    print("=" * 65)

if __name__ == '__main__':
    main()