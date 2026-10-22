import os
import sys
import random
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import List

from config.columns import Columns
from src.features import create_target_deltas, derived_variable, add_features
from src.model import LGBMMultiOutputModel, XGBMultiOutputModel, CatBoostMultiOutputModel
from src.evaluation import prepare_true_values, performance_evaluate, save_performance_matrix
from src.visualization import plot_restored_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Seperator Tank Level Pipline")

    # 데이터 경로 지정
    parser.add_argument('--data_dir', type=str, default='data', help='데이터 디렉토리 경로')
    parser.add_argument('--model_dir', type=str, default='models', help='학습된 모델 저장 디렉토리 경로')

    # 파생변수 파라미터
    parser.add_argument('--lag_steps', type=int, default=180, help='lag 값')

    # LightGBM, XGBoost 하이퍼파라미터
    parser.add_argument('--n_estimators', type=int, default=2000, help='최대 트리 개수')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='학습률')
    parser.add_argument('--max_depth', type=int, default=8, help='트리 최대 깊이')
    parser.add_argument('--num_leaves', type=int, default=63, 
                        help='트리 하나당 최대 잎(Leaf) 노드 개수 (복잡도 제어 및 과적합 방지)')
    parser.add_argument('--subsample', type=float, default=0.8, 
                        help='개별 트리를 학습할 때 사용할 데이터 샘플링 비율 (0.0~1.0, 과적합 방지 및 학습 속도 향상)')
    parser.add_argument('--colsample_bytree', type=float, default=0.8, 
                        help='개별 트리를 학습할 때 사용할 피쳐 샘플링 비율')
    parser.add_argument('--random_state', type=int, default=42, help='seed')
    parser.add_argument('--early_stopping_rounds', type=int, default=50, help='과적합 방지를 위한 조기 종료')

    # 모델 정보
    parser.add_argument('--model_name', type=str, default='xgb', choices=['lgbm', 'xgb', 'catboost'], help='모델 이름')
    parser.add_argument('--output_dir', type=str, default='results', help='성능 평가 저장 경로')
    parser.add_argument('--figure_dir', type=str, default='fig', help='그래프 저장 경로')

    args, _ = parser.parse_known_args()

    return args

def prepare_features_and_targets(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
):
    X = df[feature_cols]
    y = df[target_cols]

    return X, y

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    args = parse_args()
    set_seed(args.random_state)

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
    train_df, orig_derived_feature_cols = derived_variable(train_df)
    train_df, feature_cols = add_features(train_df, lag_steps=args.lag_steps, feature_cols=orig_derived_feature_cols)
    
    val_df, orig_derived_feature_cols = derived_variable(val_df)
    val_df, feature_cols = add_features(val_df, lag_steps=args.lag_steps, feature_cols=orig_derived_feature_cols)

    # X, y 분리
    X_train, y_train = prepare_features_and_targets(train_df, feature_cols, target_cols_list)
    X_val, y_val = prepare_features_and_targets(val_df, feature_cols, target_cols_list)

    print(f"   - Number of Input Features: {X_train.shape[1]}")

    # 2. 모델 학습
    if args.model_name == 'lgbm':
        print("\n 2. Training LightGBM Model...")
        model = LGBMMultiOutputModel(
            args.n_estimators,
            args.learning_rate,
            args.max_depth,
            args.num_leaves,
            args.subsample,
            args.colsample_bytree,
            args.random_state,
        )

        model.fit(X_train, y_train)
    
    elif args.model_name == 'xgb':
        print("\n 2. Training XGBoost Model...")
        model = XGBMultiOutputModel(
            args.n_estimators,
            args.learning_rate,
            args.max_depth,
            args.subsample,
            args.colsample_bytree,
            args.early_stopping_rounds,
            args.random_state,
        )

        model.fit(X_train, y_train, X_val, y_val)
    elif args.model_name == 'catboost':
        print("\n 2. Training CatBoost Model...")
        model = CatBoostMultiOutputModel(
            args.n_estimators,
            args.learning_rate,
            args.max_depth,
            args.early_stopping_rounds,
            args.random_state,
        )

        model.fit(X_train, y_train, X_val, y_val)

    # 3. 모델 저장
    print("\n 3. Saving Trained Model")
    model.save(
        model_name=args.model_name,
        save_dir=args.model_dir,
    )

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
    test_df, orig_derived_feature_cols = derived_variable(test_df)
    test_df, feature_cols = add_features(test_df, lag_steps=args.lag_steps, feature_cols=orig_derived_feature_cols)

    # X, y 분리
    X_test, y_test = prepare_features_and_targets(test_df, feature_cols, target_cols_list)

    print(f"   - Number of Input Features: {X_test.shape[1]}")

    # 5. 추론 실행
    print("\n 5. Running Inference...")
    model_path = os.path.join(args.model_dir, f'{args.model_name}.pkl')
    if args.model_name == 'lgbm':
        loaded_model = LGBMMultiOutputModel.load(model_path)
    elif args.model_name == 'xgb':
        loaded_model = XGBMultiOutputModel.load(model_path)
    elif args.model_name == 'catboost':
        loaded_model = CatBoostMultiOutputModel.load(model_path)
    
    y_pred = loaded_model.predict(X_test, base_target_series=test_df[[target_col]].values)
    
    # 6. 성능 평가 및 결과 저장
    file_name = f'performance_matrix'
    print("\n 6. Evaluating Predictions...")
    y_true = prepare_true_values(test_df, target_col, target_cols_list)

    summary_df = performance_evaluate(test_df, y_true, y_pred, model_name=args.model_name, target_cols_list=target_cols_list)
    save_performance_matrix(summary_df, file_name, args.output_dir)

    # 7. 시각화 및 예측 결과 저장
    print("\n 7. Saving Results & Generating Plot...")
    plot_restored_predictions(
        test_df,
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