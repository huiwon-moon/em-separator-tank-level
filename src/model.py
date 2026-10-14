import os
import joblib
import numpy as np
import pandas as pd
from typing import List
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import lightgbm as lgbm
import xgboost as xgb
from catboost import CatBoostRegressor

class LGBMMultiOutputModel:
    def __init__(
            self,
            n_estimators: int,
            learning_rate: float,
            max_depth: int,
            num_leaves: int,
            subsample: float,
            colsample_bytree: float,
            random_state: int,
            ):
        
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'eval_metric': 'mae',
            'n_jobs': -1,
            'verbose': -1,
        }

        self.base_model = lgbm.LGBMRegressor(**self.params)
        self.model = MultiOutputRegressor(self.base_model)
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """모델 학습"""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame, base_target_series: pd.Series = None):
        """예측 수행"""
        if not self.is_fitted:
            raise RuntimeError("모델이 아직 학습되지 않았습니다. fit()을 먼저 실행하세요.")

        delta_pred = self.model.predict(X_test)

        if base_target_series is not None:
            base_val = base_target_series.reshape(-1, 1)
            actual_pred = delta_pred + base_val
            return actual_pred

    def save(
            self,
            model_name: str,
            save_dir: str = 'models'
        ):
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{model_name}.pkl')
        joblib.dump(self, model_path)
        print(f'Model saved to: {model_path}')
        return model_path

    @classmethod
    def load(cls, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded_instance = joblib.load(model_path)
        print(f"Model successfully loaded from: {model_path}")
        return loaded_instance

class XGBMultiOutputModel:
    def __init__(
            self,
            n_estimators: int,
            learning_rate: float,
            max_depth: int,
            subsample: float,
            colsample_bytree: float,
            early_stopping_rounds: int,
            random_state: int,
            ):
        
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'early_stopping_rounds': early_stopping_rounds,
            'random_state': random_state,
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': 'mae',
            'n_jobs': -1,
        }

        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """모델 학습"""
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
                       )
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame, base_target_series: pd.Series = None):
            """예측 수행"""
            if not self.is_fitted:
                raise RuntimeError("모델이 아직 학습되지 않았습니다. fit()을 먼저 실행하세요.")
    
            self.model.set_params(device="cpu")

            delta_pred = self.model.predict(X_test)
    
            if base_target_series is not None:
                base_val = base_target_series.reshape(-1, 1)
                actual_pred = delta_pred + base_val
                return actual_pred
    
    def save(
            self,
            model_name: str,
            save_dir: str = 'models'
        ):
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{model_name}.pkl')
        joblib.dump(self, model_path)
        print(f'Model saved to: {model_path}')
        return model_path

    @classmethod
    def load(cls, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded_instance = joblib.load(model_path)
        print(f"Model successfully loaded from: {model_path}")
        return loaded_instance

class CatBoostMultiOutputModel:
    def __init__(
            self,
            n_estimators: int,
            learning_rate: float,
            max_depth: int,
            early_stopping_rounds: int,
            random_state: int,
            ):

        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'early_stopping_rounds': early_stopping_rounds,
            'random_state': random_state,
            'bootstrap_type': 'Bayesian',
            'task_type': 'GPU',
            'loss_function': 'MultiRMSE',
            'eval_metric': 'MultiRMSE',
            'verbose': 100,
        }

        self.model = CatBoostRegressor(**self.params)
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """모델 학습"""
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            )
        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame, base_target_series: pd.Series = None):
        """예측 수행"""
        if not self.is_fitted:
            raise RuntimeError("모델이 아직 학습되지 않았습니다. fit()을 먼저 실행하세요.")

        delta_pred = self.model.predict(X_test)

        if base_target_series is not None:
            base_val = base_target_series.reshape(-1, 1)
            actual_pred = delta_pred + base_val
            return actual_pred
    
    def save(
            self,
            model_name: str,
            save_dir: str = 'models'
        ):
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{model_name}.pkl')
        joblib.dump(self, model_path)
        print(f'Model saved to: {model_path}')
        return model_path

    @classmethod
    def load(cls, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded_instance = joblib.load(model_path)
        print(f"Model successfully loaded from: {model_path}")
        return loaded_instance