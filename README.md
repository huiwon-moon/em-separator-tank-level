# Separator Tank Level Forecasting Pipeline


Separator Tank Level의 다중 시점 미래 예측을 위한 시계열 ML/DL 파이프라인 시스템입니다.

---

## Key Features

- **다중 타겟/작업 지원**:
  - `SEP TANK LVL`: 미래 $t+1 \sim t+6$ 스텝 Multi-horizon 예측 (단위: 10s)
- **Feature Engineering 파이프라인**:
  - `caseID` 단위 시계열 그룹핑 기반 Lag, Diff, Rolling (Mean, Std) 자동 생성
- **통합 CLI & Orchestration**:
  - `main.py`, `DL_main.py`를 통합 제어 타워로 사용하여 `train`, `inference` 모드 일괄 제어

---

## Directory Structure

```text
├── data/                   # (비공개) 시계열 CSV 데이터
├── models/                 # (비공개) 학습 완료된 SEP TANK LVL ML/DL 모델 (.pkl/.pt)
├── config/
│   └── columns.py          # (비공개) feature & target 태그 정보
├── docs/                   # (추후 공개) 기술 설명서
├── fig/                    # (비공개) ML/DL vs Naive(단순 이동 예측) 평가 시각화
├── results/                # 추론 결과 CSV
├── src/
│   └── data_concat.py      # 데이터 병합
│   └── data_imputation.py  # 결측치 처리 (ffill & linear interpolation)
│   └── data_preprocess.py  # WET 모드 구간 추출
│   └── data_split.py       # 훈련/테스트 데이터 분할
│   └── evaluation.py       # 성능 평가 코드
│   └── features.py         # feature engineering
│   └── model.py            # ML 모델 구조
│   └── DL_model.py         # DL 모델 구조
│   └── visualization.py    # 시각화 코드
├── main.py                 # ML 파이프라인 실행 진입점 (Entry Point)
├── DL_main.py              # DL 파이프라인 실행 진입점 (Entry Point)
└── requirements.txt        # 의존성 라이브러리 목록
```
---

## Usage
### 1. 가상환경 생성 및 라이브러리 설치
```bash
python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 파이프라인 실행
```bash
python main.py --model_name xgb
python DL_main.py --model_name lstm
python DL_main.py --model_name gru --batch_size 128
python DL_main.py --model_name 1d_cnn --lookback 60 --step_size 30 --batch_size 256
```

---

## Validation & Saving
- **추론 메트릭**: `results/` 폴더 내에 Horizon별로 MAE, MSE, RMSE 메트릭이 포함된 CSV 파일 저장
- **시각화 결과**: 실제 SEP TANK LVL vs Naive (단순 이동 예측값) vs 모델 예측값 시계열 그래프 생성 (.png)

---

## CLI Arguments
### main.py
| 옵션 (Argument) | 타입 (Type) | 기본값 (Default) | 선택 가능 값 (Choices) | 설명 (Description) |
| :--- | :--- | :--- | :--- | :--- |
| `--n_estimators` | `int` | `20000` | 양의 정수 | 최대 트리 개수 |
| `--learning_rate` | `float` | `0.01` | 양의 실수 | 모델 학습률 |
| `--random_state` | `int` | `42` | 정수 | 재현성을 위한 난수 고정 시드값 |
| `--model_name` | `str` | `xgb` | `xgb`, `lgbm`, `catboost` | 모델 종류 |
| `--early_stopping_rounds` | `int` | `50` | 양의 정수 | 과적합 방지를 위한 조기 종료 |

### DL_main.py
| 옵션 (Argument) | 타입 (Type) | 기본값 (Default) | 선택 가능 값 (Choices) | 설명 (Description) |
| :--- | :--- | :--- | :--- | :--- |
| `--lookback` | `int` | `180` | 양의 정수 | 과거 입력 시퀀스 길이 |
| `--step_size` | `int` | `10` | 양의 정수 | 슬라이딩 윈도우 이동 간격 |
| `--batch_size` | `int` | `64` | 양의 정수 | 미니 배치 크기 |
| `--hidden_dim` | `int` | `512` | 양의 정수 | 은닉 상태의 차원 크기 |
| `--num_layers` | `int` | `2` | 양의 정수 | 레이어의 개수 |
| `--dropout` | `float` | `0.2` | 0.0 ~ 1.0 | 무작위로 뉴런을 끄는 Dropout 비율 |
| `--model_name` | `str` | `lstm` | `lstm`, `gru`, `1d_cnn` | 모델 종류 |

