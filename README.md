# AI-Team-Project

# 프로젝트 제목: 보험 예측을 위한 AI 딥러닝 모델

---

## 프로젝트 개요

이 레포지토리에는 보험 예측을 위해 개발된 다양한 머신러닝 모델과 관련 파일이 포함되어 있습니다. 프로젝트는 XGBoost 및 PyTorch와 같은 다양한 기법을 사용하여 구축된 모델과 전처리 및 훈련 과정을 상세히 설명하는 Jupyter 노트북을 포함합니다. 또한, API 배포를 위한 스크립트와 데이터 전처리 코드도 포함되어 있습니다.

캐글의 https://www.kaggle.com/competitions/playground-series-s4e7/overview 데이터를 이용하였습니다.

---

## 레포지토리 구조

- **모델**
  - `FinalModel_state_dict.pth`: 최종 훈련된 PyTorch 모델의 상태 사전 파일.
  - `insurance_model_pickle.pkl`: 보험 예측 모델의 직렬화된 파일.
  - `xgboost_model_json.json`: XGBoost 모델의 세부 정보를 포함한 JSON 파일.

- **노트북**
  - `머신러닝_xgboost.ipynb`: XGBoost 모델의 훈련 및 평가를 위한 Jupyter 노트북.
  - `전처리과정.ipynb`: 데이터 전처리 과정을 설명하는 Jupyter 노트북.
  - `최종_AI산학딥러닝.ipynb`: 최종 모델 훈련 및 성능 평가를 설명하는 Jupyter 노트북.

- **데이터**
  - `submission_file.csv`: 최종 제출 파일.
  - `test.csv`: kaggle 원본 데이터 파일.
  - `test_X.csv`: 테스트 데이터 (전처리 후).
  - `test_X_original.csv`: 테스트 데이터 (원본).
  - `test_y.csv`: 테스트 라벨 (전처리 후).
  - `test_y_original.csv`: 테스트 라벨 (원본).
  - `train_X.csv`: 훈련 데이터 (전처리 후).
  - `train_X_original.csv`: 훈련 데이터 (원본).
  - `train_y.csv`: 훈련 라벨 (전처리 후).
  - `train_y_original.csv`: 훈련 라벨 (원본).

- **API 배포**
  - `data_scaling.py`: 데이터 스케일링을 위한 스크립트.
  - `dataset.py`: 데이터셋 로딩 및 처리 관련 스크립트.
  - `main2.py`: API 배포를 위한 메인 스크립트 (버전 2).
  - `main3.py`: API 배포를 위한 메인 스크립트 (버전 3).
  - `requests.py`: API 요청을 테스트하기 위한 스크립트

- **__pycache__**
  - `main.cpython-312.pyc`: `main.py`의 컴파일된 파이썬 바이트코드 파일.
  - `main2.cpython-312.pyc`: `main2.py`의 컴파일된 파이썬 바이트코드 파일.

---

## 상세 설명

이 레포지토리는 보험 예측 모델의 개발, 평가 및 배포를 위한 포괄적인 자원을 제공합니다. 각 파일의 역할과 사용법에 대해 자세히 설명되어 있으며, 이를 통해 프로젝트의 모든 단계에서 유용하게 활용할 수 있습니다.

### 모델 파일
- **`FinalModel_state_dict.pth`**: PyTorch를 사용하여 훈련된 최종 모델의 상태 사전입니다. 이 파일은 모델의 가중치와 바이어스를 저장하고 있으며, 추론 시 로드하여 사용할 수 있습니다.
- **`insurance_model_pickle.pkl`**: 보험 예측 모델의 직렬화된 파일입니다. 이 파일은 모델을 직렬화하여 저장하고, 필요 시 역직렬화하여 사용할 수 있습니다.
- **`xgboost_model_json.json`**: XGBoost 모델의 세부 정보를 포함한 JSON 파일입니다. 이 파일은 모델의 구조, 파라미터 및 성능 지표를 포함하고 있습니다.

### 노트북 파일
- **`머신러닝_xgboost.ipynb`**: 이 Jupyter 노트북은 XGBoost 모델의 훈련 및 평가 과정을 설명합니다. 데이터 로드, 전처리, 모델 훈련 및 성능 평가에 대한 단계별 설명이 포함되어 있습니다.
- **`전처리과정.ipynb`**: 이 노트북은 데이터 전처리 과정을 설명합니다. 데이터 정리, 결측값 처리, 특성 공학 등 다양한 전처리 기법을 다루고 있습니다.
- **`최종_AI산학딥러닝.ipynb`**: 최종 모델의 훈련 및 성능 평가를 설명하는 노트북입니다. 여러 모델의 비교 및 최종 모델 선정 과정이 포함되어 있습니다.

### 데이터 파일
- **`submission_file.csv`**: 최종 제출 파일로, 예측 결과를 포함하고 있습니다.
- **`test.csv`**: kaggle에서 제공한 원본 데이터입니다.
- **`test_X.csv`**: 전처리 후의 테스트 데이터입니다.
- **`test_X_original.csv`**: 원본 테스트 데이터입니다.
- **`test_y.csv`**: 전처리 후의 테스트 라벨입니다.
- **`test_y_original.csv`**: 원본 테스트 라벨입니다.
- **`train_X_original.csv`**: 원본 훈련 데이터입니다.
- **`train_X.csv`**: 전처리 후의 훈련 데이터입니다.
- **`train_y.csv`**: 전처리 후의 훈련 라벨입니다.
- **`train_y_original.csv`**: 원본 훈련 라벨입니다.

### API 배포 파일
- **`data_scaling.py`**: 데이터 스케일링을 위한 스크립트로, 데이터를 표준화 또는 정규화하는 기능을 제공합니다.
- **`dataset.py`**: 데이터셋 로딩 및 처리 관련 스크립트로, 데이터를 불러오고 전처리하는 기능을 포함하고 있습니다.
- **`main2.py`**: API 배포를 위한 메인 스크립트 (버전 2)로, Flask 등을 이용하여 API 서버를 실행하는 코드를 포함하고 있습니다.
- **`main3.py`**: API 배포를 위한 메인 스크립트 (버전 3)로, `main2.py`의 개선된 버전입니다.
- **`requests.py`**: API 요청을 테스트하기 위한 스크립트로, HTTP 요청을 보내고 응답을 확인하는 기능을 포함하고 있습니다.

### __pycache__ 폴더 파일
- **`main.cpython-312.pyc`**: `main.py`의 컴파일된 파이썬 바이트코드 파일입니다.
- **`main2.cpython-312.pyc`**: `main2.py`의 컴파일된 파이썬 바이트코드 파일입니다.

---

이 레포지토리는 보험 예측 모델의 전 과정을 아우르며, 데이터 전처리, 모델 훈련, 평가 및 API 배포까지의 모든 단계를 포함합니다. 이를 통해 프로젝트의 모든 단계에서 유용하게 활용할 수 있습니다.
