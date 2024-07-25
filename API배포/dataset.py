from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import joblib
import torch
import torch.nn as nn

# FastAPI 인스턴스 생성
app = FastAPI()

# 모델 정의
class InsuranceModel(nn.Module):
    def __init__(self, input_dim):
        super(InsuranceModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# 입력 차원 설정 (예: 9개의 원본 특성과 8개의 파생 특성, 총 17개의 특성)
input_dim = 17

# 모델 및 스케일러 로드
model = joblib.load('insurance_model_pickle.pkl')
model.eval()
scaler = joblib.load('scaler.pkl')

# 특성 엔지니어링 함수
def feature_engineering(df):
    df['Age_Vehicle_Age'] = df['Age'] * df['Vehicle_Age']
    df['Age_Previously_Insured'] = df['Age'] * df['Previously_Insured']
    df['Vehicle_Age_Damage'] = df['Vehicle_Age'] * df['Vehicle_Damage']
    df['Previously_Insured_Damage'] = df['Previously_Insured'] * df['Vehicle_Damage']
    df['Age_squared'] = df['Age'] ** 2
    df['Vehicle_Age_squared'] = df['Vehicle_Age'] ** 2
    df['Annual_Premium_per_Age'] = df['Annual_Premium'] / (df['Age'] + 1)
    return df

# 데이터 전처리 함수
def preprocess(dataframe):
    # 특성 엔지니어링 적용
    dataframe = feature_engineering(dataframe)
    
    # 표준화 적용
    dataframe = scaler.transform(dataframe)
    
    # 텐서로 변환
    return torch.tensor(dataframe, dtype=torch.float32)

# 예측 함수
def predict(input_tensor):
    # 추론
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy().tolist()

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        # 업로드된 CSV 파일을 읽습니다
        df = pd.read_csv(file.file)
        
        # 데이터 전처리
        input_tensor = preprocess(df)
        
        # 예측 수행
        result = predict(input_tensor)
        
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





