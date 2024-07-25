from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import xgboost as xgb

# XGBoost 모델 로드
model = xgb.Booster()
model.load_model('xgboost_model_json.json')

# Scaler 로드
scaler = joblib.load('scaler.pkl')

app = FastAPI()

class InputData(BaseModel):
    Gender: int
    Age: int
    Driving_License: int
    Region_Code: int
    Previously_Insured: int
    Vehicle_Age: int
    Vehicle_Damage: int
    Annual_Premium: int
    Policy_Sales_Channel: int
    Vintage: int

class OutputData(BaseModel): 
    Response: int

@app.post("/predict-response", response_model=OutputData)
def predict_response(application: InputData):
    try:
        input_data = dict(application)

        input_df = pd.DataFrame([input_data])

        input_df_scaled = scaler.transform(input_df)

        dmatrix = xgb.DMatrix(input_df_scaled)
        prediction = model.predict(dmatrix)

        response = 1 if prediction[0] > 0.4 else 0

        return {"Response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))