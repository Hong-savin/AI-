from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler

# Load the pre-trained XGBoost model from a pickle file
model = joblib.load('xgboost_model_pickle.pkl')

# Initialize StandardScaler and fit it with some dummy data (or real data if available)
scaler = StandardScaler()
dummy_data = pd.DataFrame({
    'gender': [0],
    'age': [30],
    'Driving_License': [1],
    'Region_Code': [28],
    'Previously_Insured': [0],
    'Vehicle_Age': [1],
    'Vehicle_Damage': [1],
    'Annual_Premium': [50000],
    'Policy_Sales_Channel': [152],
    'Vintage': [100]
})
scaler.fit(dummy_data)

app = FastAPI()

class InputData(BaseModel):
    gender: int
    age: int
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

@app.post("/predict_response", response_model=OutputData)
def predict_response(application: InputData):
    try:
        input_data = dict(application)
        
        input_df = pd.DataFrame([input_data])
        
        # Transform the input data using the pre-fit scaler
        input_df_scaled = scaler.transform(input_df)
        
        dmatrix = xgb.DMatrix(input_df_scaled)
        prediction = model.predict(dmatrix)
        
        response = 1 if prediction[0] > 0.1 else 0
        
        return {"Response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))