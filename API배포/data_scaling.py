from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# 실제 데이터를 로드합니다.
data = pd.read_csv('../test_X_original.csv')


# StandardScaler를 학습시킵니다.
scaler = StandardScaler()
scaler.fit(data)

# 학습된 스케일러를 저장합니다.
joblib.dump(scaler, 'scaler.pkl')