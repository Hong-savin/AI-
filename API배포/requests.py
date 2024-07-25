import requests

url = "http://127.0.0.1:8000/predict"

# 입력 데이터
input_data = {
    "Gender" : 0,
    "Age" : 41,
    "Driving_License": 1,
    "Region_Code": 28,
    "Previously_Insured": 0,
    "Vehicle_Age": 2,
    "Vehicle_Damage": 1,
    "Annual_Premium": 30000,
    "Policy_Sales_Channel": 26
}

# POST 요청 보내기
response = requests.post(url, json=input_data)

# 응답 결과 확인
print(response.json())