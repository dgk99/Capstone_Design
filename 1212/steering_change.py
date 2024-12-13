import pandas as pd

# 기존 CSV 파일 로드
input_csv = "1212driving.csv"  # 기존 CSV 파일 경로
output_csv = "1212dataset_changed.csv"  # 변환된 CSV 저장 경로

# 데이터 불러오기
data = pd.read_csv(input_csv)

# 각도 값을 95를 기준으로 상대적 값으로 변환
data["steering_angle"] = data["steering_angle"] - 95

# 변환된 데이터 저장
data.to_csv(output_csv, index=False)
print(f"Adjusted dataset saved to {output_csv}")
