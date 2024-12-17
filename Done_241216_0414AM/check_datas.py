import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 로드
csv_file = r'C:\Users\USER\Desktop\done\Real Finish\데이터 셋 리사이징 및 오버샘플링\oversampled_data_new_1215_5.csv'
data = pd.read_csv(csv_file)

# 각도별 데이터 개수 확인
angle_counts = data['Steering Angle'].value_counts()
angle_counts.sort_index(inplace=True)

# 그래프 출력
plt.bar(angle_counts.index, angle_counts.values)
plt.xlabel('Steering Angle')
plt.ylabel('Count')
plt.title('Distribution of Steering Angles')
plt.show()
