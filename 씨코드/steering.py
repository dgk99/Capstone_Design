import cv2
import pandas as pd
import os

# 레이블 데이터를 저장할 리스트
labeled_data = []

# 이미지 폴더 경로
image_folder = "C:\\Users\\USER\\Desktop\\take_photo\\traning_data_set"
image_files = sorted(os.listdir(image_folder))  # 이미지 파일 정렬

# 레이블링 시작
for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    img = cv2.imread(image_path)
    
    # 이미지 표시
    cv2.imshow("Label Image", img)
    cv2.waitKey(1)  # 이미지 업데이트를 위해 잠깐 대기
    
    # 사용자 입력
    try:
        steering_angle = float(input(f"Enter steering angle for {image_name} (e.g., -30, 0, 15): "))
        labeled_data.append({"image_name": image_name, "steering_angle": steering_angle})
    except ValueError:
        print("Invalid input! Please enter a numeric value.")
    
    cv2.destroyAllWindows()  # 현재 이미지 창 닫기

# 결과 저장
output_csv = "labeled_dataset.csv"
pd.DataFrame(labeled_data).to_csv(output_csv, index=False)
print(f"Labeled dataset saved to {output_csv}")
