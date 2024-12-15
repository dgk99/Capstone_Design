import cv2
import os
import pandas as pd
from sklearn.utils import resample

def preprocess_and_oversample(input_dir, output_dir, csv_file):
    """
    Preprocess images by resizing and performing oversampling for class imbalance.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save processed images.
        csv_file (str): CSV file containing image filenames and steering angles.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. CSV 파일 로드
    data = pd.read_csv(csv_file)

    # 2. 각도별 데이터 그룹화 및 오버샘플링
    angle_groups = [data[data['Steering Angle'] == angle] for angle in data['Steering Angle'].unique()]
    max_count = max([len(group) for group in angle_groups])  # 최대 데이터 수 확인

    oversampled_data = []
    for group in angle_groups:
        oversampled_group = resample(group, replace=True, n_samples=max_count, random_state=42)
        oversampled_data.append(oversampled_group)

    # 오버샘플링된 데이터셋 결합
    final_data = pd.concat(oversampled_data).reset_index(drop=True)

    # 3. 이미지 전처리 및 저장
    processed_data = []  # 새로운 CSV 데이터를 담을 리스트
    for idx, row in final_data.iterrows():
        image_file = row['Filename']
        angle = row['Steering Angle']

        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"processed_{idx:05d}.jpg")

        # 이미지 로드
        image = cv2.imread(input_path)
        if image is None:
            print(f"이미지 로드 실패: {input_path}")
            continue

        # Resize만 수행
        resized_image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)

        # 처리된 이미지 저장
        cv2.imwrite(output_path, resized_image)
        processed_data.append({'Filename': f"processed_{idx:05d}.jpg", 'Steering Angle': angle})
        print(f"Processed and saved: {output_path}")

    # 4. 새로운 CSV 파일 저장
    processed_csv_path = os.path.join(output_dir, "oversampled_data_new_1215_5.csv")
    pd.DataFrame(processed_data).to_csv(processed_csv_path, index=False)
    print(f"Oversampled and processed data saved to {processed_csv_path}")

# Example usage
input_dir = r"C:\Users\USER\Desktop\done\session_20241216_023958_mm\session_20241216_023958"  # 원본 이미지 폴더
output_dir = r"C:\Users\USER\Desktop\resizing_oversampled_new_1215_5"  # 오버샘플링 및 전처리된 이미지 저장 폴더
csv_file = r"C:\Users\USER\Desktop\done\session_20241216_023958_mm\session_20241216_023958\steering_angles_mm.csv"  # 원본 CSV 파일
preprocess_and_oversample(input_dir, output_dir, csv_file)
