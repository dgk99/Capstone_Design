import cv2
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from sklearn.utils import resample

def preprocess_and_oversample_images(input_dir, output_dir, csv_file):
    """
    Preprocess images by cropping, resizing, and performing oversampling for class imbalance.
    
    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save processed and oversampled images.
        csv_file (str): CSV file containing image filenames and steering angles.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # CSV 파일 로드
    data = pd.read_csv(csv_file)
    
    # 각도별 데이터 분리
    angle_groups = {angle: data[data['Steering Angle'] == angle] for angle in data['Steering Angle'].unique()}
    
    # 다수 클래스 데이터 크기 확인
    max_count = max(len(group) for group in angle_groups.values())
    
    # 데이터 증강을 위한 transform 설정
    transform = transforms.Compose([
        transforms.RandomRotation(5),  # ±5도 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2)  # 밝기 및 대비 조정
    ])
    
    oversampled_data = []

    for angle, group in angle_groups.items():
        # 데이터 복제 및 오버샘플링
        oversampled_group = resample(group, replace=True, n_samples=max_count, random_state=42)
        
        for _, row in oversampled_group.iterrows():
            img_path = os.path.join(input_dir, row['Filename'])
            output_path = os.path.join(output_dir, row['Filename'])

            # 이미지 로드
            image = cv2.imread(img_path)
            if image is None:
                print(f"이미지 로드 실패: {img_path}")
                continue

            # 상단 20% Crop
            H, W, _ = image.shape
            cropped_image = image[int(H * 0.4):, :]  # 상단 20% 제거
            
            # 크기 조정 (66x200)
            resized_image = cv2.resize(cropped_image, (200, 66), interpolation=cv2.INTER_AREA)
            
            # OpenCV 이미지를 PIL 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            
            # 데이터 증강
            augmented_image = transform(pil_image)
            
            # 증강된 이미지를 저장
            augmented_image.save(output_path)
            oversampled_data.append({'Filename': row['Filename'], 'Steering Angle': angle})
            print(f"Processed and saved: {output_path}")
    
    # 오버샘플링된 데이터를 새로운 CSV 파일로 저장
    oversampled_csv_path = os.path.join(output_dir, 'oversampled_data_new_1215.csv')
    pd.DataFrame(oversampled_data).to_csv(oversampled_csv_path, index=False)
    print(f"Oversampled data saved to {oversampled_csv_path}")

# Example usage
input_dir = "C:\\Users\\USER\\Desktop\\session_20241215_230929_sungsik\\session_20241215_230929"  # 원본 이미지 폴더
output_dir = "C:\\Users\\USER\\Desktop\\241215_resizing_new_1215"  # 전처리 및 오버샘플링된 이미지 저장 폴더
csv_file = "C:\\Users\\USER\\Desktop\\session_20241215_230929_sungsik\\session_20241215_230929\\steering_angles_new_1215.csv"  # CSV 파일
preprocess_and_oversample_images(input_dir, output_dir, csv_file)
