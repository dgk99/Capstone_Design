import cv2
import os

def preprocess_images(input_dir, output_dir):
    """
    Preprocess images by cropping and resizing.
    
    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save processed images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 파일 리스트 가져오기
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
    
    for image_file in image_files:
        # 이미지 경로
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        # 이미지 로드
        image = cv2.imread(input_path)
        if image is None:
            print(f"이미지 로드 실패: {input_path}")
            continue
        
        # 상단 30% Crop
        H, W, _ = image.shape
        cropped_image = image[int(H * 0.3):, :]  # 상단 30% 제거
        
        # 크기 조정 (66x200)
        resized_image = cv2.resize(cropped_image, (200, 66), interpolation=cv2.INTER_AREA)
        
        # 처리된 이미지 저장
        cv2.imwrite(output_path, resized_image)
        print(f"Processed and saved: {output_path}")

# Example usage
input_dir = "C:\\Users\\USER\\Desktop\\20241213_192414"  # 원본 이미지 폴더
output_dir = "C:\\Users\\USER\\Desktop\\241213_resizing"  # 전처리된 이미지 저장 폴더
preprocess_images(input_dir, output_dir)
