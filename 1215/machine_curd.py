import os
import shutil
import cv2
import numpy as np

def calculate_sharpness(image_path):
    """
    이미지의 선명도(Sharpness)를 계산합니다.
    Laplacian 연산자의 분산을 반환합니다.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 이미지를 그레이스케일로 불러옴
    if image is None:
        return None
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()  # Laplacian 값의 분산 계산
    return variance

def auto_review_blurry_photos(photo_dir, blurry_dir, threshold=100.0):
    """
    흐린 사진을 검출하고 흐린 사진을 blurry_dir로 이동합니다.

    Args:
        photo_dir (str): 원본 사진 폴더 경로
        blurry_dir (str): 흐린 사진을 저장할 폴더 경로
        threshold (float): 흐림 판단을 위한 Laplacian 분산 임계값
    """
    os.makedirs(blurry_dir, exist_ok=True)  # 흐린 사진 폴더 생성

    image_files = [f for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"총 {len(image_files)}장의 사진을 검사합니다.\n")

    for index, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(photo_dir, image_file)
        sharpness = calculate_sharpness(image_path)

        if sharpness is None:
            print(f"[{index}] {image_file}: 불러올 수 없는 이미지입니다.")
            continue

        print(f"[{index}] {image_file}: Sharpness = {sharpness:.2f}")

        if sharpness < threshold:
            # 흐린 사진으로 판단하고 이동
            shutil.move(image_path, os.path.join(blurry_dir, image_file))
            print(f"    -> 흐린 사진으로 이동 (임계값 {threshold} 이하)\n")

    print("흐린 사진 검출 및 이동이 완료되었습니다.")

if __name__ == "__main__":
    photo_dir = input("검열할 사진 폴더 경로를 입력하세요: ").strip()
    blurry_dir = input("흐린 사진을 저장할 경로를 입력하세요: ").strip()
    threshold = float(input("흐림 판단을 위한 Sharpness 임계값을 입력하세요 (기본: 100): ") or 100)

    if not os.path.exists(photo_dir):
        print("사진 폴더 경로가 존재하지 않습니다.")
    else:
        auto_review_blurry_photos(photo_dir, blurry_dir, threshold)
