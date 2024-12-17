import cv2

def calculate_sharpness(image_path):
    """Laplacian을 사용해 이미지 선명도 계산"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance

# 흐린 사진의 Laplacian 값 측정
image_path = "C:\\Users\\USER\\Desktop\\DGK\\mygithub\\Capstone_Design\\1213\\deleted_image\\photo_0192.jpg"  # 흐린 사진의 경로 입력
sharpness = calculate_sharpness(image_path)

if sharpness is not None:
    print(f"흐린 사진의 Laplacian 선명도 값: {sharpness:.2f}")
else:
    print("이미지를 불러올 수 없습니다.")
