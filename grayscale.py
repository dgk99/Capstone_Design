import cv2
from matplotlib import pyplot as plt

# 이미지 경로 (Windows 경로를 올바르게 지정)
image_path = r'C:\Users\USER\Desktop\capstone\track picture.png'

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지가 제대로 읽혔는지 확인
if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
else:
    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 원본 이미지와 그레이스케일 이미지 비교
    plt.figure(figsize=(10,5))

    # 원본 이미지 표시
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 그레이스케일 이미지 표시
    plt.subplot(1, 2, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # 이미지 출력
    plt.show()
