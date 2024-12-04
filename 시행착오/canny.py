import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 불러오기
image_path = r'C:\Users\USER\Desktop\capstone\track picture.png'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. 전처리 없이 바로 Canny 엣지 검출
edges_without_denoise = cv2.Canny(gray_image, 100, 200)

# 2. 전처리 (Gaussian Blur 적용 후) Canny 엣지 검출
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
edges_with_denoise = cv2.Canny(blurred_image, 100, 200)

# 결과 비교
plt.figure(figsize=(10,5))

# 전처리 없이 엣지 검출
plt.subplot(1, 2, 1)
plt.imshow(edges_without_denoise, cmap='gray')
plt.title('Canny without Denoise')
plt.axis('off')

# 전처리 후 엣지 검출
plt.subplot(1, 2, 2)
plt.imshow(edges_with_denoise, cmap='gray')
plt.title('Canny with Denoise')
plt.axis('off')

plt.show()
