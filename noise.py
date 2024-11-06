import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 불러오기 (경로에 맞게 수정)
image_path = r'C:\Users\USER\Desktop\capstone\track picture.png'
image = cv2.imread(image_path)

# 이미지를 그레이스케일로 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 노이즈 추가 함수
def add_gaussian_noise(image):
    row, col = image.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col)).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image

# 소금-후추 노이즈 추가 함수
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    row, col = image.shape
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)
    
    # 소금 (흰 점) 추가
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    # 후추 (검은 점) 추가
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

# 스페클 노이즈 추가 함수
def add_speckle_noise(image):
    row, col = image.shape
    gauss = np.random.randn(row, col)
    noisy_image = image + image * gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return noisy_image

# 노이즈 제거 필터 적용
def remove_noise(image, noise_type):
    if noise_type == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif noise_type == 'salt_pepper':
        return cv2.medianBlur(image, 5)
    elif noise_type == 'speckle':
        return cv2.bilateralFilter(image, 9, 75, 75)
    else:
        return image

# 노이즈 추가한 이미지 생성
gaussian_noise_image = add_gaussian_noise(gray_image)
salt_pepper_noise_image = add_salt_and_pepper_noise(gray_image, 0.02, 0.02)
speckle_noise_image = add_speckle_noise(gray_image)

# 노이즈 제거한 이미지 생성
denoised_gaussian = remove_noise(gaussian_noise_image, 'gaussian')
denoised_salt_pepper = remove_noise(salt_pepper_noise_image, 'salt_pepper')
denoised_speckle = remove_noise(speckle_noise_image, 'speckle')

# 이미지 비교를 위한 그래프 생성
plt.figure(figsize=(12, 12))

# 1. 원본 이미지
plt.subplot(3, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 2. 가우시안 노이즈 이미지
plt.subplot(3, 2, 2)
plt.imshow(gaussian_noise_image, cmap='gray')
plt.title('Gaussian Noise')
plt.axis('off')

# 3. 소금-후추 노이즈 이미지
plt.subplot(3, 2, 3)
plt.imshow(salt_pepper_noise_image, cmap='gray')
plt.title('Salt and Pepper Noise')
plt.axis('off')

# 4. 스페클 노이즈 이미지
plt.subplot(3, 2, 4)
plt.imshow(speckle_noise_image, cmap='gray')
plt.title('Speckle Noise')
plt.axis('off')

# 5. 노이즈 제거 후 (모든 노이즈 제거된 이미지)
plt.subplot(3, 2, 5)
plt.imshow(denoised_gaussian, cmap='gray')
plt.title('Denoised Gaussian')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(denoised_salt_pepper, cmap='gray')
plt.title('Denoised Salt and Pepper')
plt.axis('off')

plt.tight_layout()
plt.show()
