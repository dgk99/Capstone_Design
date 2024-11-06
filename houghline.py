import cv2
import numpy as np

# 이미지 읽기
image_path = r'C:\Users\USER\Desktop\capstone\track picture.png'
image = cv2.imread(image_path)  # 이미지를 읽어옵니다
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

# 노이즈 제거 (Gaussian Blur 적용)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)

# Canny 엣지 검출
edges = cv2.Canny(blurred_image, 50, 150)

# 허프 변환을 사용하여 직선 검출
lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

# 검출된 직선을 이미지에 그리기
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # 직선을 이미지에 그리기
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 결과 보기
cv2.imshow("Detected Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

