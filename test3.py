import cv2
import numpy as np

# 이미지 파일 경로 설정
image_path = r'C:\Users\USER\Desktop\capstone\track picture.png'  # 이미지 파일 경로를 설정합니다.

# 이미지 파일 열기
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read image file.")
else:
    # BGR 이미지를 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 흰색 범위 설정 (HSV 범위 조정)
    lower_white = np.array([0, 0, 200])  # 흰색 하한 값
    upper_white = np.array([180, 25, 255])  # 흰색 상한 값

    # 흰색만 필터링
    mask = cv2.inRange(hsv, lower_white, upper_white)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    # 그레이스케일로 변환
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # 윤곽선 검출
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 흰색 선 윤곽선 그리기
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 3)  # 파란색 선으로 윤곽선 그리기

    # 결과 출력
    cv2.imshow("Filtered White Lines", filtered_image)
    cv2.imshow("Contours", contour_image)

    # 'q' 키를 누르면 종료
    cv2.waitKey(0)
    cv2.destroyAllWindows()
