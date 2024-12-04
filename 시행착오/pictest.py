import cv2
import numpy as np
import os

# 이미지 파일들이 있는 폴더 경로 설정
folder_path = r'C:\Users\USER\Desktop\capstone\dataset1'  # 이미지 파일이 있는 폴더의 경로를 설정합니다.

# 폴더 내 모든 이미지 파일 처리
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # 지원하는 이미지 확장자
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read image file {filename}.")
            continue

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

        # 이미지 크기 조정
        resized_image = cv2.resize(contour_image, (800, 600))  # 원하는 크기로 조정
        filtered_resized = cv2.resize(filtered_image, (800, 600))  # 필터링된 이미지 크기도 조정

        # # 결과 출력 및 저장
        cv2.imshow("Filtered White Lines", filtered_resized)
        cv2.imshow("Contours", resized_image)

        # # 결과를 저장 (원하는 경로와 파일 이름 설정)
        # result_path = os.path.join(folder_path, f'contour_{filename}')
        # cv2.imwrite(result_path, resized_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

# 모든 창 닫기
cv2.destroyAllWindows()
