import cv2
import numpy as np

# 관심 영역(ROI)을 설정하는 함수
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # ROI에서 사용할 마스크 색상
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# 기울기를 계산하는 함수
def calculate_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 9999  # x 좌표 차이가 0일 경우 기울기를 크게 설정

# 선을 그리는 함수 (빨간색으로 차선을 그리기)
def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = calculate_slope(x1, y1, x2, y2)
                # 차선의 기울기 조건 (수직에 가까운 선만 인식, 너무 평평하거나 수평선 제외)
                if 0.5 < abs(slope) < 2:  # 기울기 필터
                    cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 5)  # 빨간색으로 차선을 표시

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)  # 원본 영상과 차선 그리기
    return img

# 주어진 RGB 색상(234, 234, 230)을 필터링하는 함수
def filter_specific_color(image):
    # BGR 이미지를 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV 범위 설정 (하얀색 필터링 범위 조정)
    lower_hsv = np.array([0, 0, 200])  # 하얀색에 대한 HSV 하한 (밝기를 조금 높게 설정)
    upper_hsv = np.array([180, 25, 255])  # 하얀색에 대한 HSV 상한

    # 특정 색상 필터링
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result, mask

# 프레임을 처리하는 함수 (차선 인식)
def process_frame(frame):
    height, width = frame.shape[:2]

    # 특정 색상 필터 적용 (하얀색만 필터링)
    filtered_image, mask = filter_specific_color(frame)

    # 그레이스케일 변환
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용 (노이즈 제거) -> 커널 크기를 키워 더 강하게 노이즈 제거
    blur_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # Canny 엣지 검출 -> 더 높은 임계값으로 잔상 제거
    canny_image = cv2.Canny(blur_image, 70, 150)

    # 관심 영역 설정 (도로 부분만 남기기)
    region_of_interest_vertices = [
        (int(width * 0.2), height),  # 좌측 하단
        (int(width * 0.4), int(height * 0.65)),  # 좌측 중간 (조금 높게 설정)
        (int(width * 0.6), int(height * 0.65)),  # 우측 중간
        (int(width * 0.8), height)  # 우측 하단
    ]
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # 허프 변환을 통한 선 검출 -> 선의 최소 길이와 간격을 증가시켜 더 명확한 선만 검출
    lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi / 180, threshold=50, lines=np.array([]),
                            minLineLength=50, maxLineGap=100)

    # 차선 그리기
    line_image = draw_lines(frame, lines)

    return line_image, mask, canny_image

# 비디오 파일 읽기 (차선 검출)
video_path = r'C:\Users\USER\Desktop\capstone\10.05 open cv test3.mp4'  # test3번 비디오 경로
cap = cv2.VideoCapture(video_path)

# 비디오 파일이 제대로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 프레임을 처리하여 차선 인식
        processed_frame, mask_frame, canny_frame = process_frame(frame)

        # 결과 화면에 출력 (필터된 흑백 마스크, Canny 엣지, 컬러 차선)
        cv2.imshow('Filtered Frame (Mask)', mask_frame)  # 흑백 마스크
        cv2.imshow('Canny Edges', canny_frame)  # Canny 엣지
        cv2.imshow('Lane Detection (Video)', processed_frame)  # 컬러 차선

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
