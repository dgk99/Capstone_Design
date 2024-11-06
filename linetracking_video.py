import cv2
import numpy as np

# 관심 영역(ROI)을 설정하는 함수
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # ROI에서 사용할 마스크 색상
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# 선을 그리는 함수
def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

# 주어진 RGB 색상(234, 234, 230)을 필터링하는 함수
def filter_specific_color(image):
    # BGR 이미지를 HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV 범위 설정
    lower_hsv = np.array([0, 0, 200])  # 밝은 색에 대한 하한
    upper_hsv = np.array([180, 25, 255])  # 밝은 색에 대한 상한

    # 특정 색상 필터링
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result, mask

# 프레임을 처리하는 함수 (차선 인식)
def process_frame(frame):
    height, width = frame.shape[:2]

    # 특정 색상 필터 적용
    filtered_image, mask = filter_specific_color(frame)

    # 그레이스케일 변환
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용 (노이즈 제거)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Canny 엣지 검출
    canny_image = cv2.Canny(blur_image, 50, 150)

    # 관심 영역 설정 (도로 부분만 남기기)
    region_of_interest_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # 허프 변환을 통한 선 검출
    lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi / 180, threshold=30, lines=np.array([]),
                            minLineLength=30, maxLineGap=150)

    # 차선 그리기
    line_image = draw_lines(frame, lines)

    return line_image, mask, canny_image

# 비디오 파일 읽기
video_path = r'C:\Users\USER\Desktop\capstone\10.05 open cv test.mp4'  # 비디오 경로 설정
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

        # 결과 화면에 출력 (HSV 마스크, Canny 엣지, 최종 이미지)
        cv2.imshow('Filtered Frame (Mask)', mask_frame)
        cv2.imshow('Canny Edges', canny_frame)
        cv2.imshow('Lane Detection (Video)', processed_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
