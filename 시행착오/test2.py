import cv2
import numpy as np

# 비디오 파일 경로 설정
video_path = r'C:\Users\USER\Desktop\capstone\10.05 open cv test2.mp4'

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR 이미지를 HSV로 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 하얀색 범위 설정 (HSV 범위 조정)
        lower_white = np.array([0, 0, 200])  # 하얀색 하한 값
        upper_white = np.array([180, 50, 255])  # 하얀색 상한 값

        # 하얀색만 필터링
        mask = cv2.inRange(hsv, lower_white, upper_white)
        filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 그레이스케일로 변환
        gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)

        # CLAHE 명암 조정 (대비 강화)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_frame = clahe.apply(gray)

        # 가우시안 블러 적용 (노이즈 제거)
        blur = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)

        # Canny 엣지 검출 (임계값 조정)
        edges = cv2.Canny(blur, 40, 150)  # 하한값 40, 상한값 150

        # 허프 변환을 사용해 선 검출
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=200)

        # 검출된 선 그리기
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)  # 빨간색 선으로 차선 그리기

        # 원본 프레임과 선을 중첩
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

        # 결과 화면에 출력
        cv2.imshow("Lane Tracking", result)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
