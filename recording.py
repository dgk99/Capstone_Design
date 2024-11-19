import cv2

# 비디오 캡처 객체 생성 (카메라 장치 기본값 사용)
cap = cv2.VideoCapture(0)

# 비디오 코덱 설정 (예: MJPG 코덱)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 흑백 비디오 파일 저장 (이름과 프레임 속도, 해상도 지정)
out = cv2.VideoWriter('output_gray.avi', fourcc, 20.0, (640, 480), isColor=False)

print("Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            print("Error: Failed to capture image.")
            break

        # 프레임을 흑백으로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 흑백 프레임을 비디오 파일에 저장
        out.write(gray_frame)

        # 흑백 프레임 화면에 표시
        cv2.imshow('Gray Video', gray_frame)

        # 'q' 키를 누르면 녹화 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 모든 리소스 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped and file saved.")
