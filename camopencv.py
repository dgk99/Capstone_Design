import cv2

# 카메라 객체 생성 (0번 장치 사용)
cap = cv2.VideoCapture(0)

# 카메라 화면 비추기 루프
while cap.isOpened():
    ret, frame = cap.read()  # 프레임 읽기
    if ret:
        cv2.imshow('Live Camera', frame)  # 창에 프레임 출력
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
