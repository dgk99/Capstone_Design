import cv2

# Haar Cascade 파일 로드 (얼굴 탐지)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 카메라 연결
cap = cv2.VideoCapture(0)

while True:
    # 카메라에서 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 그레이스케일 변환 (Haar Cascade는 그레이스케일 이미지에서 동작)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 탐지
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    # 얼굴 위치에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 결과 출력
    cv2.imshow('Face Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 릴리스 및 창 닫기
cap.release()
cv2.destroyAllWindows()
