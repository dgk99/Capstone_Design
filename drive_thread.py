import cv2
import Jetson.GPIO as GPIO
import threading
import time

# GPIO 핀 설정
SERVO_PIN = 33       # 서보 모터 핀
DC_PWM_PIN = 32      # DC 모터 PWM 핀
DC_DIR_PIN1 = 29     # DC 모터 방향 핀 1
DC_DIR_PIN2 = 31     # DC 모터 방향 핀 2

# 1. RC 자동차 초기화 함수
def init_rc_car():
    """
    RC 자동차 제어를 위한 초기화 함수
    """
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    GPIO.setup(DC_PWM_PIN, GPIO.OUT)
    GPIO.setup(DC_DIR_PIN1, GPIO.OUT)
    GPIO.setup(DC_DIR_PIN2, GPIO.OUT)

    # 서보 모터 및 DC 모터 PWM 설정
    global servo, dc_motor_pwm
    servo = GPIO.PWM(SERVO_PIN, 50)  # 서보 모터: 50Hz
    dc_motor_pwm = GPIO.PWM(DC_PWM_PIN, 2000)  # DC 모터: 2kHz

    servo.start(7.5)  # 서보 초기화 (90도)
    dc_motor_pwm.start(0)  # DC 모터 정지

# 2. RC 자동차 제어 함수
def control_rc_car():
    """
    RC 자동차의 서보 모터와 DC 모터를 제어하는 함수
    """
    try:
        while True:
            direction = input("Enter command (w: forward, s: backward, a: left, d: right, q: quit): ").strip()
            if direction == 'w':  # 전진
                GPIO.output(DC_DIR_PIN1, GPIO.HIGH)
                GPIO.output(DC_DIR_PIN2, GPIO.LOW)
                dc_motor_pwm.ChangeDutyCycle(50)  # 속도 50%
                print("Moving forward...")
            elif direction == 's':  # 후진
                GPIO.output(DC_DIR_PIN1, GPIO.LOW)
                GPIO.output(DC_DIR_PIN2, GPIO.HIGH)
                dc_motor_pwm.ChangeDutyCycle(50)  # 속도 50%
                print("Moving backward...")
            elif direction == 'a':  # 좌회전
                servo.ChangeDutyCycle(5)  # 약 60도
                print("Turning left...")
            elif direction == 'd':  # 우회전
                servo.ChangeDutyCycle(10)  # 약 120도
                print("Turning right...")
            elif direction == 'q':  # 종료
                print("Stopping RC car...")
                break
            else:
                print("Invalid command. Please use w, s, a, d, or q.")
    finally:
        # 모든 동작 정지
        dc_motor_pwm.ChangeDutyCycle(0)
        servo.ChangeDutyCycle(7.5)  # 중립
        print("RC car stopped.")

# 3. OpenCV 웹캠 촬영 및 저장 함수
def capture_video():
    """
    USB 웹캠에서 실시간 영상 촬영 및 저장
    """
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    print("Starting video capture...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out.write(frame)  # 영상 저장
                cv2.imshow('Video Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
                    break
            else:
                print("Failed to capture frame.")
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video capture stopped.")

# 4. 메인 함수
def main():
    """
    메인 실행 함수
    - RC 자동차 제어 : 메인 스레드에서 실행
    - 웹캠 촬영 및 저장: 별도 스레드에서 병렬 실행
    """
    # RC 자동차 초기화
    init_rc_car()

    # 비디오 캡처 스레드 생성 및 시작
    video_thread = threading.Thread(target=capture_video)
    video_thread.start()

    # RC 자동차 제어
    control_rc_car()

    # 비디오 스레드 종료 대기
    video_thread.join()
    GPIO.cleanup()
    print("Program terminated.")

# 5. 프로그램 실행
if __name__ == "__main__":
    main()
