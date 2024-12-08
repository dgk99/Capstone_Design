import cv2
import Jetson.GPIO as GPIO
import time
import curses
from threading import Thread, Event
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import PilotNet  # 학습된 모델 import (저장된 모델 파일을 사용)

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
servo_motor_pin = 33       # 서보 모터 핀
dc_motor_pwm_pin = 32      # DC 모터 PWM 핀
dc_motor_dir_pin1 = 29     # DC 모터 방향 제어 핀 1
dc_motor_dir_pin2 = 31     # DC 모터 방향 제어 핀 2

# 서보 모터 PWM 설정 (50Hz)
GPIO.setup(servo_motor_pin, GPIO.OUT)
servo = GPIO.PWM(servo_motor_pin, 50)
servo.start(7.5)  # 서보 초기 위치

# DC 모터 설정
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 2000)  # 2kHz 주파수로 설정
dc_motor_pwm.start(0)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None  # 비디오 파일 초기화

# 상태 변수
current_direction = "stop"
paused = True
is_recording = Event()
running = True

# 모델 로드 함수
def load_model(model_path='pilotnet_model.pth'):
    model = PilotNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 모델 입력 크기 맞추기
    transforms.ToTensor(),        # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 서보 모터 각도 설정
def set_servo_angle(angle):
    duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 180) * 10))
    servo.ChangeDutyCycle(duty_cycle)

# DC 모터 속도 및 방향 설정 (부스터 포함)
def set_dc_motor(speed, direction, booster=False):
    global current_direction, paused
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        current_direction = "forward"
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        current_direction = "backward"
    elif direction == "stop":
        speed = 0
        current_direction = "stop"
    
    # 부스터 작동
    if booster:
        dc_motor_pwm.ChangeDutyCycle(100)  # 최대 속도
        time.sleep(0.5)                    # 부스터 지속 시간
    dc_motor_pwm.ChangeDutyCycle(speed)   # 설정 속도로 전환

# 녹화 스레드
def record_video():
    global out
    while running:
        if is_recording.is_set():  # 녹화 상태일 때만
            ret, frame = cap.read()
            if ret and out is not None:
                out.write(frame)
                cv2.imshow('Recording', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        time.sleep(0.1)  # 스레드 부하를 줄이기 위해 약간의 대기 시간

# 차량 제어 함수 (모델 예측 기반)
def control_vehicle_with_model(model):
    ret, frame = cap.read()
    if ret:
        # 1. 이미지를 PIL 이미지로 변환
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 2. 이미지 전처리 및 텐서 변환
        image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

        # 3. 모델 예측
        with torch.no_grad():
            steering_angle = model(image_tensor)

        # 4. 예측된 조향 값 출력 및 차량 제어
        steering_angle_value = steering_angle.item()
        set_servo_angle(steering_angle_value)
        set_dc_motor(50, "forward")  # 50% 속도로 전진

# curses 기반 차량 제어
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    # 모델 로드
    model = load_model('pilotnet_model.pth')

    global paused, out

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP:  # 전진
            if paused:
                paused = False
                is_recording.set()  # 녹화 시작
                if out is None:
                    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480), isColor=True)
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개 및 녹화 시작")
            # 직진하면서 모델 기반으로 조향 제어
            control_vehicle_with_model(model)
            stdscr.addstr(2, 0, "DC Motor: Forward at 50% speed")

        elif key == curses.KEY_DOWN:  # 후진
            if paused:
                paused = False
                is_recording.set()  # 녹화 시작
                if out is None:
                    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480), isColor=True)
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개 및 녹화 시작")
            set_dc_motor(50, "backward")
            stdscr.addstr(2, 0, "DC Motor: Backward at 50% speed")

        elif key == ord(' '):  # 일시정지
            if not paused:
                set_dc_motor(0, "stop")
                is_recording.clear()  # 녹화 일시정지
                paused = True
                stdscr.addstr(4, 0, "일시정지 상태")
            else:
                paused = False
                is_recording.set()
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개 및 녹화 시작")

        elif key == ord('e'):  # 부스터
            stdscr.addstr(5, 0, "Booster Activated!")
            set_dc_motor(50, "forward", booster=True)

        elif key == ord('q'):  # 종료
            print("종료합니다.")
            break

try:
    # 녹화 스레드 시작
    video_thread = Thread(target=record_video)
    video_thread.start()

    # curses 기반 메인 루프 실행
    curses.wrapper(main)

except KeyboardInterrupt:
    print("사용자에 의해 중단됨")

finally:
    running = False
    video_thread.join()  # 스레드 종료 대기
    if out is not None:
        out.release()
    cap.release()
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("GPIO 핀 정리 완료 및 영상 저장 완료")
