import cv2
import Jetson.GPIO as GPIO
import os
import time
import curses
import csv  # CSV 파일 저장용
from threading import Thread, Event

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

if not cap.isOpened():
    raise Exception("카메라를 열 수 없습니다.")

# 사진 저장 디렉토리 설정
photo_dir = "photos"  # 사진 저장 경로
os.makedirs(photo_dir, exist_ok=True)  # 디렉토리 생성

# 디버그 출력
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"사진 저장 디렉토리: {photo_dir}")

# 기존 파일에서 최대 번호 가져오기
existing_files = [f for f in os.listdir(photo_dir) if f.startswith("photo_") and f.endswith(".jpg")]
if existing_files:
    max_index = max(int(f.split('_')[1].split('.')[0]) for f in existing_files)
else:
    max_index = 0
photo_count = max_index + 1  # 새로운 사진 번호 시작

# CSV 파일 생성 및 초기화
csv_file = "labeled_dataset.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "steering_angle"])  # 헤더 추가

# 상태 변수
current_direction = "stop"
paused = True
is_recording = Event()
running = True
servo_angle = 95  # 초기 서보 모터 각도

# 서보 모터 각도 설정
def set_servo_angle(angle):
    global servo_angle
    servo_angle = angle  # 현재 각도 저장
    duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 180) * 10))
    servo.ChangeDutyCycle(duty_cycle)

# DC 모터 속도 및 방향 설정
def set_dc_motor(speed, direction):
    global current_direction
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
    
    dc_motor_pwm.ChangeDutyCycle(speed)

# 사진 촬영 및 레이블링
def capture_photos():
    global photo_count, servo_angle
    while running:
        if is_recording.is_set():  # 주행 중일 때만 촬영
            ret, frame = cap.read()
            if not ret:
                print("카메라에서 프레임을 읽지 못했습니다.")
                continue

            # 사진 저장
            photo_filename = os.path.join(photo_dir, f"photo_{photo_count:04d}.jpg")
            try:
                cv2.imwrite(photo_filename, frame)
                print(f"사진 저장 완료: {photo_filename}")
            except Exception as e:
                print(f"사진 저장 실패: {e}")
                continue

            # CSV 파일에 조향 값 기록
            try:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([photo_filename, servo_angle])
                    print(f"CSV 기록 완료: {photo_filename}, 각도: {servo_angle}")
            except Exception as e:
                print(f"CSV 기록 실패: {e}")

            photo_count += 1
            time.sleep(0.033)  # 약 30fps로 촬영 (간격 조절 가능)

# curses 기반 차량 제어
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    global paused

    servo_angle = 95
    dc_motor_speed = 50  # 기본 속도
    set_dc_motor(0, "stop")
    stdscr.addstr(0, 0, "위/아래 키로 전진/후진 전환, 좌우 키로 조향, 'q' 키로 종료")

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP:  # 전진
            if paused:
                paused = False
                is_recording.set()  # 사진 촬영 시작
            set_dc_motor(dc_motor_speed, "forward")
            stdscr.addstr(2, 0, "DC Motor: Forward at 50% speed")

        elif key == curses.KEY_DOWN:  # 후진
            if paused:
                paused = False
                is_recording.set()  # 사진 촬영 시작
            set_dc_motor(dc_motor_speed, "backward")
            stdscr.addstr(2, 0, "DC Motor: Backward at 50% speed")

        elif key == curses.KEY_LEFT:  # 좌회전
            servo_angle = max(0, servo_angle - 10)
            set_servo_angle(servo_angle)
            stdscr.addstr(3, 0, f"Servo angle set to {servo_angle} degrees")

        elif key == curses.KEY_RIGHT:  # 우회전
            servo_angle = min(180, servo_angle + 10)
            set_servo_angle(servo_angle)
            stdscr.addstr(3, 0, f"Servo angle set to {servo_angle} degrees")

        elif key == ord(' '):  # 일시정지
            if not paused:
                set_dc_motor(0, "stop")
                is_recording.clear()  # 사진 촬영 중지
                paused = True
                stdscr.addstr(4, 0, "일시정지 상태")
            else:
                paused = False
                is_recording.set()
                stdscr.addstr(4, 0, "일시정지 해제 및 사진 촬영 시작")

        elif key == ord('q'):  # 종료
            print("종료합니다.")
            break

try:
    # 사진 촬영 스레드 시작
    capture_thread = Thread(target=capture_photos)
    capture_thread.start()

    # curses 기반 메인 루프 실행
    curses.wrapper(main)

except KeyboardInterrupt:
    print("사용자에 의해 중단됨")

finally:
    running = False
    is_recording.clear()
    capture_thread.join()  # 스레드 종료 대기
    if cap.isOpened():
        cap.release()
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("GPIO 핀 정리 완료 및 프로그램 종료")
