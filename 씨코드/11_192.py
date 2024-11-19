import cv2
import Jetson.GPIO as GPIO
import time
import curses
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
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None  # 비디오 파일 초기화

# 상태 변수
current_direction = "stop"
paused = True  # 초기 상태는 일시정지 상태
is_recording = Event()  # 녹화 상태를 Event로 관리

# 스레드를 위한 플래그
running = True

# 서보 모터 각도 설정
def set_servo_angle(angle):
    duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 180) * 10))
    servo.ChangeDutyCycle(duty_cycle)

# DC 모터 속도 및 방향 설정
def set_dc_motor(speed, direction):
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
    dc_motor_pwm.ChangeDutyCycle(speed)

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

# curses 기반 차량 제어
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    servo_angle = 95
    dc_motor_speed = 100
    set_dc_motor(0, "stop")  # 초기 상태는 정지
    stdscr.addstr(0, 0, "위/아래 키로 전진/후진, 좌우 키로 조향, 'q' 키로 종료")

    global paused

    # 키 상태를 추적하는 딕셔너리
    key_states = {"forward": False, "backward": False, "left": False, "right": False}

    while True:
        key = stdscr.getch()

        # 키 입력 상태 업데이트
        if key == curses.KEY_UP:  # 전진
            key_states["forward"] = True
        elif key == curses.KEY_DOWN:  # 후진
            key_states["backward"] = True
        elif key == curses.KEY_LEFT:  # 좌회전
            key_states["left"] = True
        elif key == curses.KEY_RIGHT:  # 우회전
            key_states["right"] = True
        elif key == -1:  # 키를 떼었을 때 상태를 False로 설정
            key_states = {k: False for k in key_states}

        elif key == ord('q'):  # 종료
            print("종료합니다.")
            break

        # 키 상태에 따라 동작 수행
        if key_states["forward"]:
            set_dc_motor(dc_motor_speed, "forward")
            stdscr.addstr(2, 0, "DC Motor: Forward at 50% speed   ")
        elif key_states["backward"]:
            set_dc_motor(dc_motor_speed, "backward")
            stdscr.addstr(2, 0, "DC Motor: Backward at 50% speed  ")
        else:
            set_dc_motor(0, "stop")  # 전진/후진 키가 없으면 정지
            stdscr.addstr(2, 0, "DC Motor: Stopped               ")

        if key_states["left"]:
            servo_angle = max(0, servo_angle - 10)
            set_servo_angle(servo_angle)
            stdscr.addstr(3, 0, f"Servo angle set to {servo_angle} degrees   ")

        if key_states["right"]:
            servo_angle = min(180, servo_angle + 10)
            set_servo_angle(servo_angle)
            stdscr.addstr(3, 0, f"Servo angle set to {servo_angle} degrees   ")



try:
    # 녹화 스레드 시작
    video_thread = Thread(target=record_video)
    video_thread.start()

    # curses 기반 메인 루프 실행
    curses.wrapper(main)

except KeyboardInterrupt:
    print("사용자에 의해 중단됨")

finally:
    # 모든 리소스 해제
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
