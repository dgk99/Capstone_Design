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

# 상태 변수
current_direction = "stop"
paused = True
is_recording = Event()
running = True

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

# curses 기반 차량 제어
def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    servo_angle = 95
    dc_motor_speed = 50
    set_dc_motor(0, "stop")
    stdscr.addstr(0, 0, "위/아래 키로 전진/후진 전환, 좌우 키로 조향, 'q' 키로 부스터, 스페이스바로 일시정지")

    global paused

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP:  # 전진
            if paused:
                paused = False
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개")
            set_dc_motor(dc_motor_speed, "forward")
            stdscr.addstr(2, 0, "DC Motor: Forward at 50% speed")

        elif key == curses.KEY_DOWN:  # 후진
            if paused:
                paused = False
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개")
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
                paused = True
                stdscr.addstr(4, 0, "일시정지 상태")
            else:
                paused = False
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개")

        elif key == ord('q'):  # 부스터
            stdscr.addstr(5, 0, "Booster Activated!")
            set_dc_motor(dc_motor_speed, "forward", booster=True)

        elif key == ord('e'):  # 종료
            print("종료합니다.")
            break

try:
    curses.wrapper(main)

except KeyboardInterrupt:
    print("사용자에 의해 중단됨")

finally:
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    print("GPIO 핀 정리 완료")
