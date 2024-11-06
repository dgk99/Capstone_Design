import Jetson.GPIO as GPIO
import time
import keyboard  # 키보드 입력을 처리하는 라이브러리

# GPIO 핀 설정
servo_pin = 33  # 서보모터 제어 핀 (PWM)
dc_motor_pwm_pin = 32  # DC 모터 속도 제어 핀 (PWM)
dc_motor_dir_pin1 = 29  # DC 모터 방향 제어 핀 1
dc_motor_dir_pin2 = 31  # DC 모터 방향 제어 핀 2

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_pin, 50)  # 서보모터용 50Hz PWM (서보는 보통 50Hz)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DC 모터용 1kHz PWM
servo.start(0)
dc_motor_pwm.start(0)

# 서보모터 각도 설정 함수 (PWM 신호로 각도 제어)
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)  # 0도에서 180도까지의 PWM 값 계산
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # 서보모터가 위치로 이동할 시간을 줍니다.
    servo.ChangeDutyCycle(0)  # 움직임 후 신호를 꺼서 서보모터가 떨리지 않게 합니다.

# DC 모터 속도 및 방향 설정 함수
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)  # 0에서 100%까지 속도 조절

# 키보드 입력을 통해 제어하기
try:
    current_angle = 90  # 서보모터 기본 각도 90도
    while True:
        # 전진: 'w', 후진: 's', 왼쪽: 'a', 오른쪽: 'd'로 제어
        if keyboard.is_pressed('w'):  # 전진
            set_dc_motor(50, "forward")
        elif keyboard.is_pressed('s'):  # 후진
            set_dc_motor(50, "backward")
        else:
            set_dc_motor(0, "forward")  # 정지

        if keyboard.is_pressed('a'):  # 왼쪽으로 회전
            current_angle = max(0, current_angle - 10)
            set_servo_angle(current_angle)
        elif keyboard.is_pressed('d'):  # 오른쪽으로 회전
            current_angle = min(180, current_angle + 10)
            set_servo_angle(current_angle)

        time.sleep(0.1)  # 100ms 간격으로 입력을 체크
finally:
    # 종료 시 PWM 및 GPIO 정리
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
