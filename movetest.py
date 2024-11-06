import Jetson.GPIO as GPIO
import time
import keyboard

# GPIO 핀 설정
pwm_pin = 32      # DC 모터 PWM 핀
dir_pin = 29      # DC 모터 방향 제어 핀
servo_pin = 33    # 서보 모터 핀

# GPIO 초기화
GPIO.setmode(GPIO.BOARD)
GPIO.setup(pwm_pin, GPIO.OUT)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM 설정
dc_pwm = GPIO.PWM(pwm_pin, 100)  # DC 모터용 PWM 주파수 100Hz
dc_pwm.start(0)  # 초기 duty cycle을 0으로 설정
servo_pwm = GPIO.PWM(servo_pin, 50)  # 서보 모터용 PWM 주파수 50Hz
servo_pwm.start(7.5)  # 서보 초기 위치 (정중앙)

# DC 모터 전진 함수
def move_forward(speed=50):
    GPIO.output(dir_pin, GPIO.HIGH)
    dc_pwm.ChangeDutyCycle(speed)

# DC 모터 정지 함수
def stop_dc_motor():
    dc_pwm.ChangeDutyCycle(0)  # DC 모터 정지

# 서보 좌회전 함수
def turn_left():
    servo_pwm.ChangeDutyCycle(5)  # 서보를 좌회전 각도로 이동

# 서보 우회전 함수
def turn_right():
    servo_pwm.ChangeDutyCycle(10)  # 서보를 우회전 각도로 이동

# 서보 모터를 중앙으로 복귀
def straighten_servo():
    servo_pwm.ChangeDutyCycle(7.5)  # 서보를 중앙 위치로 이동

# 초기 상태 설정
dc_motor_running = False  # DC 모터가 처음에는 정지 상태로 설정

# 키보드 입력에 따른 제어
try:
    print("Press 'W' to start moving forward, 'S' to stop, 'A' to turn left, 'D' to turn right, 'Q' to quit.")
    while True:
        if keyboard.is_pressed('w') and not dc_motor_running:
            print("Starting forward motion")
            move_forward(50)  # 전진 시작
            dc_motor_running = True  # 전진 상태로 설정

        elif keyboard.is_pressed('s') and dc_motor_running:
            print("Stopping motor")
            stop_dc_motor()  # 정지
            dc_motor_running = False  # 정지 상태로 설정

        elif keyboard.is_pressed('a'):
            print("Turning left")
            turn_left()  # 서보 좌회전

        elif keyboard.is_pressed('d'):
            print("Turning right")
            turn_right()  # 서보 우회전

        elif keyboard.is_pressed('q'):
            print("Exiting")
            break

        else:
            straighten_servo()  # 키 입력이 없으면 서보 모터는 중앙으로 복귀

        time.sleep(0.1)  # 입력 확인 속도

finally:
    # GPIO 및 PWM 정리
    stop_dc_motor()
    straighten_servo()
    dc_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
    print("GPIO cleanup completed.")
