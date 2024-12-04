import Jetson.GPIO as GPIO
import keyboard
import time
import threading

# 핀 설정
pwm_pin = 32            # DC 모터 PWM 제어 핀
dir_pin = 29            # DC 모터 방향 제어 핀
servo_pin = 33          # 서보 모터 핀 (좌우 조향)

# GPIO 초기화
GPIO.setmode(GPIO.BOARD)
GPIO.setup(pwm_pin, GPIO.OUT)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM 설정 (DC 모터 및 서보 모터)
motor_pwm = GPIO.PWM(pwm_pin, 100)  # DC 모터 PWM 주파수 설정
motor_pwm.start(0)                  # 초기 속도 0으로 설정
servo_pwm = GPIO.PWM(servo_pin, 50) # 서보 모터 50Hz PWM 주파수 설정
servo_pwm.start(7.5)                # 서보 모터 중립 위치

def move_forward():
    GPIO.output(dir_pin, GPIO.HIGH)  # dir 핀 HIGH: 전진 방향
    motor_pwm.ChangeDutyCycle(70)    # 속도 설정
    print("w")

def move_backward():
    GPIO.output(dir_pin, GPIO.LOW)   # dir 핀 LOW: 후진 방향
    motor_pwm.ChangeDutyCycle(70)    # 속도 설정
    print("s")

def stop():
    motor_pwm.ChangeDutyCycle(0)     # 속도 0으로 설정하여 정지
    print("정지")

def turn_left():
    servo_pwm.ChangeDutyCycle(5)     # 서보 모터 좌회전 위치
    print("a")

def turn_right():
    servo_pwm.ChangeDutyCycle(10)    # 서보 모터 우회전 위치
    print("d")

def center_wheels():
    servo_pwm.ChangeDutyCycle(7.5)   # 서보 모터 중립 위치
    print("중립")

def status_message():
    """2초마다 코드가 실행 중임을 출력"""
    while True:
        print("코드가 실행중입니다")
        time.sleep(2)

# 스레드를 사용해 상태 메시지를 주기적으로 출력
status_thread = threading.Thread(target=status_message)
status_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
status_thread.start()

try:
    print("Press W/S to move forward/backward, A/D to turn left/right, and 'q' to quit.")
    while True:
        if keyboard.is_pressed('w'):
            move_forward()
        elif keyboard.is_pressed('s'):
            move_backward()
        elif keyboard.is_pressed('a'):
            turn_left()
        elif keyboard.is_pressed('d'):
            turn_right()
        else:
            stop()
            center_wheels()
        
        # 'q'키로 프로그램 종료
        if keyboard.is_pressed('q'):
            print("Exiting...")
            break

        time.sleep(0.1)  # 루프 주기 조절

finally:
    # GPIO 정리
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
