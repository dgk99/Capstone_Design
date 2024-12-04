import Jetson.GPIO as GPIO
import time
import curses

# GPIO 설정
GPIO.setmode(GPIO.BOARD)  # GPIO 모드를 BOARD로 설정 (핀 번호 사용)
servo_motor_pin = 33      # PWM을 지원하는 핀 번호로 설정
dc_motor_pwm_pin = 32     # PWM 핀
dc_motor_dir_pin1 = 29    # 방향 제어 핀 1
dc_motor_dir_pin2 = 31    # 방향 제어 핀 2

# 서보 모터 PWM 설정 (50Hz)
GPIO.setup(servo_motor_pin, GPIO.OUT)
servo = GPIO.PWM(servo_motor_pin, 50)
servo.start(7.5)  # 서보 초기 위치 (중앙)

# DC 모터 설정
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz 주파수
dc_motor_pwm.start(0)  # PWM 초기값 0%

# 전진 및 후진 상태 변수
current_direction = "stop"  # "forward", "backward", "stop"

def set_servo_angle(angle):
    # 서보 모터 각도 설정
    duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 180) * 10))  # 2.5~12.5 범위 제한
    try:
        servo.ChangeDutyCycle(duty_cycle)
    except OSError as e:
        print(f"Error during ChangeDutyCycle: {e}")

def set_dc_motor(speed, direction):
    # DC 모터 속도 및 방향 설정
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
        speed = 0  # 멈춤 상태로 변경
        current_direction = "stop"
    dc_motor_pwm.ChangeDutyCycle(speed)

def main(stdscr):
    # curses 설정
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    
    # 초기 값 설정
    servo_angle = 90  # 서보 초기 각도
    dc_motor_speed = 50  # DC 모터 기본 속도
    set_dc_motor(0, "stop")  # 초기에는 멈춤 상태
    stdscr.addstr(0, 0, "위/아래 키로 전진/후진 전환, 좌우 키로 조향, 'q' 키로 종료")

    while True:
        key = stdscr.getch()
        
        if key == curses.KEY_UP:  # 위쪽 화살표 키 - 전진
            if current_direction != "forward":  # 현재 후진 또는 멈춤 상태에서만 전진으로 전환
                set_dc_motor(dc_motor_speed, "forward")
            stdscr.addstr(2, 0, "DC Motor: Forward at 50% speed       ")
        
        elif key == curses.KEY_DOWN:  # 아래쪽 화살표 키 - 후진
            if current_direction != "backward":  # 현재 전진 또는 멈춤 상태에서만 후진으로 전환
                set_dc_motor(dc_motor_speed, "backward")
            stdscr.addstr(2, 0, "DC Motor: Backward at 50% speed      ")
        
        elif key == curses.KEY_LEFT:  # 왼쪽 화살표 키 - 좌회전
            servo_angle = max(0, servo_angle - 10)  # 최소 각도 0도로 제한
            set_servo_angle(servo_angle)
            stdscr.addstr(3, 0, f"Servo angle set to {servo_angle} degrees   ")
        
        elif key == curses.KEY_RIGHT:  # 오른쪽 화살표 키 - 우회전
            servo_angle = min(180, servo_angle + 10)  # 최대 각도 180도로 제한
            set_servo_angle(servo_angle)
            stdscr.addstr(3, 0, f"Servo angle set to {servo_angle} degrees   ")
        
        elif key == ord('q'):  # 'q' 키로 종료
            print("종료합니다.")
            break

try:
    curses.wrapper(main)

except KeyboardInterrupt:
    print("사용자에 의해 중단됨")

finally:
    # PWM과 GPIO 설정 정리
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    print("GPIO 핀 정리 완료")