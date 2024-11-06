import Jetson.GPIO as GPIO
import time
import curses

# GPIO 설정
GPIO.setmode(GPIO.BOARD)  # GPIO 모드를 BOARD로 설정 (핀 번호 사용)
servo_motor_pin = 33      # 서보 모터 PWM 핀 번호
dc_motor_pwm_pin = 32     # DC 모터 PWM 핀 번호
dc_motor_dir_pin = 31     # DC 모터 방향 핀 번호
GPIO.setup(servo_motor_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_motor_pin, 50)  # 서보 모터 PWM 50Hz
servo.start(7.5)  # 서보 초기 위치 (중앙)

dc_motor = GPIO.PWM(dc_motor_pwm_pin, 50)  # DC 모터 PWM 50Hz
dc_motor.start(0)  # DC 모터 초기 속도는 0

def set_servo_angle(angle):
    # 서보 모터 각도 조정
    duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 180) * 10))  # 2.5~12.5 범위 제한
    servo.ChangeDutyCycle(duty_cycle)

def move_forward(speed):
    # DC 모터 앞으로 이동
    GPIO.output(dc_motor_dir_pin, GPIO.HIGH)
    dc_motor.ChangeDutyCycle(speed)

def move_backward(speed):
    # DC 모터 뒤로 이동
    GPIO.output(dc_motor_dir_pin, GPIO.LOW)
    dc_motor.ChangeDutyCycle(speed)

def stop_motor():
    # DC 모터 정지
    dc_motor.ChangeDutyCycle(0)

def main(stdscr):
    # curses 설정
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    servo_angle = 90  # 서보 초기 각도
    motor_speed = 50  # DC 모터 속도
    
    stdscr.addstr(0, 0, "화살표 키: 좌우 조향, 앞뒤 이동 | 'q' 키: 종료")

    while True:
        key = stdscr.getch()
        
        if key == curses.KEY_LEFT:  # 왼쪽 화살표 키
            servo_angle = max(0, servo_angle - 10)  # 최소 각도 0도로 제한
            set_servo_angle(servo_angle)
            stdscr.addstr(2, 0, f"Servo angle set to {servo_angle} degrees    ")
        
        elif key == curses.KEY_RIGHT:  # 오른쪽 화살표 키
            servo_angle = min(180, servo_angle + 10)  # 최대 각도 180도로 제한
            set_servo_angle(servo_angle)
            stdscr.addstr(2, 0, f"Servo angle set to {servo_angle} degrees    ")
        
        elif key == curses.KEY_UP:  # 위쪽 화살표 키
            move_forward(motor_speed)
            stdscr.addstr(3, 0, "Moving forward at speed: {}%     ".format(motor_speed))
        
        elif key == curses.KEY_DOWN:  # 아래쪽 화살표 키
            move_backward(motor_speed)
            stdscr.addstr(3, 0, "Moving backward at speed: {}%    ".format(motor_speed))
        
        elif key == ord('q'):  # 'q' 키로 종료
            print("종료합니다.")
            break

        # 모터를 멈추기 위한 딜레이
        time.sleep(0.1)
        stop_motor()  # 키를 떼면 멈추도록 설정

try:
    curses.wrapper(main)

except KeyboardInterrupt:
    print("사용자에 의해 중단됨")

finally:
    # PWM과 GPIO 설정 정리
    servo.stop()
    dc_motor.stop()
    GPIO.cleanup()
    print("GPIO 핀 정리 완료")
