import Jetson.GPIO as GPIO
import time
import curses

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

# 모터와 상태 변수
current_direction = "stop"
paused = False

def set_servo_angle(angle):
    duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 180) * 10))
    servo.ChangeDutyCycle(duty_cycle)

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

def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    servo_angle = 95
    dc_motor_speed = 50
    set_dc_motor(0, "stop")
    stdscr.addstr(0, 0, "위/아래 키로 전진/후진 전환, 좌우 키로 조향, 스페이스바로 일시정지, 'q' 키로 종료")

    global paused

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP:
            if paused:
                paused = False
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개")
            if current_direction != "forward" and not paused:
                set_dc_motor(dc_motor_speed, "forward")
            stdscr.addstr(2, 0, "DC Motor: Forward at 50% speed")

        elif key == curses.KEY_DOWN:
            if paused:
                paused = False
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개")
            if current_direction != "backward" and not paused:
                set_dc_motor(dc_motor_speed, "backward")
            stdscr.addstr(2, 0, "DC Motor: Backward at 50% speed")

        elif key == curses.KEY_LEFT:
            servo_angle = max(0, servo_angle - 10)
            set_servo_angle(servo_angle)
            stdscr.addstr(3, 0, f"Servo angle set to {servo_angle} degrees")

        elif key == curses.KEY_RIGHT:
            servo_angle = min(180, servo_angle + 10)
            set_servo_angle(servo_angle)
            stdscr.addstr(3, 0, f"Servo angle set to {servo_angle} degrees")

        elif key == ord(' '):
            if not paused:
                set_dc_motor(0, "stop")
                paused = True
                stdscr.addstr(4, 0, "일시정지 상태")
            else:
                paused = False
                stdscr.addstr(4, 0, "일시정지 해제 후 동작 재개")

        elif key == ord('q'):
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
