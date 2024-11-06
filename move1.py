import Jetson.GPIO as GPIO
import time
import keyboard  # Keyboard 입력 감지를 위해 사용

# Set up GPIO pins for servo and DC motor control
servo_pin = 33  # PWM-capable pin for servo motor
dc_motor_pwm_pin = 32  # PWM-capable pin for DC motor speed
dc_motor_dir_pin1 = 29  # Direction control pin 1
dc_motor_dir_pin2 = 31  # Direction control pin 2

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# Configure PWM on servo and DC motor pins
servo = GPIO.PWM(servo_pin, 50)  # 50Hz for servo motor
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz for DC motor
servo.start(0)
dc_motor_pwm.start(0)

# Function to set servo angle
def set_servo_angle(angle):
    # Calculate duty cycle based on angle (0 to 180 degrees)
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Allow time for the servo to reach position
    servo.ChangeDutyCycle(0)  # Turn off signal to avoid jitter

# Function to set DC motor speed and direction
def set_dc_motor(speed, direction):
    # Set direction: 'forward' or 'backward'
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    
    # Control speed with PWM (0 to 100%)
    dc_motor_pwm.ChangeDutyCycle(speed)

# Example usage: Rotate servo and control DC motor with keyboard input
try:
    print("Press 'W' to move forward, 'S' to move backward, 'A' to turn left, 'D' to turn right, 'Q' to quit.")
    # Initial servo angle
    servo_angle = 90  # Straight forward

    # Main control loop
    while True:
        # Move forward
        if keyboard.is_pressed('w'):
            set_dc_motor(50, "forward")

        # Move backward
        elif keyboard.is_pressed('s'):
            set_dc_motor(50, "backward")

        # Turn left
        elif keyboard.is_pressed('a'):
            servo_angle = max(45, servo_angle - 15)  # Limit left turn to 45 degrees
            set_servo_angle(servo_angle)
            time.sleep(0.2)  # Add delay to avoid rapid changes

        # Turn right
        elif keyboard.is_pressed('d'):
            servo_angle = min(135, servo_angle + 15)  # Limit right turn to 135 degrees
            set_servo_angle(servo_angle)
            time.sleep(0.2)  # Add delay to avoid rapid changes

        # Stop DC motor if no W or S key is pressed
        else:
            set_dc_motor(0, "forward")  # Stop DC motor
            time.sleep(0.1)  # Short delay to check key press rate

        # Quit the loop if Q is pressed
        if keyboard.is_pressed('q'):
            print("Exiting")
            break

finally:
    # Stop all PWM and clean up GPIO
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    print("GPIO cleanup completed.")
