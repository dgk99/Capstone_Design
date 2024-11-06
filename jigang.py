import Jetson.GPIO as GPIO
import time
import subprocess
import keyboard  # Import the keyboard module for key detection

# Set the sudo password as a variable for easy updating
sudo_password = "your_password_here"

# Function to run shell commands with the sudo password
def run_command(command):
    # Form the full command with password input
    full_command = f"echo {sudo_password} | sudo -S {command}"
    # Execute the command in the shell
    subprocess.run(full_command, shell=True, check=True)

# Check if busybox is installed; if not, install it
try:
    subprocess.run("busybox --help", shell=True, check=True)
    print("busybox is already installed.")
except subprocess.CalledProcessError:
    print("busybox not found. Installing...")
    run_command("apt update && apt install -y busybox")

# Define devmem commands
commands = [
    "busybox devmem 0x700031fc 32 0x45",
    "busybox devmem 0x6000d504 32 0x2",
    "busybox devmem 0x70003248 32 0x46",
    "busybox devmem 0x6000d100 32 0x00"
]

# Execute each devmem command
for command in commands:
    run_command(command)

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

# Set initial values
servo_angle = 90  # Start the servo at a neutral position
dc_motor_speed = 0  # Start with the motor stopped

# Function to set servo angle
def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)  # Turn off signal to avoid jitter

# Function to set DC motor speed and direction
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# Keyboard control loop
try:
    print("Control DC motor and servo with keyboard:")
    print("W = Forward, S = Backward, A = Rotate Left, D = Rotate Right, Q = Quit")
    
    while True:
        if keyboard.is_pressed('w'):
            set_dc_motor(50, "forward")  # Set motor speed to 50% forward
        elif keyboard.is_pressed('s'):
            set_dc_motor(50, "backward")  # Set motor speed to 50% backward
        elif keyboard.is_pressed('a'):
            servo_angle = max(0, servo_angle - 10)  # Decrease angle
            set_servo_angle(servo_angle)
        elif keyboard.is_pressed('d'):
            servo_angle = min(180, servo_angle + 10)  # Increase angle
            set_servo_angle(servo_angle)
        elif keyboard.is_pressed('q'):
            break  # Exit the control loop

        time.sleep(0.1)  # Short delay to allow time for keypress detection

finally:
    # Stop all PWM and clean up GPIO
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()