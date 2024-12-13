import cv2
import Jetson.GPIO as GPIO
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import PilotNet  # 학습된 모델 import (저장된 모델 파일을 사용)

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
servo_motor_pin = 33       # 서보 모터 핀
dc_motor_pwm_pin = 32      # DC 모터 PWM 핀
dc_motor_dir_pin1 = 29     # DC 모터 방향 제어 핀 1
dc_motor_dir_pin2 = 31     # DC 모터 방향 제어 핀 2

# 서보 모터 PWM 설정 (50Hz)
GPIO.setup(servo_motor_pin, GPIO.OUT)
servo = GPIO.PWM(servo_motor_pin, 50)
servo.start(7.5)  # 서보 초기 위치 (7.5는 중립 위치, 필요에 따라 변경)

# DC 모터 설정
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 2000)  # 2kHz 주파수로 설정
dc_motor_pwm.start(0)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

# 상태 변수
current_direction = "stop"

# 모델 로드 함수
def load_model(model_path='pilotnet_model.pth'):
    model = PilotNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 모델 입력 크기 맞추기
    transforms.ToTensor(),        # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 모델 출력(-30 ~ 30)을 서보 모터 입력(0 ~ 180)으로 변환하는 함수
def convert_steering_angle(model_angle):
    """
    모델의 출력값 (-30 ~ 30)을 서보 모터 각도 (0 ~ 180)로 변환.
    """
    return 90 + (model_angle * 3)

# 서보 모터 각도 설정
def set_servo_angle(angle):
    # 서보 모터 각도 (2.5~12.5) 범위로 변환하여 Duty Cycle 설정
    duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 180) * 10))
    servo.ChangeDutyCycle(duty_cycle)

# DC 모터 속도 및 방향 설정
def set_dc_motor(speed, direction):
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
        speed = 0
        current_direction = "stop"
    
    dc_motor_pwm.ChangeDutyCycle(speed)   # 설정 속도로 전환

# 차량 제어 함수 (모델 예측 기반)
def control_vehicle_with_model(model):
    ret, frame = cap.read()
    if ret:
        # 1. 이미지를 PIL 이미지로 변환
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 2. 이미지 전처리 및 텐서 변환
        image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

        # 3. 모델 예측
        with torch.no_grad():
            steering_angle = model(image_tensor)

        # 4. 모델 출력(-30 ~ 30)을 서보 모터 입력(0 ~ 180)으로 변환
        steering_angle_value = steering_angle.item()
        servo_angle = convert_steering_angle(steering_angle_value)

        # 5. 예측된 각도 출력 및 차량 제어
        print(f"예측된 조향 각도 (모델 출력): {steering_angle_value}°, 서보 모터 입력: {servo_angle}°")
        set_servo_angle(servo_angle)  # 변환된 각도로 서보모터 제어
        set_dc_motor(50, "forward")  # 50% 속도로 전진

# 메인 실행 코드
if __name__ == "__main__":
    try:
        # 모델 로드
        model = load_model('pilotnet_model.pth')

        while True:
            control_vehicle_with_model(model)  # 모델로 차량 제어
            
            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료합니다.")
                break

            time.sleep(0.1)  # 주기적으로 제어 (0.1초마다 반복)

    except KeyboardInterrupt:
        print("사용자에 의해 중단됨")

    finally:
        cap.release()
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("GPIO 핀 정리 완료")
