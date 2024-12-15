import cv2
import Jetson.GPIO as GPIO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
servo_motor_pin = 33
dc_motor_pwm_pin = 32
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

# 서보 모터 PWM 설정
GPIO.setup(servo_motor_pin, GPIO.OUT)
servo = GPIO.PWM(servo_motor_pin, 50)
servo.start(7.5)

# DC 모터 설정
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 2000)
dc_motor_pwm.start(0)

# ClassificationModel 정의
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)  # 학습 코드와 동일한 필터 크기
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 200)  # 학습 코드에서 Fully Connected 레이어 크기 맞춤
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 5)  # 클래스 개수는 5로 유지

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 서보 모터 각도 설정
def set_servo_angle(angle):
    servo_angle = 95 + angle
    duty_cycle = max(2.5, min(12.5, 2.5 + (servo_angle / 180) * 10))
    servo.ChangeDutyCycle(duty_cycle)

# DC 모터 속도 및 방향 설정
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    elif direction == "stop":
        speed = 0
    dc_motor_pwm.ChangeDutyCycle(speed)

# 모델 로드
def load_model(model_path='classification_model.pth'):
    model = ClassificationModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 차량 제어 함수
def control_vehicle_with_model(model):
    ret, frame = cap.read()
    if ret:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_class = torch.max(outputs, 1)
            angle_mapping = {0: -30, 1: -15, 2: 0, 3: 15, 4: 30}
            predicted_angle = angle_mapping[predicted_class.item()]
            print(f"Predicted class: {predicted_class.item()}, angle: {predicted_angle}°")
            set_servo_angle(predicted_angle)
            set_dc_motor(70, "forward")

# 메인 실행
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    try:
        model = load_model('classification_model.pth')
        while True:
            control_vehicle_with_model(model)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("User interruption.")
    finally:
        cap.release()
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        print("Clean up done.")
