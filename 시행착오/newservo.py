import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# I2C 통신 설정
i2c = busio.I2C(board.SCL, board.SDA)

# PCA9685 객체 생성
pca = PCA9685(i2c)
pca.frequency = 50  # 서보 모터는 일반적으로 50Hz로 동작

# PCA9685의 0번 채널에 서보 연결
servo_motor = servo.Servo(pca.channels[0])

# 서보모터의 각도 제어
try:
    while True:
        angle = int(input("서보 각도 (0-180도): "))
        servo_motor.angle = angle
        time.sleep(1)
except KeyboardInterrupt:
    pass

# 종료 시 리소스 정리
finally:
    pca.deinit()
