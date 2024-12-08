import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt

# SteeringDataset 클래스 정의
class SteeringDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['image_name']
        image_path = f"C:/Users/USER/Desktop/DGK/mygithub/Capstone_Design/{image_name}"
        print(f"이미지 경로: {image_path}")  # 경로 확인
        frame = cv2.imread(image_path)  # 이미지 읽기
        if frame is None:
            raise ValueError(f"이미지 {image_path}을(를) 읽을 수 없습니다.")
        frame = cv2.resize(frame, (200, 66))  # 크기 조정
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        steering_angle = float(row['steering_angle'])
        return torch.tensor(frame, dtype=torch.float32), torch.tensor(steering_angle, dtype=torch.float32)


# 모델 정의
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),  # 크기를 맞추기 위해 64*1*18로 수정
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 조향각 예측
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# 장치 설정 (GPU 사용 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 데이터 로드 (상대 경로 수정)
dataset = SteeringDataset("C:/Users/USER/Desktop/DGK/mygithub/Capstone_Design/labeled_dataset.csv")  # 절대 경로로 수정
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 초기화 및 GPU 이동
model = PilotNet().to(device)  # 모델을 GPU로 이동
criterion = nn.SmoothL1Loss()  # 손실 함수: Smooth L1 Loss
optimizer = Adam(model.parameters(), lr=0.001)  # 초기 학습률

# 학습률 스케줄러 설정 (학습이 진행됨에 따라 학습률을 감소)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# 학습률 Warm-up 적용 (초기 몇 에폭 동안 학습률 증가)
class WarmUpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up_steps, last_epoch=-1):
        self.warm_up_steps = warm_up_steps
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_up_steps:
            lr = [base_lr * (self.last_epoch + 1) / self.warm_up_steps for base_lr in self.base_lrs]
        else:
            lr = self.base_lrs
        return lr

# Warm-up 스케줄러 추가
warm_up_scheduler = WarmUpScheduler(optimizer, warm_up_steps=5)  # 5 에폭 동안 warm-up

# 손실 값을 저장할 리스트와 최저 손실 값 초기화
losses = []
best_loss = float('inf')  # 초기값은 무한대로 설정

# 학습 루프
epochs = 200
for epoch in range(epochs):
    total_loss = 0
    model.train()  # 학습 모드로 설정
    for images, angles in dataloader:
        images, angles = images.to(device), angles.to(device)  # 데이터도 GPU로 이동
        optimizer.zero_grad()  # 기존 그래디언트 초기화
        outputs = model(images)  # 모델 예측
        loss = criterion(outputs.squeeze(), angles)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
        total_loss += loss.item()

    # 평균 손실 계산
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)  # 손실 값을 저장

    # 최저 손실 값 갱신 시 모델 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "./best_pilotnet_model.pth")
        print(f"Epoch {epoch+1}: 새로운 Best 모델 저장 (Loss: {best_loss:.4f})")

    # 학습률 업데이트
    warm_up_scheduler.step()  # 학습률 warm-up
    scheduler.step()  # 학습률 감소

    # 매 epoch마다 학습 상태 출력
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# 최종 모델 저장 (학습 종료 시점의 모델)
torch.save(model.state_dict(), "./final_pilotnet_model.pth")
print("모델 학습 완료 및 최종 모델 저장 완료")

# 손실 값 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), losses, label='Training Loss', color='blue', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("training_loss_graph.png")  # 그래프를 이미지로 저장
plt.show()
