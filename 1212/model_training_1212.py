import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# 1. PilotNet 모델 정의
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()

        # CNN Layer
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layer (fc1 입력 크기 조정)
        self.fc1 = nn.Linear(64 * 8 * 8, 100)  # 이미지 크기 (64 * 8 * 8)로 수정
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)  # 조향 값 출력 (1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 최종 조향 값 예측

        return x

# 2. DrivingDataset 클래스 정의 (CSV 파일에서 데이터 로드)
class DrivingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # CSV 파일 읽기
        self.transform = transform
        self.base_dir = "C:\\Users\\USER\\Desktop\\DGK\\mygithub\\Capstone_Design//1212driving"  # 폴더 이름 수정

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # 첫 번째 컬럼: 이미지 경로
        steering_angle = self.data.iloc[idx, 1]  # 두 번째 컬럼: 조향 값

        # 이미지 경로가 상대 경로일 경우 절대 경로로 변환
        img_name = os.path.join(self.base_dir, img_name)

        # 경로 출력으로 확인
        print(f"이미지 경로: {img_name}")

        try:
            # 이미지 불러오기
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {img_name}")
            raise

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(steering_angle, dtype=torch.float)

# 3. 데이터 전처리 및 증강 추가
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기 조정
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 밝기, 대비, 채도, 색조 변형
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전 (50% 확률)
    transforms.RandomRotation(degrees=10),  # 랜덤 회전 (±10도)
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 4. DataLoader 설정 (훈련 데이터셋 준비)
dataset = DrivingDataset(csv_file='1212dataset_changed.csv', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. 모델, 손실 함수, 옵티마이저 정의
model = PilotNet()
criterion = nn.MSELoss()  # 회귀 문제이므로 MSE 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 모델 훈련
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 학습 모드로 설정
    for images, steering_angles in dataloader:
        optimizer.zero_grad()  # 기울기 초기화
        
        # 모델 예측
        outputs = model(images)
        
        # 손실 계산
        loss = criterion(outputs.squeeze(), steering_angles)  # 예측 값과 실제 값을 비교
        
        # 역전파
        loss.backward()
        
        # 옵티마이저 업데이트
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 7. 모델 저장
torch.save(model.state_dict(), 'pilotnet_model1.pth')
