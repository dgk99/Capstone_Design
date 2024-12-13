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

        # Fully Connected Layer
        self.fc1 = nn.Linear(64 * 8 * 5, 100)  # 이미지 크기 (66x200 기준)
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
        
        # 비어 있는 각도 값을 기본값 95로 대체
        self.data['servo_angle'] = self.data['servo_angle'].fillna(95)
        self.transform = transform
        self.base_dir = "C:\\Users\\USER\\Desktop\\DGK\\mygithub\\Capstone_Design\\241213_resizing"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # 첫 번째 컬럼: 이미지 경로
        steering_angle = self.data.iloc[idx, 1]  # 두 번째 컬럼: 조향 값

        # 이미지 경로가 상대 경로일 경우 절대 경로로 변환
        img_name = os.path.join(self.base_dir, img_name)

        try:
            # 이미지 불러오기
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"이미지 파일이 존재하지 않습니다. 건너뜁니다: {img_name}")
            return None  # None 반환하여 DataLoader에서 처리

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(steering_angle, dtype=torch.float)

# Custom collate function to handle None values
def collate_fn(batch):
    # None 값이 없는 데이터만 반환
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:  # 유효한 데이터가 없으면 빈 배치 반환
        return None
    return torch.utils.data.default_collate(batch)

# 3. 데이터 전처리 (이미지 리사이즈 및 텐서 변환)
transform = transforms.Compose([
    transforms.Resize((66, 200)),  # PilotNet 입력 크기
    transforms.ToTensor()         # 텐서로 변환
])

# 4. DataLoader 설정 (훈련 데이터셋 준비)
csv_file_path = "C:\\Users\\USER\\Desktop\\DGK\\mygithub\\Capstone_Design\\driving_data_20241213_192414.csv"
dataset = DrivingDataset(csv_file=csv_file_path, transform=transform)
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    collate_fn=collate_fn  # Custom collate function 사용
)

# 5. 모델, 손실 함수, 옵티마이저 정의
model = PilotNet()
criterion = nn.MSELoss()  # 회귀 문제이므로 MSE 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 모델 훈련
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 학습 모드로 설정
    total_loss = 0
    for batch in dataloader:
        if batch is None:  # 빈 배치 건너뛰기
            continue

        images, steering_angles = batch
        optimizer.zero_grad()  # 기울기 초기화

        # 모델 예측
        outputs = model(images)
        
        # 손실 계산
        loss = criterion(outputs.squeeze(), steering_angles)  # 예측 값과 실제 값을 비교
        total_loss += loss.item()
        
        # 역전파
        loss.backward()
        
        # 옵티마이저 업데이트
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

# 7. 모델 저장
torch.save(model.state_dict(), 'pilotnet_model2.pth')
print("모델 저장 완료: pilotnet_model2.pth")
