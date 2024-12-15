import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split  # 데이터 분할

# 1. 각도 값 매핑 함수
def map_angle_to_class(angle):
    """
    50~110도 각도를 5단계로 변환:
    50 -> 0, 65 -> 1, 80 -> 2, 95 -> 3, 110 -> 4
    """
    if angle <= 57.5:
        return 0  # 50
    elif angle <= 72.5:
        return 1  # 65
    elif angle <= 87.5:
        return 2  # 80
    elif angle <= 102.5:
        return 3  # 95
    else:
        return 4  # 110

# 2. DrivingDataset 클래스 정의
class DrivingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # CSV 파일 읽기
        self.transform = transform
        self.base_dir = "C:\\Users\\USER\\Desktop\\241215_resizing_curve"  # 폴더 이름 수정

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # 첫 번째 컬럼: 이미지 경로
        steering_angle = self.data.iloc[idx, 1]  # 두 번째 컬럼: 조향 값

        # 이미지 경로가 상대 경로일 경우 절대 경로로 변환
        img_name = os.path.join(self.base_dir, img_name)

        # 이미지 로드
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {img_name}")
            raise

        # 조향 각도를 클래스(0~4)로 변환
        steering_class = map_angle_to_class(steering_angle)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(steering_class, dtype=torch.long)

# 3. 데이터 전처리 (이미지 리사이즈 및 텐서 변환)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기 조정
    transforms.ToTensor(),        # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 4. DataLoader 설정 및 데이터 분할
csv_file = 'C:\\Users\\USER\\Desktop\\DGK\\mygithub\\Capstone_Design\\1215\\oversampled_data_12152.csv'  # CSV 파일 경로 수정
full_dataset = DrivingDataset(csv_file=csv_file, transform=transform)

# 데이터를 인덱스 리스트로 분할
train_indices, val_indices = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,  # 20%를 밸리데이션 세트로 사용
    random_state=42  # 재현성을 위한 시드 값
)

# Subset을 사용하여 나눔
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. Classification 모델 정의
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 5)  # 5개의 클래스 출력 (0~4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 출력층은 softmax 적용 전 logits
        return x

# 6. 모델, 손실 함수, 옵티마이저 정의
model = ClassificationModel()
criterion = nn.CrossEntropyLoss()  # 분류 문제이므로 CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 모델 학습 및 검증
num_epochs = 75
best_val_loss = float('inf')  # 초기값을 무한대로 설정
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    # 학습
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        # 모델 예측
        outputs = model(images)
        
        # 손실 계산
        loss = criterion(outputs, labels)
        
        # 역전파 및 최적화
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 검증
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    # 검증 손실이 낮을 때 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 최종 모델 저장
final_model_path = 'final_model_curve.pth'
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")
