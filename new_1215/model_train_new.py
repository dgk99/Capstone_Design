import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split  # 데이터 분할

# 1. PilotNet 모델 정의
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 5)  # 5개의 각도 클래스 출력

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # logits 출력
        return x

# 2. 각도 값 매핑 함수
def map_angle_to_class(angle):
    """
    각도 값을 5개 클래스로 매핑:
    65 -> 0, 80 -> 1, 95 -> 2, 110 -> 3, 125 -> 4
    """
    if angle <= 72:    # 65도 기준
        return 0
    elif angle <= 87:  # 80도 기준
        return 1
    elif angle <= 102: # 95도 기준
        return 2
    elif angle <= 117: # 110도 기준
        return 3
    else:              # 125도 기준
        return 4

# 3. DrivingDataset 클래스 정의
class DrivingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.base_dir = "C:\\Users\\USER\\Desktop\\resizing_oversampled_new_1215_4"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        steering_angle = self.data.iloc[idx, 1]
        img_name = os.path.join(self.base_dir, img_name)

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 조향 각도를 클래스(0~4)로 변환
        steering_class = map_angle_to_class(steering_angle)
        return image, torch.tensor(steering_class, dtype=torch.long)

# 4. 데이터 전처리 및 분할
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

csv_file = 'C:\\Users\\USER\\Desktop\\resizing_oversampled_new_1215_4\\oversampled_data_new_1215_4.csv'
full_dataset = DrivingDataset(csv_file=csv_file, transform=transform)

# 데이터 분할
train_indices, val_indices = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    random_state=42
)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델, 손실 함수, 옵티마이저 정의
model = PilotNet().to(device)  # CUDA로 모델 이동
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 모델 학습 및 검증
num_epochs = 50
best_val_loss = float('inf')
best_model_path = 'best_pilotnet_model_1215_4_50epoch.pth'

for epoch in range(num_epochs):
    # 학습
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 데이터를 CUDA로 이동
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
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
            images, labels = images.to(device), labels.to(device)  # 데이터를 CUDA로 이동
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

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # 가장 낮은 검증 손실에서 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch {epoch+1} with validation loss {val_loss:.4f}")

print("Model training complete. Best model saved.")
