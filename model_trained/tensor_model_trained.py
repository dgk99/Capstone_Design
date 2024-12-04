import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# --- 1. 데이터 로드 및 전처리 ---
def preprocess_image(image_path, target_size=(128, 128)):
    """이미지를 전처리: 크기 조정 및 정규화"""
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)  # 크기 조정
    normalized_image = resized_image / 255.0       # 정규화
    return normalized_image

def load_data(csv_path, image_folder):
    """CSV 파일과 이미지를 로드하여 전처리"""
    # CSV 로드
    data = pd.read_csv(csv_path)

    # 이미지와 조향 데이터 준비
    images = []
    angles = []
    for _, row in data.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        if os.path.exists(image_path):  # 이미지 파일 확인
            images.append(preprocess_image(image_path))
            angles.append(row['steering_angle'])
        else:
            print(f"이미지 파일 누락: {row['image_name']}")

    # Numpy 배열로 변환
    images = np.array(images)
    angles = np.array(angles)
    return images, angles

# --- 2. 데이터 분리 ---
def split_data(images, angles):
    """데이터를 학습, 검증, 테스트로 분리"""
    X_train, X_test, y_train, y_test = train_test_split(images, angles, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- 3. 모델 설계 ---
def build_model(input_shape):
    """CNN 모델 설계"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)  # 출력: 조향 각도
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 4. 모델 학습 및 평가 ---
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """모델 학습 및 평가"""
    # 학습
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    
    # 평가
    loss = model.evaluate(X_test, y_test)
    print(f"테스트 손실: {loss}")
    return history

# --- 5. 실행 ---
if __name__ == "__main__":
    # 파일 경로 설정
    CSV_PATH = "labeled_dataset.csv"
    IMAGE_FOLDER = "C:\\Users\\USER\\Desktop\\take_photo\\traning_data_set"

    # 데이터 로드 및 전처리
    print("데이터 로드 중...")
    images, angles = load_data(CSV_PATH, IMAGE_FOLDER)
    print(f"로드된 데이터: 이미지 {images.shape}, 조향 데이터 {angles.shape}")

    # 데이터 분리
    print("데이터 분리 중...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, angles)
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # 모델 설계
    print("모델 설계 중...")
    model = build_model(input_shape=(128, 128, 3))

    # 모델 학습 및 평가
    print("모델 학습 및 평가 중...")
    history = train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    print("작업 완료!")
    
    model.save("trained_model.h5")
    print("모델이 저장되었습니다.")
