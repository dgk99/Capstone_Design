import os
import shutil  # 파일 이동용
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt

def display_instructions():
    print("\n사진 검열 모드:")
    print("[d: 삭제] [s: 복구] [f: 다음 (괜찮음)] [q: 종료]")

def show_image_with_matplotlib(img_path, current_index, total_images):
    """Matplotlib을 사용해 이미지를 표시"""
    img = Image.open(img_path)  # 이미지를 RGB로 로드
    plt.imshow(img)
    plt.axis("off")  # 축 제거
    plt.title(f"Review Photo ({current_index}/{total_images})")  # 진행 상태 표시
    plt.show(block=False)  # 비블록 모드로 표시
    plt.pause(0.1)  # 잠시 멈춤

def review_photos(photo_dir, deleted_dir, approved_dir):
    os.makedirs(deleted_dir, exist_ok=True)
    os.makedirs(approved_dir, exist_ok=True)

    deleted_photos = deque()  # 삭제된 사진 저장용 (복구 가능)
    approved_photos = []  # 괜찮다고 넘긴 사진 저장
    image_files = sorted([f for f in os.listdir(photo_dir) if f.endswith('.jpg')])
    total_images = len(image_files)  # 총 사진 개수
    current_index = 0

    while current_index < len(image_files):
        img_path = os.path.join(photo_dir, image_files[current_index])

        # 이미지 표시 (Matplotlib 사용)
        plt.close()  # 이전 이미지 닫기
        show_image_with_matplotlib(img_path, current_index + 1, total_images)  # 현재 사진 번호와 총 사진 개수
        print(f"\n현재 사진 이름: {image_files[current_index]}")
        display_instructions()

        key = input("명령을 입력하세요 (d/s/f/q): ").strip().lower()

        if key == 'd':  # 사진 삭제
            deleted_path = os.path.join(deleted_dir, image_files[current_index])
            shutil.move(img_path, deleted_path)  # 사진 이동
            print(f"사진 삭제 및 이동: {image_files[current_index]}")
            deleted_photos.append((deleted_path, current_index))
            image_files.pop(current_index)
            total_images -= 1  # 총 사진 개수 감소

        elif key == 's':  # 삭제한 사진 복구
            if deleted_photos:
                last_deleted, last_index = deleted_photos.pop()
                restored_name = os.path.basename(last_deleted)
                restored_path = os.path.join(photo_dir, restored_name)
                shutil.move(last_deleted, restored_path)  # 사진 복구
                image_files.insert(last_index, restored_name)
                total_images += 1  # 총 사진 개수 증가
                current_index = min(current_index, last_index)  # 인덱스 업데이트
                print(f"사진 복구: {restored_name}")
            else:
                print("복구할 사진이 없습니다.")

        elif key == 'f':  # 괜찮은 사진으로 이동
            approved_path = os.path.join(approved_dir, image_files[current_index])
            shutil.move(img_path, approved_path)  # 괜찮은 사진 이동
            print(f"괜찮은 사진으로 이동: {image_files[current_index]}")
            approved_photos.append(image_files[current_index])  # 이동된 사진 기록
            image_files.pop(current_index)
            total_images -= 1  # 총 사진 개수 감소

        elif key == 'q':  # 종료
            print("검열 종료")
            break

        else:
            print("유효하지 않은 입력입니다. 다시 시도하세요.")

    print("\n사진 검열이 완료되었습니다.")
    print(f"괜찮은 사진: {len(approved_photos)}장 이동 완료")
    plt.close()

if __name__ == "__main__":
    photo_dir = input("검열할 사진 폴더 경로를 입력하세요: ").strip()
    deleted_dir = input("삭제한 사진을 저장할 경로를 입력하세요: ").strip()
    approved_dir = input("괜찮은 사진을 저장할 경로를 입력하세요: ").strip()

    if not os.path.exists(photo_dir):
        print("사진 폴더 경로가 존재하지 않습니다.")
    else:
        review_photos(photo_dir, deleted_dir, approved_dir)
