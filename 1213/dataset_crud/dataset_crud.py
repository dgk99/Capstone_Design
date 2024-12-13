import os
import cv2
from collections import deque
import shutil  # 파일 이동용

def display_instructions():
    print("\n사진 검열 모드:")
    print("[d: 삭제] [s: 복구] [f: 다음] [q: 종료]")

# 메인 루프
def review_photos(photo_dir, deleted_dir):
    os.makedirs(deleted_dir, exist_ok=True)

    deleted_photos = deque()  # 삭제된 사진 저장용 (복구 가능)
    image_files = sorted([f for f in os.listdir(photo_dir) if f.endswith('.jpg')])
    current_index = 0

    while current_index < len(image_files):
        img_path = os.path.join(photo_dir, image_files[current_index])
        img = cv2.imread(img_path)

        # 이미지 표시
        cv2.imshow("Review Photo", img)
        print(f"\n현재 사진 이름: {image_files[current_index]}")
        display_instructions()
        key = cv2.waitKey(0)

        if key == ord('d'):  # 사진 삭제
            deleted_path = os.path.join(deleted_dir, image_files[current_index])
            shutil.move(img_path, deleted_path)  # 사진 이동
            print(f"사진 삭제 및 이동: {image_files[current_index]}")
            deleted_photos.append((deleted_path, current_index))
            image_files.pop(current_index)

        elif key == ord('s'):  # 삭제한 사진 복구
            if deleted_photos:
                last_deleted, last_index = deleted_photos.pop()
                restored_name = os.path.basename(last_deleted)
                restored_path = os.path.join(photo_dir, restored_name)
                shutil.move(last_deleted, restored_path)  # 사진 복구
                image_files.insert(last_index, restored_name)
                current_index = min(current_index, last_index)  # 인덱스 업데이트
                print(f"사진 복구: {restored_name}")
            else:
                print("복구할 사진이 없습니다.")

        elif key == ord('f'):  # 다음 사진으로 넘어가기
            print("다음 사진으로 넘어갑니다.")
            current_index += 1

        elif key == ord('q'):  # 종료
            print("검열 종료")
            break

        else:
            print("유효하지 않은 입력입니다. 다시 시도하세요.")

        cv2.destroyWindow("Review Photo")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    photo_dir = input("검열할 사진 폴더 경로를 입력하세요: ").strip()
    deleted_dir = input("삭제한 사진을 저장할 경로를 입력하세요: ").strip()

    if not os.path.exists(photo_dir):
        print("사진 폴더 경로가 존재하지 않습니다.")
    else:
        review_photos(photo_dir, deleted_dir)
