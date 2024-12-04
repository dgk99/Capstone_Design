import cv2
import numpy as np

# 1. 이미지 대비 조정 (히스토그램 평활화 사용)
def adjust_contrast(image):
    # 이미지를 YUV 색상 공간으로 변환
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # Y 채널(밝기)의 히스토그램 평활화
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # 이미지를 다시 BGR로 변환
    contrast_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return contrast_image

# 2. 이미지 전처리 (그레이스케일 및 블러)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

# 3. Canny 엣지 검출 (파라미터 조정)
def detect_edges(image):
    # Canny 엣지 검출 (파라미터를 조정하여 더 잘 감지)
    edges = cv2.Canny(image, 50, 150)  # Canny 파라미터 조정
    return edges

# 4. 관심 영역(ROI) 설정
def region_of_interest(image):
    height, width = image.shape
    # 관심 영역 설정 (사각형 영역)
    polygons = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.5)),
        (0, int(height * 0.5))
    ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# 5. 곡선 피팅 (직선 또는 곡선을 근사)
def fit_curve(lines):
    # 곡선을 근사하기 위해 검출된 차선을 저장할 배열
    all_x = []
    all_y = []
    
    # 검출된 차선(선분)의 좌표를 모두 수집
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                all_x += [x1, x2]
                all_y += [y1, y2]
    
    # 수집된 좌표로 다항식 피팅 (2차 곡선)
    if len(all_x) > 0:
        poly_coeffs = np.polyfit(all_y, all_x, 2)  # y 좌표에 따른 x 좌표의 2차 다항식 피팅
        return poly_coeffs
    else:
        return None

# 6. 피팅된 곡선을 이미지에 그리기
def draw_curve(image, poly_coeffs):
    if poly_coeffs is not None:
        plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        plot_x = np.polyval(poly_coeffs, plot_y)

        # 곡선을 이미지에 그리기
        for i in range(len(plot_y) - 1):
            cv2.line(image, (int(plot_x[i]), int(plot_y[i])), (int(plot_x[i+1]), int(plot_y[i+1])), (0, 255, 0), 5)

# 전체 파이프라인
def process_image(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 이미지 대비 조정
    contrast_image = adjust_contrast(image)
    
    # 이미지 전처리
    preprocessed_image = preprocess_image(contrast_image)
    # 엣지 검출
    edges = detect_edges(preprocessed_image)
    # 관심 영역 마스크 적용
    roi = region_of_interest(edges)
    # 허프 변환을 이용한 선 검출 (파라미터 조정)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=30, maxLineGap=100)
    # 곡선 피팅
    poly_coeffs = fit_curve(lines)
    # 곡선을 이미지에 그리기
    draw_curve(image, poly_coeffs)
    
    return image

# 지정한 경로의 이미지를 처리
image_path = r"C:\Users\USER\Desktop\capstone\circle track picture.png"  # 사용자 지정 경로에 따라 수정
result_image = process_image(image_path)

# 결과 이미지 출력
cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
