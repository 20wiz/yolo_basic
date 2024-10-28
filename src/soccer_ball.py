from ultralytics import YOLO
import cv2
import numpy as np

import torch
print('cuda available ? ', torch.cuda.is_available()) 

def detect_soccer_ball(image_path):
    """
    축구공을 감지하는 함수
    """
    # device = 'cpu'   
    device = 'cuda:0'  # GPU 사용 시 'cuda:0'

    # YOLO 모델 로드 및 CPU 설정
    model = YOLO('yolov8n.pt')
    model.to(device)

    # 이미지에서 객체 감지
    results = model(image_path)
    
    # 원본 이미지 로드
    img = cv2.imread(image_path)
    
    # 결과 처리
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 클래스 이름 확인
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # 공인 경우에만 처리 ('sports ball' 또는 'ball')
            if class_name in ['sports ball', 'ball']:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 신뢰도 점수
                confidence = float(box.conf[0])
                
                # 바운딩 박스 그리기 (빨간색)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 레이블 추가
                label = f'Soccer Ball: {confidence:.2f}'
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img


if __name__ == "__main__":

    # 이미지 파일에서 감지
    image_path = '.\\test1.webp'  # 본인의 이미지 경로로 수정
    result = detect_soccer_ball(image_path)
    cv2.imshow('Soccer Ball Detection', result)
    cv2.waitKey(0)
    # cv2.imwrite('result.jpg', result)
    
