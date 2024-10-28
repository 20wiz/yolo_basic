from ultralytics import YOLO
import cv2
import numpy as np

def detect_soccer_ball(image_path):
    """
    축구공을 감지하는 함수
    """
    device = 'cpu'
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
            
            # 축구공인 경우에만 처리 ('sports ball' 또는 'ball')
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

def webcam_ball_detection():
    """
    웹캠을 통한 실시간 축구공 감지
    """
    device = 'cpu'
    # YOLO 모델 로드 및 CPU 설정
    model = YOLO('yolov8n.pt')
    model.to(device)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 객체 감지 수행
        results = model(frame)
        
        # 결과 처리
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if class_name in ['sports ball', 'ball']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f'Soccer Ball: {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 결과 화면 표시
        cv2.imshow('Soccer Ball Detection', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 이미지 파일에서 감지
    image_path = '.\\1.webp'  # 본인의 이미지 경로로 수정
    result = detect_soccer_ball(image_path)
    # cv2.imwrite('result.jpg', result)
    
    # 웹캠에서 실시간 감지
    webcam_ball_detection()