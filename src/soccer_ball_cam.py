from ultralytics import YOLO
import cv2
import numpy as np
import time

def webcam_ball_detection():
    """
    웹캠을 통한 실시간 축구공 감지
    """
    device = 'cpu'
    # YOLO 모델 로드 
    # model_name = "yolov8n.pt"  # nano
    model_name = "yolov8m.pt"  # medium
 
    model = YOLO(model_name) # 
    model.to(device)
    cap = cv2.VideoCapture(0)
    
    
    while True:
        start_time = time.time()
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
        
        # FPS 계산
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        # 모델 이름과 FPS 표시
        cv2.putText(frame, f'Model: {model_name}', (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 결과 화면 표시
        cv2.imshow('Soccer Ball Detection', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # 웹캠에서 실시간 감지
    webcam_ball_detection()