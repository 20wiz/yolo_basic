from ultralytics import YOLO
import cv2
import numpy as np

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

    # 웹캠에서 실시간 감지
    webcam_ball_detection()