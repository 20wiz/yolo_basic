from ultralytics import YOLO
import cv2
import numpy as np
import time

def webcam_ball_detection():
    """
    Real-time soccer ball detection using webcam
    """
    device = 'cpu'
    # Load YOLO model 
    # model_name = "yolov8n.pt"  # nano
    model_name = "yolov8m.pt"  # medium
 
    model = YOLO(model_name)
    model.to(device)
    cap = cv2.VideoCapture(0)
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        results = model(frame)
        
        # Process results
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
        
        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        # Display model name and FPS
        cv2.putText(frame, f'Model: {model_name}', (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display result frame
        cv2.imshow('Soccer Ball Detection', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Real-time detection from webcam
    webcam_ball_detection()