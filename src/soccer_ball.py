from ultralytics import YOLO
import cv2
import numpy as np

import torch
print('cuda available ? ', torch.cuda.is_available()) 

def detect_soccer_ball(image_path):
    # device = 'cpu'   
    device = 'cuda:0'  # Use 'cuda:0' for GPU

    # Load YOLO model and set to device
    model = YOLO('yolov8n.pt')
    model.to(device)

    # Detect objects in the image
    results = model(image_path)
    
    # Load the original image
    img = cv2.imread(image_path)
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Check class name
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Process only if it's a ball ('sports ball' or 'ball')
            if class_name in ['sports ball', 'ball']:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Confidence score
                confidence = float(box.conf[0])
                
                # Draw bounding box (red)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label
                label = f'Soccer Ball: {confidence:.2f}'
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img


if __name__ == "__main__":

    # Detect from image file
    image_path = '.\\test1.webp'  # Modify to your image path
    result = detect_soccer_ball(image_path)
    
    # Resize the image to half its original size
    result_resized = cv2.resize(result, (result.shape[1] // 2, result.shape[0] // 2))
    
    cv2.imshow('Soccer Ball Detection', result_resized)
    cv2.waitKey(0)
    # cv2.imwrite('result.jpg', result)

