from ultralytics import YOLO
import cv2
import time

save_not_detected_frames = False

def detect_soccer_ball_video(video_path, output_path=None):
    """
    Function to detect soccer balls in a video
    
    Args:
        video_path (str): Input video file path
        output_path (str, optional): Path to save the result video. If None, do not save
    """
    # Use CPU
    # device = 'cpu'   
    device = 'cuda:0'  # Use GPU with 'cuda:0'

    # Load YOLO model 
    # model_name = "yolov8n.pt"  # nano
    # model_name = "yolov8s.pt"  # small
    # model_name = "yolov8m.pt"  # medium
    # model_name = "yolov8s-1280.pt"  # small, 1280 high-res
    # model_name = "yolo11s.pt"  # 11 small 
    model_name = "yolo11m.pt"  # v11 medium 

    model = YOLO(model_name) # 
    model.to(device)
    
    # Create video capture object
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frames = int(fps * 4)  # Calculate frames until end time
    
    frame_count = 0
    ball_not_detected_count = 0

    # Set up result video saving
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Variables for FPS calculation
    prev_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    # Frame counter
    frame_counter = 0
    ball_detected_frames = 0  # Number of frames where soccer ball is detected
        
    try:
        while frame_count < end_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_counter += 1
            
            # Perform object detection
            # results = model(frame, device=device)
            # results = model(frame, device=device, conf=0.2, iou=0.3)
            # results = model(frame, device=device, conf=0.2, iou=0.3, imgsz=1280) # 
            # results = model(frame, device=device, conf=0.2, iou=0.3, imgsz=1920) # 
            results = model(frame, device=device, conf=0.2,  imgsz=1920) # 

            # FPS calculation
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                prev_time = current_time
            
            # Calculate progress
            progress = (frame_counter / total_frames) * 100
            
            # Process and visualize results
            ball_detected = False
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    if class_name in ['sports ball', 'ball']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f'Soccer Ball: {confidence:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        ball_detected = True                        
            
            if ball_detected:
                ball_detected_frames += 1
            else:
                ball_not_detected_count += 1
                # save frame as a image file
                if save_not_detected_frames:
                    cv2.imwrite(f'output/frame_{frame_count}.jpg', frame)

            # Display FPS and progress
            cv2.putText(frame, f'FPS: {fps_display}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Progress: {progress:.1f}%', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save result
            if output_path:
                out.write(frame)

            # Resize frame to half
            resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            
            # Display result
            cv2.imshow('Soccer Ball Detection', resized_frame)
    
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except Exception as e:
        print(f"Error occurred during processing: {e}")
        
    finally:
        # Release resources
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

    # Calculate precision, recall, and F1-score
    true_positives = ball_detected_frames
    false_negatives = ball_not_detected_count
    false_positives = 0  # Assuming no false positives for simplicity

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print the (ball park) metrics
    print(f"Number of frames with detected soccer balls: {ball_detected_frames}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

if __name__ == "__main__":
    # Set video file path
    video_path = "soccer_1.mp4"  # Change to your video file path
    output_path = "output_soccer.mp4"  # Path to save the result
    
    try:
        detect_soccer_ball_video(
            video_path=video_path,
            output_path=output_path  # Set to None if you do not want to save
        )
    except Exception as e:
        print(f"Error occurred: {e}")