# track by YOLO
import torch
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLO11 model
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11s.pt")
# model = YOLO("yolo11m.pt")
model = YOLO("yolov8m.pt")
# model = YOLO("yolo11m-seg.pt")
# device = 'cpu'   
device = 'cuda:0'  # GPU 사용 시 'cuda:0'
model.to(device)

# Initialize the DeepSort tracker
tracker = DeepSort(max_age=30, n_init=3)

# Open the video file
video_path = "soccer_1.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

# 비디오 속성 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
end_frames = int(fps * 5)  # 종료 시간 까지 프레임 수 계산

# Counter for frames with detected soccer balls
ball_detected_frames = 0

# Loop through the video frames
frame_count = 0
while cap.isOpened() and frame_count < end_frames:
    # Read a frame from the video
    success, frame = cap.read()
    frame_count += 1

    if success:
        results =model(frame, device=device, conf=0.3,  imgsz=1920) #

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Check if a soccer ball is detected
        ball_detected = False
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if class_name in ['sports ball', 'ball']:
                    ball_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Convert the bounding box to ltwh format
                    ltwh = [x1, y1, x2 - x1, y2 - y1]
                    
                    # Move the tensor to the CPU and convert to NumPy array
                    ltwh = torch.tensor(ltwh).cpu().numpy()
                    confidence = torch.tensor(confidence).cpu().numpy()
                    
                    detections.append((ltwh, confidence, class_id))
                    continue
            # if ball_detected:
            #     break

        if ball_detected:
            ball_detected_frames += 1

        # Update the DeepSort tracker with the detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Visualize the tracking results
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Draw the bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)


        # resized_frame = cv2.resize(annotated_frame, (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2))
        resized_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # Display the annotated frame
        cv2.imshow("YOLO and DeepSort Tracking", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Print the number of frames with detected soccer balls
print(f"Number of frames with detected soccer balls: {ball_detected_frames}")