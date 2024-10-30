# track by YOLO

import cv2
from ultralytics import YOLO

# Load the YOLO11 model
# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11s.pt")
# model = YOLO("yolo11m.pt")
model = YOLO("yolov8m.pt")
# model = YOLO("yolo11m-seg.pt")
# device = 'cpu'   
device = 'cuda:0'  # Use 'cuda:0' for GPU
model.to(device)

# Open the video file
video_path = "soccer_1.mp4"
output_path = "output_soccer_track.mp4"  # Path to save the result

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Cannot open video: {video_path}")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
end_frames = int(fps * 5)  # Calculate the number of frames until the end time

# Set up result video saving
if output_path:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Counter for frames with detected soccer balls
ball_detected_frames = 0

# Loop through the video frames
frame_count = 0
while cap.isOpened() and frame_count < end_frames:
    # Read a frame from the video
    success, frame = cap.read()
    frame_count += 1

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        # results = model.track(frame, persist=True, conf=0.3,  iou=0.2, imgsz=1920, classes=32)
        results = model.track(frame, persist=True, conf=0.3,  iou=0.2, imgsz=1920)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Check if a soccer ball is detected
        ball_detected = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if class_name in ['sports ball', 'ball']:
                    ball_detected = True
                    # break
            if ball_detected:
                break

        if ball_detected:
            ball_detected_frames += 1

        # Save result
        if output_path:
            out.write(annotated_frame)

        resized_frame = cv2.resize(annotated_frame, (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2))

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", resized_frame)

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