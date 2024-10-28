# capture web cam image and save it to a file
# Usage: python cam.py <filename>
# Example: python cam.py test.jpg

import cv2

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture a single frame
ret, frame = cap.read()

# Check if the frame was captured correctly
if not ret:
    print("Error: Could not read frame.")
    exit()

# Save the captured frame to a file
# cv2.imwrite('captured_image.jpg', frame)
# print("Image captured and saved to 'captured_image.jpg'")

# show image on screen
cv2.imshow('Captured Image', frame)
cv2.waitKey(0)

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()


