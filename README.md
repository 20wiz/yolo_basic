# Soccer Ball Detection with YOLO

This project uses the YOLO (You Only Look Once) model to detect soccer balls in images and real-time webcam feeds. The project is implemented in Python and uses the `ultralytics` library for YOLO, along with OpenCV for image processing.

## Project Structure

- `dataset/`: Contains the dataset for training and validation.
- `env_yolo/`: Virtual environment directory.
- `output/`: Directory for output files.
- `README.md`: This file.
- `requirements.txt`: List of dependencies.
- `src/`: Source code directory.
  - `cam_test.py`: Script to capture an image from the webcam.
  - `soccer_ball.py`: Script to detect soccer balls in images.
  - `soccer_ball_cam.py`: Script for real-time soccer ball detection using a webcam.
  - `soccer_ball_video.py`: Script to detect soccer balls in videos.
  - `soccer_ball_track.py`: Script to track soccer balls in videos using YOLO.
  - `soccer_ball_deepsort.py`: Script to track soccer balls in videos using DeepSort.
  
- `yolo*.pt`: YOLO model files.

## Setup

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv env_yolo
    source env_yolo/Scripts/activate  # On Windows
    # source env_yolo/bin/activate    # On Unix or MacOS
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **YOLO model:**

    `yolov8n.pt` file will be downloaded in the root directory of the project.

## Usage

### Detect Soccer Ball in an Image

1. **Run the `soccer_ball.py` script:**

    ```sh
    python src/soccer_ball.py
    ```

2. **Modify the `image_path` variable in the script to point to your image file:**

    ```python
    image_path = '.\\test1.webp'  # Change to your image path
    ```

### Real-Time Soccer Ball Detection with Webcam

1. **Run the `soccer_ball_cam.py` script:**

    ```sh
    python src/soccer_ball_cam.py
    ```

2. **The script will start the webcam and display the real-time detection results. Press 'q' to quit.**

### Detect Soccer Ball in a Video

1. **Run the `soccer_ball_video.py` script:**

    ```sh
    python src/soccer_ball_video.py
    ```

2. **Modify the `video_path` variable in the script to point to your video file:**

    ```python
    video_path = 'soccer_1.mp4'  # Change to your video path
    ```

### Track Soccer Ball in a Video using DeepSort

1. **Run the `soccer_ball_deepsort.py` script:**

    ```sh
    python src/soccer_ball_deepsort.py
    ```

2. **Modify the `video_path` variable in the script to point to your video file:**

    ```python
    video_path = 'soccer_1.mp4'  # Change to your video path
    ```

### Track Soccer Ball in a Video using YOLO

1. **Run the `soccer_ball_track.py` script:**

    ```sh
    python src/soccer_ball_track.py
    ```

2. **Modify the `video_path` variable in the script to point to your video file:**

    ```python
    video_path = 'soccer_1.mp4'  # Change to your video path
    ```

### Capture an Image from Webcam

1. **Run the `cam_test.py` script:**

    ```sh
    python src/cam_test.py
    ```

2. **The script will capture an image from the webcam and display it.**

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [Claude](https://claude.ai/)