# Soccer Ball Detection with YOLO

This project uses the YOLO (You Only Look Once) model to detect soccer balls in images and real-time webcam feeds. The project is implemented in Python and uses the `ultralytics` library for YOLO, along with OpenCV for image processing.

## Project Structure

- `env_yolo/`: Virtual environment directory.
- `README.md`: This file.
- `requirements.txt`: List of dependencies.
- `src/`: Source code directory.
  - `cam_test.py`: Script to capture an image from the webcam.
  - `soccer_ball.py`: Script to detect soccer balls in images and real-time webcam feeds.
- `yolov8n.pt`: YOLO model file.

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/20wiz/yolo_basic.git
    cd https://github.com/20wiz/yolo_basic.git
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

4. **Download the YOLO model:**

    Place the `yolov8n.pt` file in the root directory of the project.

## Usage

### Detect Soccer Ball in an Image

1. **Run the `soccer_ball.py` script:**

    ```sh
    python src/soccer_ball.py
    ```

2. **Modify the `image_path` variable in the script to point to your image file:**

    ```python
    image_path = '.\\1.webp'  # Change to your image path
    ```

### Capture an Image from Webcam

1. **Run the `cam_test.py` script:**

    ```sh
    python src/cam_test.py
    ```

2. **The script will capture an image from the webcam and display it.**


### Real-Time Soccer Ball Detection with Webcam

1. **Run the `soccer_ball.py` script:**

    ```sh
    python src/soccer_ball.py
    ```

2. **The script will start the webcam and display the real-time detection results. Press 'q' to quit.**



## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)