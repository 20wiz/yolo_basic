from ultralytics import YOLO
import cv2
import time

def detect_soccer_ball_video(video_path, output_path=None):
    """
    비디오에서 축구공을 감지하는 함수
    
    Args:
        video_path (str): 입력 비디오 파일 경로
        output_path (str, optional): 결과 비디오 저장 경로. None이면 저장하지 않음
    """
    # CPU 사용
    device = 'cpu' # GPU 사용 시 'cuda:0'
    
    # YOLO 모델 로드
    model = YOLO('yolov8n.pt')
    model.to(device)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
    
    # 비디오 속성 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 결과 비디오 저장 설정
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # FPS 계산을 위한 변수
    prev_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    # 프레임 카운터
    frame_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_counter += 1
            
            # 객체 감지 수행
            results = model(frame, device=device)
            
            # FPS 계산
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                prev_time = current_time
            
            # 진행률 계산
            progress = (frame_counter / total_frames) * 100
            
            # 결과 처리 및 시각화
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    if class_name in ['sports ball', 'ball']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f'Soccer Ball: {confidence:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # FPS와 진행률 표시
            cv2.putText(frame, f'FPS: {fps_display}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Progress: {progress:.1f}%', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 결과 저장
            if output_path:
                out.write(frame)

            # 프레임 크기를 1/4로 조정
            resized_frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
            
            # 결과 화면 표시
            cv2.imshow('Soccer Ball Detection', resized_frame)
    
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"처리 중 에러 발생: {e}")
        
    finally:
        # 리소스 해제
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        print("처리 완료!")

if __name__ == "__main__":
    # 비디오 파일 경로 설정
    video_path = "soccer_1.mp4"  # 본인의 비디오 파일 경로로 수정
    output_path = "output_soccer.mp4"  # 결과 저장할 경로
    
    try:
        detect_soccer_ball_video(
            video_path=video_path,
            output_path=output_path  # 저장하지 않으려면 None으로 설정
        )
    except Exception as e:
        print(f"에러 발생: {e}")