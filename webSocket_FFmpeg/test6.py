import cv2
import mediapipe as mp
import numpy as np
import subprocess
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import base64
import socket
import ffmpeg

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


# Curl counter variables
counter = 0 
left_stage = None  # 왼쪽 팔의 stage 변수 초기화
right_stage = None  # 오른쪽 팔의 stage 변수 초기화
landmarks = None  # landmarks 변수를 초기화

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

#preprocess_image로 mediaPipe에서 사용가능한 형태로 변경해주기
def preprocess_image(encoded_image):
    decoded_data = base64.b64decode(encoded_image)
    np_data = np.frombuffer(decoded_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    return image

frame_data_list = []

def generate_frames(image_data):
    global counter, left_stage, right_stage, frame_data_list  # counter와 각 팔의 stage를 global 변수로 사용

    process = (
        ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='640x480')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='640x480')
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
      
            results = pose.process(image)
    
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            results = pose.process(image)
            annotated_image = image.copy()
        
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # 이미지 데이터를 전처리하여 mediapipe가 처리할 수 있는 형태로 변환
                processed_image = preprocess_image(frame_bytes)
            
                # Get coordinates for the left arm
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
                # Get coordinates for the right arm
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
                # Calculate angles for left and right arms
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
                # Visualize angles
                cv2.putText(processed_image, f"Left Angle: {left_angle}", 
                        tuple(np.multiply(left_elbow, [860, 680]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(processed_image, f"Right Angle: {right_angle}", 
                        tuple(np.multiply(right_elbow, [860, 680]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic for left arm
                if left_angle > 160:
                    left_stage = "left_down"
                    print("Left Down - Positive")
                if left_angle < 40:
                    if left_stage == 'left_down':
                        left_stage = "left_up"
                        print("Left Up - Positive")
                        counter += 1
                        print("Counter:", counter)
    
                # Curl counter logic for right arm
                if right_angle > 160:
                    right_stage = "right_down"
                    print("Right Down - Positive")
                if right_angle < 40:
                    if right_stage == 'right_down':
                        right_stage = "right_up"
                        print("Right Up - Positive")
                        counter += 1
                        print("Counter:", counter)

                
                #Image data encoding
                ret, buffer = cv2.imencode('.jpg', annotated_image)
                frame_bytes = base64.b64encode(buffer).decode('utf-8')

                #Add frame data to the list
                #frame_data_list.append(frame_bytes)
                # frame_data_list에 이미지 데이터 추가
                frame_data_list.append(processed_image)
                
                # 클라이언트로 프레임을 전송합니다.
                socketio.emit('frame', {'image': frame_bytes})

                process.stdin.write(buffer.tobytes())
                output = process.stdout.read(640 * 480 * 3)
                frame_bytes = base64.b64encode(output).decode('utf-8')

                yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes.encode() + b'\r\n')
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
                # Rep data
                cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
                # Stage data
                cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, left_stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, right_stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
                # Print landmark visibility for left arm
                print("Left Shoulder Visibility:", landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility)
                print("Left Elbow Visibility:", landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility)
                print("Left Wrist Visibility:", landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility)
        
                # Print landmark visibility for right arm
                print("Right Shoulder Visibility:", landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility)
                print("Right Elbow Visibility:", landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility)
                print("Right Wrist Visibility:", landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 ) 
        
                cv2.imshow('Mediapipe Feed', image)
                       
            except Exception as e:
                print("Exception:", e)
                # 예외 발생 시 리소스 해제를 위해 루프를 빠져나오도록 설정
                break

@socketio.on('frameData')  # 클라이언트로부터 'frameData' 이벤트를 받음
def handle_frame_data(data):
    # 클라이언트로부터 받은 데이터 처리
    image_data = data['image']  # 클라이언트로부터 받은 이미지 데이터

    # 이미지 데이터 처리 후, 클라이언트로 전송
    if frame_data_list:
        # 이미지 데이터 처리 (mediapipe를 사용한 분석)
        processed_data = frame_data_list.pop(0)
        #processed_data = generate_frames(image_data)  # 이미지 처리 함수
        # 처리된 데이터를 클라이언트로 전송
        emit('generate_frames', {'result': processed_data})  

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    try:
        socketio.run(app, port=5001)  # 포트 번호 변경
    finally:
        # 프로그램이 종료될 때 카메라 리소스를 해제
        cap.release()
        cv2.destroyAllWindows()