from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)
    # Set the width and height of the frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1290)  # 가로 해상도 1290 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)  # 세로 해상도 800 설정
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    #counter = 0
    left_counter = 0
    right_counter = 0 
    left_stage = None  # 왼쪽 팔의 stage 변수 초기화
    right_stage = None  # 오른쪽 팔의 stage 변수 초기화

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ... (이전 코드에서 가져온 나머지 부분)
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                # Get coordinates for the left arm
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Get coordinates for the right arm
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                def calculate_angle(a,b,c):
                    a = np.array(a) # First
                    b = np.array(b) # Mid
                    c = np.array(c) # End
                    
                    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    angle = np.abs(radians*180.0/np.pi)
                    
                    if angle >180.0:
                        angle = 360-angle
                        
                    return angle
                
                # Calculate angles for left and right arms
                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Visualize angles
                # 좌우 반전되지 않도록 텍스트 좌표 계산
                text_pos_left = tuple(np.multiply(left_elbow, [860, 680]).astype(int))
                text_pos_right = tuple(np.multiply(right_elbow, [860, 680]).astype(int))

                cv2.putText(image, f"Left Angle: {left_angle}", 
                        text_pos_left, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Right Angle: {right_angle}", 
                        text_pos_right, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 선의 색상을 나타내는 변수 초기화
                dott_color = (245, 117, 66)  # 초기 점의 색상 설정, 주황색
                line_color = (245, 117, 66)  # 초기 선의 색상 설정, 주황색

                # Curl counter logic for left arm
                if left_angle > 160:
                    left_stage = "left_down"
                    print("Left Down - Positive")
                if left_angle < 40:
                    if left_stage == 'left_down':
                        left_stage = "left_up"
                        print("Left Up - Positive")
                        left_counter += 1
                        left_color = (66, 135, 245)  # 원하는 다른 색상으로 변경, 파란색
                        print("Counter:", left_counter)
                if 60 < left_angle and left_angle < 150:
                    dott_color = (66, 0, 245)  # 빨간색
                    line_color = (66, 135, 245)  # 파란색

                # Curl counter logic for right arm
                if right_angle > 160:
                    right_stage = "right_down"
                    print("Right Down - Positive")
                if right_angle < 40:
                    if right_stage == 'right_down':
                        right_stage = "right_up"
                        print("Right Up - Positive")
                        right_counter += 1
                        #right_color = (66, 135, 245)  # 원하는 다른 색상으로 변경, 파란색
                        print("Counter:", right_counter)
                if 60 < right_angle and right_angle < 150:
                    dott_color = (66, 0, 245)  # 빨간색
                    line_color = (66, 135, 245)  # 파란색
            except AttributeError:
                print("No pose landmarks detected.")
            except Exception as e:
                print(f"An error occurred: {e}")

            # Image 좌우 반전
            image = cv2.flip(image, 1)

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (1290,70), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (30,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)
            # Count left
            cv2.putText(image, str(left_counter), 
                        (205,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            # Count right
            cv2.putText(image, str(right_counter), (880, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Stage data
            cv2.putText(image, 'STAGE', (image.shape[1] - 120, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, left_stage, 
                (image.shape[1] - 1010, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, right_stage, 
                (image.shape[1] - 340, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Render detections
            # 관절 좌표값을 좌우 반전하여 그리기
            flipped_landmarks = results.pose_landmarks

            if flipped_landmarks is not None:
                for landmark in flipped_landmarks.landmark:
                    landmark.x = 1.0 - landmark.x  # 좌우 반전

            mp_drawing.draw_landmarks(image, flipped_landmarks, mp_pose.POSE_CONNECTIONS,
                          mp_drawing.DrawingSpec(color=dott_color, thickness=2, circle_radius=2), 
                          mp_drawing.DrawingSpec(color=line_color, thickness=2, circle_radius=2) 
                          )
            
            cv2.imshow('Mediapipe Feed', image)

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)