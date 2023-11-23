import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

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

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            if results.pose_landmarks:  # 결과에 포즈 랜드마크가 있는지 확인
                landmarks = results.pose_landmarks.landmark

                if landmarks is not None:  # 랜드마크가 존재하는지 확인하는 조건 추가
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
                    cv2.putText(image, f"Left Angle: {left_angle}", 
                        tuple(np.multiply(left_elbow, [860, 680]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Right Angle: {right_angle}", 
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

                    if landmarks is not None:
                        if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility is not None:
                            print("Left Shoulder Visibility:", landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility)
    
        except Exception as e:
            print("Exception:", e)
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (600,73), (245,117,16), -1)
        
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
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, right_stage, 
                    (250,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        
        if landmarks is not None:
            if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility is not None:
                print("Left Shoulder Visibility:", landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility)
                print("Left Elbow Visibility:", landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility)
                print("Left Wrist Visibility:", landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility)

                # Print landmark visibility for right arm
                print("Right Shoulder Visibility:", landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility)
                print("Right Elbow Visibility:", landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility)
                print("Right Wrist Visibility:", landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility)

        # Print landmark visibility for left arm
        #print("Left Shoulder Visibility:", landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 ) 

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            landmarks = None
            break

    cap.release()
    cv2.destroyAllWindows()