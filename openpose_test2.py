import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, render_template

# 카메라 해상도
width = 600
height = 400

# OpenPose에서 기본 제공되는 부위별 값
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

# OpenPose에서 기본 제공되는 선으로 연결될 부위
POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# 각 파일 경로
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_iter_160000.caffemodel"

# 위의 경로에 있는 네트워크 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

app = Flask(__name__)

# 비디오 캡처 및 표시를 담당하는 함수
def display_video():
    # 웹캠 사용
    cap = cv2.VideoCapture(0)

    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 프레임에 무언가 추가적인 작업을 수행하려면 여기에 코드를 추가

        # 비디오 프레임을 OpenCV 윈도우에 표시
        cv2.imshow('Video', frame)

        # 'q' 키를 누르면 비디오 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 모든 작업이 끝나면 해제
    cap.release()
    cv2.destroyAllWindows()

# 별도의 스레드에서 OpenPose와 각도 계산을 수행하는 함수
def process_frame_and_calculate_angle():
    # 웹캠 사용
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = generate_virtual_frame(frame)

        # 화면에 결과 표시
        cv2.imshow('Video', frame)

        # 'q' 키를 누르면 비디오 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 모든 작업이 끝나면 해제
    cap.release()
    cv2.destroyAllWindows()

# OpenPose 및 각도 계산 함수
def generate_virtual_frame(frame):
    # 높이, 너비 및 채널 수를 가져옴
    imageHeight, imageWidth, _ = frame.shape

    # 이미지를 전처리하여 신경망에 입력할 수 있는 형태로 변경
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

    # 신경망의 입력 설정
    net.setInput(inpBlob)

    # 신경망의 예측 결과를 반환
    output = net.forward()

    # 출력의 높이와 너비를 가져옴
    H = output.shape[2]
    W = output.shape[3]

    points = [None] * 15

    for i in range(0, 15):
        # 해당 신체부위 신뢰도 얻기
        probMap = output[0, i, :, :]

        # global 최대값 찾기
        _, prob, _, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = int(imageWidth * point[0] / W)
        y = int(imageHeight * point[1] / H)

        # 키포인트 검출한 결과가 0.1보다 크면(검출한 곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
        if prob > 0.1:
            x = int(imageWidth * point[0] / W)
            y = int(imageHeight * point[1] / H)
            points[i] = (x, y)
        else:
            points.append(None)

    partA, partB, partC = points[4], points[2], points[8]  # 오른손목, 오른어깨, 오른골반

    if points[partA] and points[partB] and points[partC]:
        x1, y1 = points[partA] # 오른손목 좌표
        x2, y2 = points[partB] # 오른어깨 좌표
        x3, y3 = points[partC] # 오른골반 좌표

        # 좌표 지점에 원을 그려서 표시
        cv2.circle(frame, (x1, y1), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED) 
        cv2.circle(frame, (x2, y2), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, (x3, y3), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)

        # 각도 계산
        angle = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.degrees(angle)

        # 각도가 0 미만이면 360을 더해서 각도를 양수로 수정
        if angle < 0:
            angle += 360

        # 각도가 180 초과라면 360을 빼서 각도를 수정한 뒤 양수로 변환
        if angle > 180:
            angle -= 360
            angle = -angle

        # 각도가 60 미만 90 초과라면 잘못된 자세라는 판정으로 선을 빨간색으로 변경
        if 60 < angle < 90:
            cv2.line(frame, points[partA], points[partB], (0, 0, 255), 2)
            cv2.line(frame, points[partB], points[partC], (0, 0, 255), 2)

    # 캡처된 프레임 반환
    return frame

# 비디오 프레임을 바이트 스트림으로 변환하여 웹 페이지로 전송
def capture_frame():
    # 가상의 프레임 생성
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame_bytes = generate_virtual_frame(frame)

        if not ret:
            break

        # 프레임을 바이트 스트림으로 변환
        ret, buffer = cv2.imencode('.jpg', frame_bytes)
        frame_bytes = buffer.tobytes()

        # 프레임 반환
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(capture_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 별도의 스레드에서 OpenPose 및 각도 계산을 처리
    processing_thread = threading.Thread(target=process_frame_and_calculate_angle)
    processing_thread.daemon = True
    processing_thread.start()

    #app.run(debug=True)
    app.run(port=5500, debug=True)
    