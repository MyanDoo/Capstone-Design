import cv2

def get_supported_resolution(cap):
    supported_resolutions = []
    for i in range(20):
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width == 0 or height == 0:
            break
        supported_resolutions.append((int(width), int(height)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)
    return supported_resolutions

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 지원하는 해상도 출력
supported_resolutions = get_supported_resolution(cap)
print("Supported Resolutions:", supported_resolutions)