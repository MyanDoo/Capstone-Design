const video = document.getElementById('webcam-feed');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const img = document.getElementById('captured-image');

// 웹캠 스트림 캡처 후 이미지 크기 조절
function captureAndResizeImage() {
    canvas.width = video.videoWidth * 2; // 이미지 너비를 2배로 조절
    canvas.height = video.videoHeight * 2; // 이미지 높이를 2배로 조절
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    img.src = canvas.toDataURL('image/png'); // Canvas에서 이미지 URL로 변환하여 이미지 태그에 적용
}

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error('Error accessing the webcam:', err);
  }
}

startWebcam();

document.getElementById("homeBtn").onclick = function() {
  location.href = 'templates/mainHome.html';
};

