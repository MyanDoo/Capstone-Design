document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const socket = io.connect('http://' + document.domain + ':5001');

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                video.srcObject = stream;
                video.play();

                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = video.width;
                canvas.height = video.height;

                setInterval(() => {
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);

                    const frame = canvas.toDataURL('image/jpeg'); // 프레임을 JPEG 형식으로 변환

                    // 웹캠 데이터를 서버로 전송
                    socket.emit('frameData', { image: frame }); // 수정된 부분: frameData 이벤트 사용
                }, 1000 / 30); // 매 프레임마다 30fps로 설정, 필요에 따라 조정 가능
            })
            .catch((error) => {
                console.error('웹캠 에러:', error);
            });

        // 서버로부터 처리된 데이터 수신
        socket.on('generate_frames', function(data) {s
            // 서버로부터 받은 데이터 처리
            const imageElement = document.getElementById('processed-image');
            const imageData = data.result; // 이미지 데이터
            // 여기에서 받은 데이터를 사용하여 클라이언트 측에서의 추가 작업 수행 가능
            console.log('Received data from server:', data);
            // 이미지 데이터를 받아서 이미지 요소에 삽입
            imageElement.src = 'data:image/jpeg;base64,' + imageData;
        });
    } else {
        console.error('웹캠이 지원되지 않습니다.');
    }
});
