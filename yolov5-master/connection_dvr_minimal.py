from flask import Flask, Response
import cv2

app = Flask(__name__)

# Replace with your DVR/camera's video source
video_source = "rtsp://smart:1234@192.168.1.207:554/avstream/channel=7/stream=1.sdp"

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(video_source)
        while True:
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
