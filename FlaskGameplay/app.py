from flask import Flask, render_template
import random
import cv2
from flask import Response

app = Flask(__name__)

# List of image filenames
image_files = ['paper.jpeg', 'scissor.jpeg', 'stone.jpeg']

# Initialize OpenCV VideoCapture
video_capture = cv2.VideoCapture(0)  # 0 corresponds to the default camera

def generate_frames():
    while True:
        success, frame = video_capture.read()  # Read a frame from the video stream
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Get a random image filename
    random_image = random.choice(image_files)
    return render_template('index.html', random_image=random_image)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
