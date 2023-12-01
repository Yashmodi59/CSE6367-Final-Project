from flask import Flask, render_template
import random
import cv2
import tensorflow as tf
import numpy as np
from flask import Response
from Gameplaying.game_play_detection import run_gesture_recognition_for_flask
app = Flask(__name__)

# List of image filenames
image_files = ['paper.jpeg', 'scissor.jpeg', 'stone.jpeg']

actions = np.array(['Paper', 'Stone', 'Scissor'])

model = tf.keras.models.load_model('CSE6367-Final-Project\Gameplaying\models\\actions.h5')
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


# Initialize OpenCV VideoCapture
video_capture = cv2.VideoCapture(0)  # 0 corresponds to the default camera

def generate_frames():
    while True:
        success, frame = video_capture.read()  # Read a frame from the video stream
        if not success:
            break
        else:
            # Run gesture recognition
            gesture_frame = run_gesture_recognition_for_flask(actions, model, colors, frame.copy())

            # Encode the frames to JPEG
            _, buffer1 = cv2.imencode('.jpg', frame)
            _, buffer2 = cv2.imencode('.jpg', gesture_frame)

            if not buffer1 or not buffer2:
                break

            frame_data = buffer1.tobytes()
            gesture_frame_data = buffer2.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + gesture_frame_data + b'\r\n')

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
 