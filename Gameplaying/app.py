from flask import Flask, render_template
import random
import cv2
from utils import MediapipeHandler
import tensorflow as tf
import numpy as np
from flask import Response
from utility import prob_viz

app = Flask(__name__)

# List of image filenames
image_files = ['paper.jpeg', 'scissor.jpeg', 'stone.jpeg']

actions = np.array(['Paper', 'Stone', 'Scissor'])

model = tf.keras.models.load_model('Gameplaying\models\\actions.h5')
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

mediapipe_handler = MediapipeHandler()
sequence = []
sentence = []
threshold = 0.8
random_image = None
cap = cv2.VideoCapture(0)

def gen_frames():
    global sequence  # Declare sequence as a global variable
    global sentence
    global random_image  # Declare random_image as a global variable

    with mediapipe_handler.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            image, results = mediapipe_handler.mediapipe_detection(frame, holistic)
            mediapipe_handler.draw_styled_landmarks(image, results)

            keypoints = mediapipe_handler.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 420), (640, 470), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (0, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def text_feed():
    while cap.isOpened():
        if len(sentence) > 0:
            yield f"data: {' '.join(sentence)}\n\n"

def output_feed():
    while cap.isOpened():
        if len(sentence) > 0:
            output = ""
            user_choice = sentence[-1]
            computer_choice = str(random_image.split('.')[0])

            if user_choice == computer_choice:
                output = "It's a tie!"
            elif (
                (user_choice == 'Stone' and computer_choice == 'Scissor') or
                (user_choice == 'Paper' and computer_choice == 'Stone') or
                (user_choice == 'Scissor' and computer_choice == 'Paper')
            ):
                output = "You win!"
            else:
                output = "Computer wins!"

            yield f"output: {output}\n\n"
            print(output)

@app.route('/text_feed')
def text_feed_route():
    return Response(text_feed(), content_type='text/event-stream')

@app.route('/scoring')
def score_feed_route():
    return Response(output_feed(), content_type='text/event-stream')

@app.route('/')
def index():
    # Get a random image filename
    global random_image
    random_image = random.choice(image_files)
    return render_template('index.html', random_image=random_image)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
