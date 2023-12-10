from flask import Flask, render_template
import cv2
import threading
import time
import numpy as np
import tensorflow as tf
from utils import MediapipeHandler
from utility import prob_viz
import random
app = Flask(__name__)
actions = np.array([ 'stone','paper', 'scissor'])
image_files = ['paper.jpeg', 'scissor.jpeg', 'stone.jpeg']

model = tf.keras.models.load_model('Gameplaying\models\\actions.h5')
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

mediapipe_handler = MediapipeHandler()
sequence = []
sentence = []
threshold = 0.8

# Global variables
video_feed_open = False
output_sentence = ""
video_thread = None  # Keep track of the video thread
video_timer = None  # Keep track of the timer

def open_video_feed():
    global video_feed_open, output_sentence, sequence, sentence
    # Create an instance of MediapipeHandler
    mediapipe_handler_instance = MediapipeHandler()

    # Access mediapipe instances from the handler
    mp_holistic_instance = mediapipe_handler_instance.mp_holistic
    mp_drawing_instance = mediapipe_handler_instance.mp_drawing

    # OpenCV Video Capture
    cap = cv2.VideoCapture(0)

    # Set video window dimensions
    cap.set(3, 640)
    cap.set(4, 480)

    while video_feed_open:
        # Set mediapipe model
        with mp_holistic_instance.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_handler_instance.mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            mediapipe_handler_instance.draw_styled_landmarks(image, results)

            # Prediction logic
            keypoints = mediapipe_handler_instance.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

                # Viz logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the frame on the webpage
            cv2.imshow('Video Feed', image)
            cv2.imwrite('Gameplaying\static\\result_image.png', image)

            cv2.waitKey(1)
            # print("hello")
            if not video_feed_open:
                break

    # Release the camera when the video feed is closed
    cap.release()
    cv2.destroyAllWindows()

    # Set output sentence after video feed is closed
    output_sentence = ' '.join(sentence)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html', output="Welcome to game playing")

@app.route('/open_video_feed')
def open_video():
    global video_feed_open, video_thread, video_timer

    if not video_feed_open:
        video_feed_open = True

        # Start a thread to open the video feed
        video_thread = threading.Thread(target=open_video_feed)
        video_thread.start()

        # Schedule the video feed to be closed after 1 minute
        video_timer = threading.Timer(60, close_video)
        video_timer.start()

        output_sentence = ' '.join(sentence)

    return render_template('index.html', output="Click Go To Result Button once frame closed")

@app.route('/result')
def result():
    global output_sentence
    # Pass the output sentence and an image to the result.html template
    print("_____",sentence)
    rt = ''
    random_image = random.choice(image_files)
    user_choice = sentence[-1]
    computer_choice = str(random_image.split('.')[0])
    if user_choice == computer_choice:
        rt = "It's a tie!"
    elif (
        (user_choice == 'stone' and computer_choice == 'scissor') or
        (user_choice == 'paper' and computer_choice == 'stone') or
        (user_choice == 'scissor' and computer_choice == 'paper')
    ):
        rt = "You win!"
    else:
        rt = "Computer wins!"

    return render_template('scoring.html', score = rt, output=sentence[-1],user_image = 'static/result_image.png', image_path='static/' + random_image)

def close_video():
    global video_feed_open, output_sentence, video_thread

    if video_feed_open:
        video_feed_open = False
        output_sentence = "Video feed closed after 1 minute."

        # Wait for the video thread to finish
        video_thread.join()
        # output_sentence = ' '.join(sentence)


# ... (remaining code)

if __name__ == '__main__':
    app.run(debug=True)
