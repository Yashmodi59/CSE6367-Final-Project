import cv2
import numpy as np
import tensorflow as tf
from utils import MediapipeHandler  # Assuming you have a file named 'mediapipe_handler.py' with the class definition
from utility import prob_viz
# gesture_recognition.py
import cv2
import numpy as np

def run_gesture_recognition_for_flask(actions, model, colors, frame):
    # Create an instance of MediapipeHandler (adjust import path as needed)
    mediapipe_handler = MediapipeHandler()

    # New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    # Access mediapipe instances from the handler
    mp_holistic_instance = mediapipe_handler.mp_holistic
    mp_drawing_instance = mediapipe_handler.mp_drawing

    # Make detections
    image, results = mediapipe_handler.mediapipe_detection(frame, mp_holistic_instance.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5))
    print(results)

    # Draw landmarks
    mediapipe_handler.draw_styled_landmarks(image, results)

    # Prediction logic
    keypoints = mediapipe_handler.extract_keypoints(results)
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

    return image

def run_gesture_recognition(actions, model, colors):
    # Create an instance of MediapipeHandler
    mediapipe_handler = MediapipeHandler()

    # New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    # Access mediapipe instances from the handler
    mp_holistic_instance = mediapipe_handler.mp_holistic
    mp_drawing_instance = mediapipe_handler.mp_drawing

    cap = cv2.VideoCapture(0)

    # Set mediapipe model
    with mp_holistic_instance.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_handler.mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            mediapipe_handler.draw_styled_landmarks(image, results)

            # Prediction logic
            keypoints = mediapipe_handler.extract_keypoints(results)
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

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Call the function when running this script
if __name__ == "__main__":
    actions = np.array(['Paper', 'Stone', 'Scissor'])
    model = tf.keras.models.load_model('CSE6367-Final-Project\Gameplaying\models\\actions.h5')
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
    run_gesture_recognition(actions, model, colors)
