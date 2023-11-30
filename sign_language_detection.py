import tensorflow as tf
import cv2
import numpy as np
from utility import prob_viz
actions = np.array(['hello', 'thanks', 'iloveyou'])
model = tf.kersas.models.load_model('action.h5')
colors = [(245,117,16), (117,245,16), (16,117,245)]

from utils import MediapipeHandler
mediapipe_handler = MediapipeHandler()
# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8
# mp_holistic_instance = mediapipe_handler.mp_holistic
# mp_drawing_instance = mediapipe_handler.mp_drawing

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mediapipe_handler.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_handler.mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        mediapipe_handler.draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = mediapipe_handler.extract_keypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        #3. Viz logic
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
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()