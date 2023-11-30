import cv2
import mediapipe as mp
import os
import numpy as np

class MediapipeHandler:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic  # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

    def draw_styled_landmarks(self, image, results):
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                       self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                       self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                       )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                       )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

    def collect_frames(self, DATA_PATH, actions, no_sequences, sequence_length, start_folder):
        cap = cv2.VideoCapture(0)

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for action in actions:
                for sequence in range(start_folder, start_folder + no_sequences):
                    for frame_num in range(sequence_length):
                        ret, frame = cap.read()
                        image, results = self.mediapipe_detection(frame, holistic)
                        self.draw_styled_landmarks(image, results)

                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(500)
                        else:
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)

                        keypoints = self.extract_keypoints(results)  # Assuming you have a method extract_keypoints
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

        cap.release()
        cv2.destroyAllWindows()
