import os
import numpy as np
import cv2
def create_directories(DATA_PATH, actions, no_sequences, sequence_length, start_folder):
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        
        # Check if the action directory exists, if not, create it
        if not os.path.exists(action_path):
            os.makedirs(action_path)
        
        dirmax = 0
        try:
            # Try to get the maximum value from existing directories
            dirmax = np.max(np.array(os.listdir(action_path)).astype(int))
        except ValueError:
            pass  # Handle the case where there are no directories yet
        
        for sequence in range(1, no_sequences + 1):
            try:
                # Create the sequence directory inside the action directory
                os.makedirs(os.path.join(action_path, str(dirmax + sequence)))
            except FileExistsError:
                pass  # Handle the case where the directory already exists

def load_sequences_and_labels(DATA_PATH, actions, sequence_length):
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    return np.array(sequences), np.array(labels)

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame