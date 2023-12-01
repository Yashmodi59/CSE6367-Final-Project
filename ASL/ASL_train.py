from utils import MediapipeHandler
import os
import numpy as np
from utility import create_directories,load_sequences_and_labels
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import create_and_train_lstm_model,calculate_accuracy_and_confusion_matrix

DATA_PATH = os.path.join('Project\ASL\ASL_Train_Data')
actions = np.array(['Hello', 'goodbye', 'Thank you!', 'yes', 'no', 'I love you!'])
no_sequences = 30
sequence_length = 30
start_folder = 1
input_shape = (30, 126)  # Assuming input shape based on your example
num_classes = len(actions)  # Assuming actions is the list of classes

def run_action_recognition_pipeline(DATA_PATH, actions, no_sequences, sequence_length, start_folder, boolean_function=True):

    mediapipe_handler = MediapipeHandler()
    if boolean_function:
        create_directories(DATA_PATH, actions, no_sequences, sequence_length, start_folder)
        mediapipe_handler.collect_frames(DATA_PATH,actions,no_sequences,sequence_length,start_folder)
    sequences, labels = load_sequences_and_labels(DATA_PATH, actions, sequence_length)
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    model = create_and_train_lstm_model(X_train, y_train, actions, log_dir='Logs', epochs=50)
    accuracy, confusion_matrix = calculate_accuracy_and_confusion_matrix(model,X_test,y_test)
    print("Accuracy of testing data:-", accuracy)
    model.save("Project\ASL\models\ASLmodel.h5")


run_action_recognition_pipeline(DATA_PATH, actions, no_sequences, sequence_length, start_folder, boolean_function=False)