from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
def create_and_train_lstm_model(X_train, y_train, actions, log_dir='Logs', epochs=2000):
    # Create TensorBoard callback
    tb_callback = TensorBoard(log_dir=log_dir)

    # Create Sequential model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback])

    return model

def calculate_accuracy_and_confusion_matrix(model, X_test, y_test):
    # Step 1: Predict using the model
    yhat = model.predict(X_test)

    # Step 2: Convert predictions and true labels to lists
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    # Step 3: Calculate confusion matrix
    confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)

    # Step 4: Calculate accuracy score
    accuracy = accuracy_score(ytrue, yhat)

    return accuracy, confusion_matrix
