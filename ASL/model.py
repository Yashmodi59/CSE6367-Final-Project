from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

def create_and_train_lstm_model(X, y, actions, log_dir='Logs', epochs=2000, n_splits=5):
    # Create StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize variables to store overall metrics
    overall_loss = []
    overall_accuracy = []

    # Create Sequential model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(actions.shape[0], activation='softmax'))

    # Plot the model architecture
    # plot_model(model, to_file=f'fold_{fold_index}_model.png', show_shapes=True, show_layer_names=True)

    # Loop through each fold
    for fold_index, (train_index, val_index) in enumerate(skf.split(X, np.argmax(y, axis=1))):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Compile the model
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # Create a unique log directory for each fold
        fold_log_dir = os.path.join(log_dir, f'fold_{fold_index}')
        os.makedirs(fold_log_dir, exist_ok=True)

        # Create TensorBoard callback
        tb_callback = TensorBoard(log_dir=fold_log_dir)

        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback, EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)],
                            validation_data=(X_val, y_val))

        # Evaluate the model on the validation set
        loss, accuracy = model.evaluate(X_val, y_val)

        print(f'Fold Loss: {loss}, Fold Accuracy: {accuracy}')

        overall_loss.append(loss)
        overall_accuracy.append(accuracy)

    # Calculate and print the overall metrics
    mean_loss = np.mean(overall_loss)
    mean_accuracy = np.mean(overall_accuracy)
    print(f'Overall Mean Loss: {mean_loss}, Overall Mean Accuracy: {mean_accuracy}')

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
