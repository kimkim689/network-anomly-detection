import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import os

def preprocess_data_for_cnn(X, y, time_steps=10, step=1):
    X_reshaped, y_reshaped = [], []
    for i in range(0, len(X) - time_steps, step):
        X_reshaped.append(X[i:i+time_steps])
        y_reshaped.append(y[i+time_steps])
    return np.array(X_reshaped), np.array(y_reshaped)

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(X_train, y_train, X_test, y_test):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape data for CNN
    X_train_cnn, y_train_cnn = preprocess_data_for_cnn(X_train_scaled, y_train)
    X_test_cnn, y_test_cnn = preprocess_data_for_cnn(X_test_scaled, y_test)

    # Create and train the model
    model = create_cnn_model((X_train_cnn.shape[1], X_train_cnn.shape[2]))
    model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test_cnn)
    y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)

    print(f"Accuracy: {accuracy_score(y_test_cnn, y_pred_classes)}")
    print(f"ROC-AUC: {roc_auc_score(y_test_cnn, y_pred)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_cnn, y_pred_classes))
    print("\nClassification Report:")
    print(classification_report(y_test_cnn, y_pred_classes))

    # Save the model and scaler
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(model_dir, 'cnn_model.h5'))
    np.save(os.path.join(model_dir, 'cnn_scaler.npy'), scaler.scale_)
    np.save(os.path.join(model_dir, 'cnn_mean.npy'), scaler.mean_)

    return model, scaler

def load_cnn_model():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(model_dir, 'cnn_model.h5'))
    scaler = StandardScaler()
    scaler.scale_ = np.load(os.path.join(model_dir, 'cnn_scaler.npy'))
    scaler.mean_ = np.load(os.path.join(model_dir, 'cnn_mean.npy'))
    return model, scaler