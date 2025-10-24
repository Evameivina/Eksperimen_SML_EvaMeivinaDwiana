import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

with mlflow.start_run() as run:
    
    params = mlflow.active_run().data.params
    epochs = int(params.get("epochs", 100))
    batch_size = int(params.get("batch_size", 64))

    print(f"Training model dengan {epochs} epochs dan batch size {batch_size}")

    data = load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(X.shape[1],)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    mlflow.log_metric("loss", loss)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.tensorflow.log_model(model, "model")

    print(f"Training selesai Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
