import argparse
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

print(f"Training model dengan {args.epochs} epochs dan batch size {args.batch_size}")

data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with mlflow.start_run():
    
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", args.batch_size)
    
    # Train model
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Log metrics
    mlflow.log_metric("loss", loss)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.tensorflow.log_model(model, "model")

    print(f"Training selesai. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
