import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Path dataset hasil preprocessing
dataset_path = "namadataset_preprocessing/StudentsPerformance_preprocessed.csv"

# Membaca dataset
data = pd.read_csv(dataset_path)
print("Dataset berhasil dibaca:", data.shape)

# Membuat kolom rata-rata nilai
data["average_score"] = data[["math score", "reading score", "writing score"]].mean(axis=1)

# Menentukan batas rata-rata 
threshold = data["average_score"].mean()
data["performance"] = data["average_score"].apply(lambda x: "high" if x >= threshold else "low")

# Pisahkan fitur dan target
X = data.drop(columns=["performance"])
y = data["performance"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Menentukan parameter grid untuk tuning
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

# Menginisialisasi model dan GridSearchCV
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)

# Menjalankan experiment MLflow
with mlflow.start_run():
    print("Proses tuning dimulai...")
    grid.fit(X_train, y_train)

    # Mengambil model terbaik
    best_model = grid.best_estimator_
    print("Model terbaik:", grid.best_params_)

    # Evaluasi model
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="high")
    prec = precision_score(y_test, y_pred, pos_label="high")
    rec = recall_score(y_test, y_pred, pos_label="high")

    # Manual logging ke MLflow
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.log_param("best_params", grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

print("Proses tuning dan logging selesai. Cek MLflow UI untuk melihat hasil eksperimen.")
