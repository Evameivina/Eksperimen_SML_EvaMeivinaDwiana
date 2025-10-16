import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Path dataset hasil preprocessing
dataset_path = "namadataset_preprocessing/StudentsPerformance_preprocessed.csv"

# Membaca dataset
data = pd.read_csv(dataset_path)
print("Kolom dataset:", list(data.columns))
print(f"Dataset berhasil dibaca dengan ukuran: {data.shape}")

# Membuat kolom rata-rata skor
data["average_score"] = data[["math score", "reading score", "writing score"]].mean(axis=1)

# Menentukan batas rata-rata (threshold)
threshold = data["average_score"].mean()
data["performance"] = data["average_score"].apply(
    lambda x: "high" if x >= threshold else "low"
)

# Distribusi label
print("\nDistribusi kategori 'performance':")
print(data["performance"].value_counts())

# Memisahkan fitur dan target
X = data.drop(columns=["performance"])
y = data["performance"]

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Mengaktifkan autolog MLflow
mlflow.sklearn.autolog()

# Training model
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

print("Model berhasil dilatih. Cek MLflow UI untuk melihat hasil run dan artefak.")
