# automate_EvaMeivinaDwiana.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Path file input & output
RAW_PATH = r"C:\Users\Eva\Documents\Eksperimen_SML_EvaMeivinaDwiana\namadataset_raw\StudentsPerformance.csv"
OUTPUT_DIR = r"C:\Users\Eva\Documents\Eksperimen_SML_EvaMeivinaDwiana\preprocessing\namadataset_preprocessing"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "StudentsPerformance_preprocessed.csv")

# Folder output (Opsional)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
data = pd.read_csv(RAW_PATH)
print("Dataset berhasil dibaca")
print(data.head())

# Definisikan kolom numerik & kategorikal
numerical_cols = ['math score', 'reading score', 'writing score']
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# Buat preprocessor (scaling + encoding)
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False, drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, numerical_cols),
        ('cat', encoder, categorical_cols)
    ]
)

# Preprocessing
data_preprocessed = preprocessor.fit_transform(data)

# Mengambil nama kolom hasil encoding
encoded_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)

# Menggabungkan nama kolom numerik + encoded
all_cols = numerical_cols + list(encoded_cols)

# Membuat DataFrame hasil preprocessing
df_preprocessed = pd.DataFrame(data_preprocessed, columns=all_cols)

# Menyimpan hasil preprocessing
df_preprocessed.to_csv(OUTPUT_PATH, index=False)
print(f"Hasil preprocessing berhasil disimpan ke: {OUTPUT_PATH}")

# Menampilkan 5 baris pertama (contoh)
print("Contoh hasil preprocessing:")
print(df_preprocessed.head())
