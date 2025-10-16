import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# File input dan output
raw_file = r"C:\Users\Eva\Documents\Eksperimen_SML_EvaMeivinaDwiana\namadataset_raw\StudentsPerformance.csv"
output_folder = r"C:\Users\Eva\Documents\Eksperimen_SML_EvaMeivinaDwiana\preprocessing\namadataset_preprocessing"
output_file = os.path.join(output_folder, "StudentsPerformance_preprocessed.csv")

os.makedirs(output_folder, exist_ok=True)

# Dataset
data = pd.read_csv(raw_file)
print("Dataset berhasil dibaca, contoh data:")
print(data.head())

# Menentukan kolom numerik dan kategorikal
num_cols = ['math score', 'reading score', 'writing score']
cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

# Preprocessing
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Menggabungkan semua transformasi
transformer = ColumnTransformer([
    ('num', scaler, num_cols),
    ('cat', encoder, cat_cols)
])

# Menjalankan preprocessing
processed_data = transformer.fit_transform(data)

# Mengambil nama kolom hasil encoding
encoded_features = transformer.named_transformers_['cat'].get_feature_names_out(cat_cols)

# Menggabungkan semua kolom jadi satu
final_columns = num_cols + list(encoded_features)

# Menyimpan ke DataFrame biar lebih mudah dibaca
df_result = pd.DataFrame(processed_data, columns=final_columns)

# Menyimpan hasil akhir 
df_result.to_csv(output_file, index=False)
print(f"Hasil preprocessing disimpan di: {output_file}")
