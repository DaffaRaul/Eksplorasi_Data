
from pathlib import Path

import pandas as pd

# 1. Menentukan lokasi dataset
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_path = base_dir / "covid_19_indonesia_time_series_all.csv"

if not data_path.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {data_path}")

# 2. Membaca dataset dengan pandas
# Dataset ini memakai format CSV standar (separator koma), jadi tidak perlu sep=';'
df = pd.read_csv(data_path)

print("=" * 70)
print("EKSPLORASI DATASET COVID-19 INDONESIA")
print("=" * 70)

# 3. Menampilkan beberapa baris awal dan akhir
print("\n1. 5 data pertama:")
print(df.head())

print("\n2. 5 data terakhir:")
print(df.tail())

# 4. Ukuran dataset
print("\n3. Ukuran dataset:")
print("Baris:", df.shape[0])
print("Kolom:", df.shape[1])
print(df.shape)

# 5. Nama kolom
print("\n4. Nama kolom:")
print(df.columns.tolist())

# 6. Informasi dataset
print("\n5. Informasi tipe data:")
df.info()

# 7. Statistik deskriptif untuk kolom numerik
print("\n6. Statistik deskriptif (kolom numerik):")
print(df.describe(include='number'))

# 8. Mengecek data duplikat
print("\n7. Jumlah duplikat: ")
print("Baris duplikat:", df.duplicated().sum())

# 9. Mengecek nilai null
print("\n8. Jumlah nilai null per kolom:")
print(df.isnull().sum().sort_values(ascending=False))
print("\nTotal nilai null seluruh dataset:", df.isnull().sum().sum())

# 10. Menghitung distribusi lokasi (contoh kolom kategori)
print("\n9. Distribusi lokasi (Location):")
print(df["Location"].value_counts().head(10))

# 11. Menghitung kasus tertinggi per lokasi
print("\n10. Total kasus tertinggi per lokasi:")
print(df.groupby("Location")["Total Cases"].max().sort_values(ascending=False).head(10))

# 12. Cek outlier dengan IQR untuk beberapa kolom numerik penting
numeric_cols = df.select_dtypes(include='number').columns
print("\n11. Deteksi outlier (IQR) untuk kolom numerik utama:")
outlier_summary = []
for col in ["New Cases", "New Deaths", "Total Cases", "Total Deaths", "Population"]:
    if col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        outlier_summary.append({"Kolom": col, "Q1": q1, "Q3": q3, "IQR": iqr, "Lower": lower_bound, "Upper": upper_bound, "Outlier": outlier_count})

print(pd.DataFrame(outlier_summary).to_string(index=False))

# 13. Cek apakah perlu encoding data
print("\n12. Kolom kategori yang mungkin perlu di-encode:")
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
print(cat_cols)

# 14. Kesimpulan eksplorasi
print("\n13. Kesimpulan: ")
print(f"- Jumlah baris duplikat: {df.duplicated().sum()}")
print(f"- Total nilai null: {df.isnull().sum().sum()}")
print("- Outlier ada pada beberapa kolom numerik, tetapi tidak selalu perlu dihapus jika tujuan analisis bersifat eksplorasi. Untuk model prediksi, biasanya perlu penanganan outlier.")
print("- Encoding data diperlukan jika kolom kategori akan dipakai dalam model machine learning. Untuk analisis deskriptif, encoding belum selalu dibutuhkan.")

# 15. Simpan hasil eksplorasi sederhana ke CSV
summary = {
    "Metode": ["Duplicate", "Null", "Outlier", "Encoding"],
    "Hasil": [
        str(df.duplicated().sum()),
        str(df.isnull().sum().sum()),
        "Ada pada kolom numerik tertentu (cek IQR)",
        "Bila digunakan untuk model ML, perlu one-hot/label encoding"
    ]
}

pd.DataFrame(summary).to_csv(base_dir / "ringkasan_eksplorasi_covid.csv", index=False)
print("\nFile ringkasan_eksplorasi_covid.csv telah dibuat.")