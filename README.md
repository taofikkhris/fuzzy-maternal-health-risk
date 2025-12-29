# Maternal Health Risk Prediction - Fuzzy Logic System

Sistem pakar berbasis **Logika Fuzzy (Mamdani Inference)** untuk memprediksi tingkat risiko kesehatan ibu hamil berdasarkan parameter fisiologis.

## 📋 Ikhtisar
Proyek ini menggunakan algoritma Fuzzy Logic untuk mengklasifikasikan risiko kesehatan menjadi tiga kategori: **Low Risk**, **Mid Risk**, dan **High Risk**. Berbeda dengan Machine Learning tradisional, sistem ini bekerja menggunakan aturan "IF-THEN" yang dapat dipahami manusia, sehingga hasil keputusannya dapat dijelaskan secara medis.

### Performa Model
- **Akurasi**: ~63.31%
- **Recall (High Risk)**: 85% (Sangat sensitif mendeteksi kasus berisiko tinggi)
- **Metode**: Mamdani Fuzzy Inference System menggunakan fungsi keanggotaan trapesium dan segitiga.

## 🛠️ Persyaratan Sistem
Pastikan Anda sudah menginstal Python 3.x dan pustaka berikut:
- `numpy`
- `pandas`
- `scikit-fuzzy`
- `scikit-learn`

Instalasi melalui pip:
```bash
pip install numpy pandas scikit-fuzzy scikit-learn
```

## 🚀 Cara Penggunaan
1. Pastikan file `Maternal Health Risk Data Set.csv` berada di folder yang sama dengan skrip.
2. Jalankan program utama:
```bash
python fuzzy_maternal_health_risk.py
```

### Fitur Program:
- **Batch Prediction**: Program otomatis menghitung akurasi terhadap seluruh dataset saat pertama kali dijalankan.
- **Interactive Mode**: Anda dapat memasukkan data pasien secara manual (Umur, Tekanan Darah, Gula Darah, dll) untuk mendapatkan prediksi instan.

## 📊 Parameter Input
Program menerima 6 parameter utama:
1. **Age**: Umur pasien (tahun).
2. **SystolicBP**: Tekanan darah sistolik (mmHg).
3. **DiastolicBP**: Tekanan darah diastolik (mmHg).
4. **BS (Blood Sugar)**: Kadar gula darah (mmol/L).
5. **BodyTemp**: Suhu tubuh (Fahrenheit).
6. **HeartRate**: Detak jantung (bpm) - *Opsional (saat ini dinonaktifkan di aturan default).*

## 📁 Struktur Proyek
- `fuzzy_maternal_health_risk.py`: Skrip utama yang berisi logika fuzzy dan antarmuka pengguna.
- `Maternal Health Risk Data Set.csv`: Dataset sumber untuk pengujian akurasi.
- `system_explanation.md`: Penjelasan mendalam mengenai proses perhitungan matematis (Fuzzifikasi hingga Defuzzifikasi).

## 💡 Penjelasan Singkat Metode
Sistem ini memproses data melalui empat tahap:
1. **Fuzzifikasi**: Merubah input angka menjadi derajat keanggotaan fuzzy (0-1).
2. **Inferensi**: Menjalankan aturan logika IF-THEN (contoh: *JIKA Gula Darah Tinggi MAKA Risiko Tinggi*).
3. **Agregasi**: Menggabungkan hasil dari semua aturan yang aktif.
4. **Defuzzifikasi**: Menggunakan metode **Centroid** (titik berat) untuk merubah hasil fuzzy kembali menjadi angka tunggal (Skor Risiko).

---
*Dibuat untuk tujuan edukasi dan pengembangan sistem pakar kesehatan.*
