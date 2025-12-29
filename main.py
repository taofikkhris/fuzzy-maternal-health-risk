import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- KELAS SISTEM FUZZY ---

class MaternalRiskSystem:
    def __init__(self, params=None):
        # Parameter Default (telah dioptimalkan secara manual)
        self.params = {
            'low_bs_end': 7.2,
            'normal_bs_start': 6.8, 'normal_bs_end': 8.1,
            'high_bs_start': 7.9,
            'systolic_cutoff': 130,
            'age_mid_start': 35,
            'age_old_start': 40
        }
        if params:
            self.params.update(params)

        # Antecedents (Input)
        # Mendefinisikan variabel input dan rentang nilainya (Universe of Discourse)
        self.age = ctrl.Antecedent(np.arange(10, 71, 1), 'Age')          # Umur
        self.systolic_bp = ctrl.Antecedent(np.arange(60, 161, 1), 'SystolicBP') # Tekanan Darah Sistolik
        self.diastolic_bp = ctrl.Antecedent(np.arange(49, 101, 1), 'DiastolicBP') # Tekanan Darah Diastolik
        self.bs = ctrl.Antecedent(np.arange(6, 20, 0.1), 'BS')           # Gula Darah (Blood Sugar)
        self.body_temp = ctrl.Antecedent(np.arange(98, 104, 0.1), 'BodyTemp') # Suhu Tubuh
        self.heart_rate = ctrl.Antecedent(np.arange(7, 91, 1), 'HeartRate')   # Detak Jantung

        # Consequent (Output)
        # Mendefinisikan variabel output (Tingkat Risiko)
        self.risk_level = ctrl.Consequent(np.arange(0, 11, 1), 'RiskLevel')

        self.__define_membership_functions()
        self.__define_rules()
        
        # Simulasi
        # Membangun sistem kontrol fuzzy berdasarkan aturan yang dibuat
        self.risk_ctrl = ctrl.ControlSystem(self.rules)
        self.risk_sim = ctrl.ControlSystemSimulation(self.risk_ctrl)

    def __define_membership_functions(self):
        # Menggunakan Fungsi Keanggotaan yang diselaraskan dengan Decision Tree
        
        p = self.params
        
        # Umur (Age)
        # young: muda, mid: paruh baya, old: tua
        self.age['young'] = fuzz.trapmf(self.age.universe, [0, 0, 19, 30])
        self.age['mid'] = fuzz.trimf(self.age.universe, [20, 35, 50])
        self.age['old'] = fuzz.trapmf(self.age.universe, [p['age_old_start'], 55, 100, 100])

        # Tekanan Darah Sistolik (SystolicBP)
        # Batas tegas Decision Tree sekitar 130 mmHg
        self.systolic_bp['low'] = fuzz.trapmf(self.systolic_bp.universe, [0, 0, 90, 100]) 
        self.systolic_bp['normal'] = fuzz.trapmf(self.systolic_bp.universe, [95, 100, 129, 131]) 
        self.systolic_bp['high'] = fuzz.trapmf(self.systolic_bp.universe, [129, 131, 200, 200])

        # Tekanan Darah Diastolik (DiastolicBP)
        # Normal di bawah 90
        self.diastolic_bp['low'] = fuzz.trapmf(self.diastolic_bp.universe, [0, 0, 60, 70])
        self.diastolic_bp['normal'] = fuzz.trapmf(self.diastolic_bp.universe, [65, 70, 89, 91])
        self.diastolic_bp['high'] = fuzz.trapmf(self.diastolic_bp.universe, [89, 91, 200, 200])

        # Gula Darah (BS)
        # Batas Decision Tree: 8.0
        # Low: Rendah, Normal: Normal, High: Tinggi
        self.bs['low'] = fuzz.trapmf(self.bs.universe, [0, 0, 6.8, p['low_bs_end']]) 
        self.bs['normal'] = fuzz.trapmf(self.bs.universe, [p['normal_bs_start'], 7.2, 7.9, p['normal_bs_end']]) 
        self.bs['high'] = fuzz.trapmf(self.bs.universe, [p['high_bs_start'], 8.1, 50.0, 50.0])

        # Suhu Tubuh (BodyTemp)
        self.body_temp['normal'] = fuzz.trapmf(self.body_temp.universe, [0, 0, 98.4, 99.5])
        self.body_temp['high'] = fuzz.trapmf(self.body_temp.universe, [99, 100, 200, 200])

        # Detak Jantung (HeartRate)
        self.heart_rate['low'] = fuzz.trapmf(self.heart_rate.universe, [0, 0, 60, 75])
        self.heart_rate['normal'] = fuzz.trimf(self.heart_rate.universe, [65, 75, 90])
        self.heart_rate['high'] = fuzz.trapmf(self.heart_rate.universe, [85, 95, 200, 200])

        # Output Tingkat Risiko (RiskLevel)
        self.risk_level['low'] = fuzz.trimf(self.risk_level.universe, [0, 2, 4])
        self.risk_level['mid'] = fuzz.trimf(self.risk_level.universe, [3, 5, 7])
        self.risk_level['high'] = fuzz.trimf(self.risk_level.universe, [6, 8, 10])

    def __define_rules(self):

        # --- Faktor Risiko Tinggi ---
        # 1. Gula Darah Tinggi -> Risiko Tinggi
        r1 = ctrl.Rule(self.bs['high'], self.risk_level['high'])
        
        # 2. Tekanan Darah Tinggi -> Risiko Tinggi
        r2a = ctrl.Rule(self.systolic_bp['high'], self.risk_level['high'])
        r2b = ctrl.Rule(self.diastolic_bp['high'], self.risk_level['high'])
        
        # --- Faktor Risiko Rendah ---
        # 3. Gula Darah Normal/Rendah DAN Tensi Normal/Rendah -> Risiko Rendah
        # Menggabungkan Low dan Normal BS menjadi satu konsep "Tidak Tinggi"
        r3 = ctrl.Rule((self.bs['low'] | self.bs['normal']) & 
                       (self.systolic_bp['low'] | self.systolic_bp['normal']) & 
                       (self.diastolic_bp['low'] | self.diastolic_bp['normal']), 
                       self.risk_level['low'])

        # --- Nuansa Risiko Menengah (Mid Risk) ---
        # 4. Gula Darah Rendah TAPI Tensi Tinggi?
        # Aturan khusus untuk menurunkan Risiko Tinggi ke Menengah jika Gula Darah sangat rendah.
        r4 = ctrl.Rule(self.bs['low'] & self.systolic_bp['high'], self.risk_level['mid'])
        
        # 5. Efek Suhu pada kelompok Risiko Rendah
        # Jika metrik lain aman tapi Suhu Tinggi -> Naik jadi Mid Risk
        r5 = ctrl.Rule((self.bs['low'] | self.bs['normal']) & 
                       (self.systolic_bp['low'] | self.systolic_bp['normal']) & 
                       self.body_temp['high'], 
                       self.risk_level['mid'])

        # 6. Efek Umur
        # Umur Tua + Gula/Tensi Normal -> Cenderung Mid Risk
        r6 = ctrl.Rule(self.age['old'] & self.bs['normal'] & self.systolic_bp['normal'], self.risk_level['mid'])

        self.rules = [r1, r2a, r2b, r3, r4, r5, r6]

    def predict(self, data):
        """
        Melakukan prediksi berdasarkan data input dictionary.
        data: dict dengan keys yang cocok dengan nama antecedent (Age, SystolicBP, dll)
        """
        try:
            # Memasukkan nilai input ke sistem simulasi
            self.risk_sim.input['Age'] = float(data['Age'])
            self.risk_sim.input['SystolicBP'] = float(data['SystolicBP'])
            self.risk_sim.input['DiastolicBP'] = float(data['DiastolicBP'])
            self.risk_sim.input['BS'] = float(data['BS'])
            self.risk_sim.input['BodyTemp'] = float(data['BodyTemp'])
            # self.risk_sim.input['HeartRate'] = float(data['HeartRate']) # Tidak digunakan dalam aturan sederhana
            # Melakukan perhitungan fuzzy (Fuzzifikasi -> Inferensi -> Defuzzifikasi)
            self.risk_sim.compute()
            
            # Mengambil hasil skor risiko (angka crisp)
            risk_score = self.risk_sim.output['RiskLevel']
            
            # Melabeli skor menjadi kategori teks
            if risk_score <= 3.5:
                label = "low risk"
            elif risk_score <= 6.5:
                label = "mid risk"
            else:
                label = "high risk"
                
            return risk_score, label
            
        except Exception as e:
            # Fallback jika terjadi error (misalnya nilai di luar jangkauan)
            # print(f"Error Prediksi: {e}")
            return -1, "unknown"

# --- EKSEKUSI UTAMA ---

def main():
    print("=== Sistem Prediksi Risiko Kesehatan Ibu (Fuzzy Logic) ===")
    
    # Memuat Dataset
    try:
        df = pd.read_csv('Maternal Health Risk Data Set.csv')
        print(f"Dataset berhasil dimuat: {len(df)} data.")
    except FileNotFoundError:
        print("Error: File 'Maternal Health Risk Data Set.csv' tidak ditemukan.")
        return

    # Inisialisasi Sistem Fuzzy
    print("Menginisialisasi Model Fuzzy...")
    fuzzy_system = MaternalRiskSystem()

    # Prediksi Batch
    print("Menjalankan prediksi pada dataset...")
    predicted_labels = []
    
    unknown_count = 0

    for index, row in df.iterrows():
        input_data = {
            'Age': row['Age'],
            'SystolicBP': row['SystolicBP'],
            'DiastolicBP': row['DiastolicBP'],
            'BS': row['BS'],
            'BodyTemp': row['BodyTemp'],
            'HeartRate': row['HeartRate']
        }
        
        _, label = fuzzy_system.predict(input_data)
        
        if label == "unknown":
            unknown_count += 1
            predicted_labels.append("mid risk") # Default fallback
        else:
            predicted_labels.append(label)

    # Evaluasi Hasil
    actual_labels = df['RiskLevel'].str.lower()
    accuracy = accuracy_score(actual_labels, predicted_labels)
    
    print("\n--- Hasil Performa Model ---")
    print(f"Total Sampel: {len(df)}")
    print(f"Gagal Prediksi: {unknown_count}")
    print(f"Akurasi: {accuracy * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(actual_labels, predicted_labels, labels=["low risk", "mid risk", "high risk"]))
    print("\nClassification Report:")
    print(classification_report(actual_labels, predicted_labels, labels=["low risk", "mid risk", "high risk"]))

    # Mode Interaktif
    print("\n--- Mode Prediksi Manual ---")
    print("Masukkan data pasien untuk memprediksi risiko (atau ketik 'n' untuk keluar).")
    
    while True:
        choice = input("\nPrediksi sekarang? (y/n): ")
        if choice.lower() in ['n', 'exit', 'keluar']:
            break
            
        try:
            print("--- Input Data Pasien ---")
            age = float(input("Umur (tahun): "))
            sys_bp = float(input("Tekanan Darah Sistolik (mmHg): "))
            dia_bp = float(input("Tekanan Darah Diastolik (mmHg): "))
            bs = float(input("Gula Darah (mmol/L): "))
            temp = float(input("Suhu Tubuh (F): "))
            hr = float(input("Detak Jantung (bpm): "))
            
            data = {
                'Age': age, 'SystolicBP': sys_bp, 'DiastolicBP': dia_bp,
                'BS': bs, 'BodyTemp': temp, 'HeartRate': hr
            }
            
            score, label = fuzzy_system.predict(data)
            print("--------------------------")
            print(f"HASIL: {label.upper()}")
            print(f"Skor Risiko: {score:.2f} (0-10)")
            print("--------------------------")
            
        except ValueError:
            print("Input tidak valid. Harap masukkan angka.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
