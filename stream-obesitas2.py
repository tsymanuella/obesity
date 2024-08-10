import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

try:
    
    # Membaca model
    diabetes_model = pickle.load(open('obesitas_model.sav', 'rb'))

    # Membaca scaler
    scaler = pickle.load(open('StandardScaler.pkl', 'rb'))
    
    # Membaca label encoder
    with open('LabelEnc2.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    
except FileNotFoundError as e:
    st.error(f"FileNotFoundError: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Judul web
st.title("Prediksi Tingkat Obesitas")
col1, col2 = st.columns(2)

with col1:
    Age = st.text_input("Usia")
    if Age != '':
        Age = float(Age)  # Konversi ke float

with col2:
    Height = st.text_input("Tinggi Badan")
    if Height != '':
        Height = float(Height)  # Konversi ke float

with col1:
    Weight = st.text_input("Berat Badan")
    if Weight != '':
        Weight = float(Weight)  # Konversi ke float

# Dropdown input fields
Gender_input = st.selectbox("Pilih jenis kelamin:", ('Male', 'Female'))
CALC_input = st.selectbox("Seberapa Sering Mengkonsumsi Alkohol:", ('no', 'Sometimes', 'Frequently', 'Always'))
FAVC_input = st.selectbox("Apakah Anda Sering Mengkonsumsi Makanan Tinggi Kalori:", ('yes', 'no'))
SCC_input = st.selectbox("Apakah Anda Memantau Asupan Kalori:", ('yes', 'no'))
SMOKE_input = st.selectbox("Apakah Anda Merokok:", ('yes', 'no'))
FHO_input = st.selectbox("Apakah Anda Memiliki Anggota Keluarga yang Kelebihan Berat Badan:", ('yes', 'no'))
CAEC_input = st.selectbox("Seberapa Sering Anda Makan di Antara Makanan:", ('no', 'Sometimes', 'Frequently', 'Always'))
MTRANS_input = st.selectbox("Jenis Transportasi Apa yang Anda Gunakan:", ('Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'))

try:
    # Encoding categorical input fields using the loaded label encoders
    Gender_y = label_encoders['Gender'].transform([Gender_input])[0]
    CALC_y = label_encoders['CALC'].transform([CALC_input])[0]
    FAVC_y = label_encoders['FAVC'].transform([FAVC_input])[0]
    SCC_y = label_encoders['SCC'].transform([SCC_input])[0]
    SMOKE_y = label_encoders['SMOKE'].transform([SMOKE_input])[0]
    family_history_with_overweight_y = label_encoders['family_history_with_overweight'].transform([FHO_input])[0]
    CAEC_y = label_encoders['CAEC'].transform([CAEC_input])[0]
    MTRANS_y = label_encoders['MTRANS'].transform([MTRANS_input])[0]
except KeyError as e:
    st.error(f"KeyError: {e}")
except Exception as e:
    st.error(f"An error occurred during encoding: {e}")

# Other input fields
FCVC_input = st.selectbox("Seberapa Sering Mengkonsumsi Sayuran:", ('Tidak Pernah', 'Kadang-Kadang', 'Sering', 'Selalu'))
FCVC_mapping = {'Selalu': 3, 'Sering': 2, 'Kadang-Kadang': 1, 'Tidak Pernah': 0}
FCVC_y = FCVC_mapping[FCVC_input]

NCP_input = st.selectbox("Berapa Banyak Makan Utama yang Dikonsumsi Setiap Hari:", ('Tidak ada jawaban', 'Lebih dari 3', '3', 'antara 1&2'))
NCP_mapping = {'Tidak ada jawaban': 3, 'Lebih dari 3': 2, '3': 1, 'antara 1&2': 0}
NCP_y = NCP_mapping[NCP_input]

CH2O_input = st.selectbox("Berapa Banyak Air yang Dikonsumsi Setiap Hari:", ('Lebih dari 2 Liter', 'antara 1&2 Liter', 'kurang dari 1 Liter'))
CH2O_mapping = {'Lebih dari 2 Liter': 2, 'antara 1&2 Liter': 1, 'kurang dari 1 Liter': 0}
CH2O_y = CH2O_mapping[CH2O_input]

FAF_input = st.selectbox("Seberapa Sering Anda Melakukan Aktivitas Fisik:", ('4/5 kali seminggu', '2/3 kali seminggu', '1/2 kali seminggu', 'tidak pernah'))
FAF_mapping = {'4/5 kali seminggu': 3, '2/3 kali seminggu': 2, '1/2 kali seminggu': 1, 'tidak pernah': 0}
FAF_y = FAF_mapping[FAF_input]

TUE_input = st.selectbox("Berapa Lama Anda Menggunakan Perangkat Elektronik:", ('Lebih dari 3 jam', 'antara 1 dan 3 jam', 'kurang dari 1 jam', 'tidak ada'))
TUE_mapping = {'Lebih dari 3 jam': 3, 'antara 1 dan 3 jam': 2, 'kurang dari 1 jam': 1, 'tidak ada': 0}
TUE_y = TUE_mapping[TUE_input]

# Prediction code
Prediksi_Obesitas = ''
if st.button("Ayo Cek!"):
    if Age and Height and Weight:
        try:
            # Scaling numerical input features
            scaled_features = scaler.transform([[Age, Height, Weight]])
            
            # Combining all features into a single array
            features = np.array([
                scaled_features[0][0], scaled_features[0][1], scaled_features[0][2],
                Gender_y, CALC_y, FAVC_y, FCVC_y, NCP_y, SCC_y, SMOKE_y, CH2O_y, family_history_with_overweight_y, FAF_y, TUE_y, CAEC_y, MTRANS_y
            ]).reshape(1, -1)
            
            # Making prediction with Decision Tree
            Prediksi_Obesitas = diabetes_model.predict(features)
            
            # Interpreting the prediction result
            if Prediksi_Obesitas[0] == 0:
                Prediksi_Obesitas = "Insufficient Weight"
            elif Prediksi_Obesitas[0] == 1:
                Prediksi_Obesitas = "Normal Weight"
            elif Prediksi_Obesitas[0] == 2:
                Prediksi_Obesitas = "Overweight Level 1"
            elif Prediksi_Obesitas[0] == 3:
                Prediksi_Obesitas = "Overweight Level 2"
            elif Prediksi_Obesitas[0] == 4:
                Prediksi_Obesitas = "Overweight Level 3"   
            elif Prediksi_Obesitas[0] == 5:
                Prediksi_Obesitas = "Obesity Type 1" 
            elif Prediksi_Obesitas[0] == 6:
                Prediksi_Obesitas = "Obesity Type 2" 
            else:
                Prediksi_Obesitas = "tidak ditemukan jenis obesitas"
            
            st.success(Prediksi_Obesitas)
        except ValueError as e:
            st.error(f"ValueError: {e}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
