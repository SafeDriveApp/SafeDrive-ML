import numpy as np
import cv2
import joblib
import dlib
from scipy.spatial import distance

# Muat model
model = joblib.load('model_random_forest.pkl')

# Muat detektor wajah dan prediktor landmarks
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat')  # Pastikan file predictor ada

# Fungsi untuk menghitung EAR
def calculate_ear(landmarks):
    A = distance.euclidean(landmarks[36], landmarks[41])
    B = distance.euclidean(landmarks[37], landmarks[40])
    C = distance.euclidean(landmarks[38], landmarks[39])
    ear = (A + B) / (2.0 * C)
    return ear

# Fungsi untuk menghitung MAR
def calculate_mar(landmarks):
    A = distance.euclidean(landmarks[48], landmarks[54])
    B = distance.euclidean(landmarks[49], landmarks[53])
    C = distance.euclidean(landmarks[50], landmarks[52])
    mar = (A + B + C) / 3.0
    return mar

# Fungsi untuk mengekstrak fitur dari gambar
def extract_features_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    ear_values = []
    mar_values = []
    nose_x_coords = []

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks_list = [(p.x, p.y) for p in landmarks.parts()]

        ear = calculate_ear(landmarks_list)
        mar = calculate_mar(landmarks_list)
        
        ear_values.append(ear)
        mar_values.append(mar)
        nose_x_coords.append(landmarks.part(30).x)

    if not ear_values:  # Jika tidak ada wajah terdeteksi
        return [0, 0, 0]  # Nilai default jika tidak ada wajah terdeteksi
    
    # Kembalikan hanya 3 fitur: EAR, MAR, dan nose_x
    return [np.mean(ear_values), np.mean(mar_values), np.mean(nose_x_coords)]


# Fungsi untuk melakukan prediksi
def predict(image):
    features = extract_features_from_image(image)
    
    # Cek apakah ada nilai NaN atau tidak lengkap pada fitur yang diekstraksi
    if np.any(np.isnan(features)) or len(features) != 3:
        return "Error: Fitur tidak lengkap atau mengandung nilai NaN"

    # Melakukan prediksi menggunakan model tanpa imputasi
    prediction = model.predict([features])
    
    return prediction
