import cv2
import dlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mediapipe as mp
from scipy.spatial import distance

# Inisialisasi detektor dlib dan model MediaPipe
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Fungsi untuk menghitung Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Fungsi untuk menghitung Mouth Aspect Ratio (MAR)
def calculate_mar(mouth_points):
    A = distance.euclidean(mouth_points[1], mouth_points[7])
    B = distance.euclidean(mouth_points[2], mouth_points[6])
    C = distance.euclidean(mouth_points[3], mouth_points[5])
    D = distance.euclidean(mouth_points[0], mouth_points[4])
    mar = (A + B + C) / (3.0 * D)
    return mar

# Fungsi untuk mendeteksi landmark wajah
def get_landmarks(gray, face):
    landmarks = predictor(gray, face)
    points = []
    for i in range(68):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    return points

# Pengaturan pengambilan gambar dari webcam
cap = cv2.VideoCapture(0)

# Variabel untuk menyimpan data EAR, MAR, dan koordinat hidung
ear_list, mar_list, nose_x_list, nose_y_list = [], [], [], []

# Thresholds dan counter untuk deteksi kantuk
ear_threshold = 0.25
mar_threshold = 0.3
counter = 0
data_samples = []
labels = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        
        for face in faces:
            # Deteksi landmark wajah
            landmarks = get_landmarks(gray, face)

            # EAR untuk mata kiri dan kanan
            left_eye_points = [landmarks[i] for i in [36, 37, 38, 39, 40, 41]]
            right_eye_points = [landmarks[i] for i in [42, 43, 44, 45, 46, 47]]
            ear = (calculate_ear(left_eye_points) + calculate_ear(right_eye_points)) / 2.0
            ear_list.append(ear)

            # MAR untuk mulut
            mouth_points = [landmarks[i] for i in [48, 50, 52, 54, 56, 58, 60, 64]]
            mar = calculate_mar(mouth_points)
            mar_list.append(mar)

            # Koordinat hidung
            nose_x, nose_y = landmarks[33][0], landmarks[33][1]
            nose_x_list.append(nose_x)
            nose_y_list.append(nose_y)

            # Simpan data setiap 15 pembacaan
            if len(ear_list) >= 15:
                data_vector = [
                    np.mean(ear_list), np.std(ear_list),
                    np.mean(mar_list), np.std(mar_list),
                    np.mean(nose_x_list), np.std(nose_x_list),
                    np.mean(nose_y_list), np.std(nose_y_list)
                ]
                data_samples.append(data_vector)
                label = 1 if ear < ear_threshold and mar > mar_threshold else 0  # 1 untuk kantuk, 0 untuk tidak kantuk
                labels.append(label)

                # Hapus data yang lama setelah disimpan
                ear_list, mar_list, nose_x_list, nose_y_list = [], [], [], []

                # Pelatihan model Random Forest jika data cukup
                if len(data_samples) >= 100:  # Misalnya, kumpulkan 100 sampel sebelum pelatihan
                    X_train, X_test, y_train, y_test = train_test_split(data_samples, labels, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    print("Akurasi model Random Forest:", accuracy)

                    # Prediksi kantuk
                    is_drowsy = model.predict([data_vector])[0]
                    if is_drowsy == 1:
                        print("Driver drowsy! Alarm 'Wake Up!'")
                    else:
                        print("Driver alert")

        # Tampilkan gambar dengan informasi kantuk
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
