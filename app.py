from flask import Flask, render_template, request, jsonify
import os
import cv2
from prediction import predict  # Mengimpor fungsi predict dari prediction.py

app = Flask(__name__, template_folder='templates')

# Paths untuk menyimpan video dan frame
SAVE_PATH = 'saved_videos'
FRAMES_PATH = 'frames'

# Pastikan direktori ada
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(FRAMES_PATH, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame file provided'}), 400

    frame_file = request.files['frame']
    frame_path = os.path.join(FRAMES_PATH, frame_file.filename)
    frame_file.save(frame_path)

    # Membaca gambar dari file
    image = cv2.imread(frame_path)

    # Melakukan prediksi dengan fungsi dari prediction.py
    prediction = predict(image)

    # Mengirimkan hasil prediksi kembali ke frontend
    if prediction == 0:
        return jsonify({'prediction': '0'}), 200  # Kirimkan prediction
    else:
        return jsonify({'prediction': '1'}), 200  # Kirimkan prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
