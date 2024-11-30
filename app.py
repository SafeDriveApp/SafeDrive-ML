from flask import Flask, render_template, request, jsonify
import cv2
import os

app = Flask(__name__, template_folder='templates')

# Define paths
SAVE_PATH = 'saved_videos'
FRAMES_PATH = 'frames'

# Ensure directories exist
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(FRAMES_PATH, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame file provided'}), 400

    frame_file = request.files['frame']
    frame_path = os.path.join(FRAMES_PATH, frame_file.filename)
    frame_file.save(frame_path)
    
    return jsonify({'message': 'Frame uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)