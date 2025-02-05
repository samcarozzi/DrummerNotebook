from flask import Flask, request, jsonify
from flask_cors import CORS
import os

#flask implementation to ensure working
app = Flask(__name__)
CORS(app)  # Allow requests from Next.js

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload-wav', methods=['POST'])
def upload_wav():
    """Receive a .wav file and return success message"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the WAV file here (if needed)
    print(f"Received and saved: {file_path}")

    return jsonify({'message': 'Processing complete'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
