from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload-wav', methods=['POST'])
def upload_wav():
    """Receive a .wav file, save it, process it, and return the processed file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Process the audio and get the output file path
        output_file_path = process_audio(file_path)  # Assuming this returns the path to the processed file

        # Return the processed file directly in the response
        return send_file(
            output_file_path,
            mimetype='audio/wav',  # Set the MIME type for .wav files
            as_attachment=True,    # Prompt the browser to download the file
            download_name='processed_output.wav'  # Suggested file name for download
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5050)