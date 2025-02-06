from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa
import subprocess as sp
import torchaudio
import torchvision
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from ipyfilechooser import FileChooser
import pretty_midi
from scipy.io.wavfile import write
import numpy as np
import os
import requests
from io import BytesIO

from drums import drum_labels, DrumCNN, DrumDataset
from drums import load_model, separate_stems, onset_times, load_dataset, run_model, transcriptions_to_midi_and_audio, run_midi



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
