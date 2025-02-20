import os
import torch
import torchaudio
import torchvision
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
import requests
import pretty_midi
import subprocess as sp
from io import BytesIO
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from scipy.io.wavfile import write


drum_labels = ['kick', 'snare', 'hihat', 'tom', 'crash', 'ride']



class DrumCNN(nn.Module):
    def __init__(self):
        super(DrumCNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.4)
        
        self.fc3 = nn.Linear(256, len(drum_labels))
    
    def forward(self, x):
        # Convolutional Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    

class DrumDataset(Dataset):
    #def __init__(self, df, audio, transform, window_size=8192):
    def __init__(self, df, audio, transform, window_size=8192):
        self.df = df
        self.window_size = window_size
        self.transform = transform
        self.audio = audio
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load the onset time, label, and track name for the given index
        row = self.df.iloc[idx]
        onset_time = row['onset_time']
        labels = row[drum_labels].astype(int).values.flatten()
        labels = torch.tensor(labels).float()

        audio = self.audio[0]
        sr = self.audio[1]

        target_start_sample = int(onset_time * sr) - self.window_size // 2
        target_end_sample = int(onset_time * sr) + self.window_size // 2
        start_sample = max(0, target_start_sample)
        end_sample = min(audio.shape[-1], target_end_sample)

        onset_window = audio[start_sample:end_sample]
        onset_window = nn.functional.pad(onset_window, (start_sample - target_start_sample, target_end_sample - end_sample))

        spec = self.transform(onset_window)
        return spec, labels
    
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: x.to("cpu")),
    torchvision.transforms.Lambda(lambda x: torch.stack([
            #torchaudio.transforms.MelSpectrogram(
                #n_fft=1024,
                #hop_length=64,
                #n_mels=128
            #).to("cpu")(x),
            torchaudio.transforms.MelSpectrogram(
                n_fft=1024,
                    hop_length=64,
                n_mels=128,
                f_min=20,      # Lower minimum frequency
                f_max=6700     # Focus on percussive range
            ).to("cpu")(x),
    torchaudio.transforms.MFCC(
                n_mfcc=128,
                melkwargs={'n_fft': 1024, 'hop_length': 64, 'n_mels': 128}).to("cpu")(x)
            ], dim=0).to("cpu")),
    torchvision.transforms.Lambda(lambda x: torch.stack([
            torchaudio.transforms.AmplitudeToDB().to("cpu")(x[0]),
            x[0],
            x[1]
        ], dim=0).to("cpu")),
    torchvision.transforms.Lambda(lambda x: x.to("cpu"))
])

def load_model():
    print("load_model")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # URL of the raw .pth file on GitHub
    url = "https://github.com/skittree/DrummerScore/raw/master/notebooks/models/HeartsOnFire-v.1.0.4_nfft1024_91.58.pth"


    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Ensure we got the file

    # Load the model from memory (BytesIO)
    checkpoint = torch.load(BytesIO(response.content), map_location=torch.device('cpu'))

    # Print model keys (optional)
    print(checkpoint.keys())

    model = DrumCNN().to(device)
    model.load_state_dict(checkpoint['model'])
    return model

def separate_stems(file_path):
    
    # Extract the parent folder name (e.g., "baiana")
    base_dir = os.path.splitext(os.path.basename(file_path))[0]
    print(base_dir)

    # Construct the new path dynamically
    output_folder = os.path.join("drums")
    print("this is separate stems: " + output_folder)

    command = ["demucs", "-d", "cpu", "--mp3", "--mp3-bitrate", "320", "--two-stems=drums", file_path, "--out", output_folder]
    
    sp.run(command, check=True)


#fix
def onset_times_create(file_path):

    """

    # Extract the parent folder name (e.g., "baiana")
    base_dir = os.path.splitext(os.path.basename(file_path))[0]
    print(base_dir)


    parent_folder = os.path.dirname(os.path.dirname(file_path))
    print(parent_folder)
    
    
    # Construct the new path dynamically
    demucs_path = os.path.join(parent_folder, "drums","htdemucs", base_dir, "drums.mp3")

    print(demucs_path)

    """

    demucs_path = file_path
   


    audio, sr = torchaudio.load(demucs_path, format="mp3")
    if audio.shape[0] == 2:
        audio = torch.mean(audio, dim=0, keepdim=False)

    y = audio.numpy()
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=1024)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    onset_times = pd.DataFrame(librosa.frames_to_time(onset_frames, sr=sr), columns=['onset_time'])
    onset_times['onset_time'] = onset_times['onset_time'].apply(
        lambda x: float(x) if isinstance(x, (np.ndarray, list)) else x
    )
    onset_times['onset_time'] = onset_times['onset_time'].astype(float)

    onset_times[drum_labels] = False
    # onset_times = onset_times.iloc[1:]
    # onset_times = onset_times.reset_index(drop=True)
    return (onset_times, audio, sr)
    #print(onset_times.iloc[len(onset_times)//2 - 5 : len(onset_times)//2 + 5])
    # Check if any value is True in the drum_labels columns
    #print(onset_times[drum_labels].any().any())

def load_dataset(onset_times, audio, sr):
    #pred_dataset = DrumDataset(onset_times, (audio, sr), transforms)

    pred_dataset = DrumDataset(onset_times, (audio, sr), transforms)  # Reduced

    pred_loader = DataLoader(pred_dataset, batch_size=16, shuffle=False)  
    return pred_loader  

def run_model(model, onset_times, pred_loader):
    # Ensure CUDA is disabled and set the device to CPU
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    predicted_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(pred_loader, total=len(pred_loader), unit='batch', desc="Labeling")):
            inputs = inputs.to(device)  # Ensure inputs are on CPU
            outputs = model(inputs).cpu()  # Model outputs on CPU
            #predicted_labels.extend((outputs > 0.0).numpy().tolist())  # Threshold and append predictions

            # Apply a lower threshold for toms to make them more likely to be detected
            thresholded_outputs = (outputs > 0.5).numpy().astype(int)
            
            # If tom's index is known (assuming 3rd position in drum_labels)
            tom_index = drum_labels.index("tom")
            thresholded_outputs[:, tom_index] = (outputs[:, tom_index] > 0.3).numpy().astype(int)  # Lowered for toms
            
            predicted_labels.extend(thresholded_outputs.tolist())

    # Ensure the number of predicted labels matches the DataFrame rows
    if len(predicted_labels) != len(onset_times):
        print(f"Length of predicted_labels: {len(predicted_labels)}")
        print(f"Length of onset_times: {len(onset_times)}")
        raise ValueError("The number of predicted labels does not match the number of rows in 'onset_times'!")

    # Assign predicted labels to onset_times DataFrame
    onset_times[drum_labels] = predicted_labels
    return onset_times

def transcriptions_to_midi_and_audio(file_path, onset_times, tempo=120):
    """
    Converts drum transcriptions to MIDI and synthesizes audio using pyFluidSynth with a SoundFont.
    The output directory and file paths are automatically determined from `file_path`.

    Parameters:
        onset_times (pd.DataFrame): DataFrame containing onset times and drum labels.
        soundfont_path (str): Path to the SoundFont file (.sf2).
        tempo (int): Tempo for the MIDI file (default: 120 BPM).
    """
    

    """
    # Extract the base name of the file (without extension)
    file_stem = os.path.splitext(os.path.basename(file_path))[0]  # "sambeat"

    # Construct the output folder structure dynamically
    parent_folder = os.path.dirname(os.path.dirname(file_path))  # Moves up one directory
    output_folder = os.path.join(parent_folder, "drums", "htdemucs", file_stem)
    print(output_folder) """

    output_folder = file_path

    # Initialize PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Create drum instrument
    drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)

    # Add drum notes from onset_times DataFrame
    for index, row in onset_times.iterrows():
        onset_time = float(row['onset_time'])
        if row['kick']:
            drum_instrument.notes.append(pretty_midi.Note(
                velocity=100, pitch=36, start=onset_time, end=onset_time + 0.1))
        if row['snare']:
            drum_instrument.notes.append(pretty_midi.Note(
                velocity=100, pitch=38, start=onset_time, end=onset_time + 0.1))
        if row['hihat']:
            drum_instrument.notes.append(pretty_midi.Note(
                velocity=100, pitch=42, start=onset_time, end=onset_time + 0.1))
        if row['tom']:
            drum_instrument.notes.append(pretty_midi.Note(
                velocity=100, pitch=48, start=onset_time, end=onset_time + 0.1))
        if row['crash']:
            drum_instrument.notes.append(pretty_midi.Note(
                velocity=100, pitch=49, start=onset_time, end=onset_time + 0.1))
        if row['ride']:
            drum_instrument.notes.append(pretty_midi.Note(
                velocity=100, pitch=51, start=onset_time, end=onset_time + 0.1))

    # Add drum instrument to MIDI object
    midi.instruments.append(drum_instrument)

    # Construct the correct MIDI file path
    midi_file_path = os.path.join(output_folder, "drums.mid")

    # Save MIDI file
    midi.write(midi_file_path)
    print(f"✅ MIDI file written to: {midi_file_path}")

def process_audio(file_path):
    print("🔹 Loading Model...")
    model = load_model()

    print("🔹 Separating Stems...")
    #separate_stems(file_path)

    print("🔹 Extracting Onset Times...")
    (onset_times, audio, sr) = onset_times_create("/Users/samcarozzi/Documents/GitHub/HIL-Backend/flask/drums/htdemucs/ghostbusters/drums.mp3")

    print("🔹 Preparing Dataset...")
    pred_loader = load_dataset(onset_times, audio, sr)

    print("🔹 Running Model...")
    run_model(model, onset_times, pred_loader)

    print("🔹 Generating MIDI...")
    transcriptions_to_midi_and_audio("/Users/samcarozzi/Documents/GitHub/HIL-Backend/flask/drums/htdemucs/ghostbusters", onset_times, tempo=120)

    print("✅ Processing Complete!")



def main():
    print("main")
    
    ##/Users/samcarozzi/Documents/GitHub/HIL-Backend/drums/test/sambeat.mp3
    process_audio("/Users/samcarozzi/Documents/GitHub/HIL-Backend/drums/test/ghostbusters.wav")

if __name__ == "__main__":
    try:
        print("✅ main() started!")
        main()
        print("✅ main() finished successfully!")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    