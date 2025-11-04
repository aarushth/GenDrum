import os
import librosa
import torchaudio
from spec_loader import load_spec

def get_wav_files(directory):
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".wav")
    ]

for filename in get_wav_files("nonaugmentdata/hits"):
    y, sr = librosa.load(filename, sr=None, mono=True)
    waveform, sample_rate = torchaudio.load(filename, format="wav")
    spec = load_spec(waveform, sample_rate)