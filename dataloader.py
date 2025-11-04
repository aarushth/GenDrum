import os
import torch
import torchaudio
from torch.utils.data import Dataset
from spec_loader import load_spec

def peak_normalize(waveform):
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform

def get_wav_files(directory):
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".wav")
    ]


class AudioDataset(Dataset):
    def __init__(self, dirs:list):
        self.labels = []
        self.file_list = []
        self.dirs = dirs
        i = 0
        for dir in dirs:
            files = get_wav_files(dir)
            self.file_list += files
            self.labels += [i] * len(files)
            i+=1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_list[idx], format=".wav")
        spec = load_spec(waveform, sample_rate)
        label = self.labels[idx]
        return spec, torch.tensor(label)