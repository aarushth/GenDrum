from librosa import load, onset, samples_to_time
from torchaudio import load as torch_load
import numpy as np
from spec_loader import load_spec, load_model
from torch import argmax, softmax
from pydub import AudioSegment

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


filename = "input/demo.wav"

y, sr = load(filename, sr=None, mono=True)
chunks = onset_samples = onset_samples = onset.onset_detect(
    y=y, sr=sr, hop_length=64, backtrack=True, units='samples'
)
chunks_time = samples_to_time(onset_samples, sr=sr)

max_chunk_duration = 0.25  # seconds
max_chunk_size = int(max_chunk_duration * sr)


waveform, sample_rate = torch_load(filename, format="wav")

duration_ms = (waveform.shape[1] / sample_rate) * 1000

chunks = np.append(chunks, waveform.shape[-1])


model = load_model("models/audio_classifier_v13.pth")
classes = {0 : "Clap", 1: "Snap", 2:"Hit"}

shift=round(sr/100)

classified_chunks = []
for t in range(len(chunks)-1):
    start = max(chunks[t]-shift, 0)
    end = chunks[t+1]-shift
    if end - start > max_chunk_size:
        end = start + max_chunk_size

    chunk = waveform[:, start:end]

    spec = load_spec(chunk, sample_rate).unsqueeze(0)

    output = model(spec)
    prediction = argmax(output, dim=1)
    probabilities = softmax(output, dim=1)
    predicted_prob = probabilities[0][prediction].item()
    classified_chunks.append((prediction.item(), chunks_time[t]*1000))
    # print(f"{chunks_time[t]},{chunks[t]} Predicted class: {classes[prediction.item()]}, Probability: {predicted_prob}")


track = AudioSegment.silent(duration=duration_ms)

# your samples
kick = AudioSegment.from_file("samples/kick.wav")
snare = AudioSegment.from_file("samples/snare.wav")
hat = AudioSegment.from_file("samples/hat.wav")
samples = [snare, hat, kick]


for chunk in classified_chunks:
    track = track.overlay(samples[chunk[0]], chunk[1])

track.export(f"output/beat.wav", format="wav")