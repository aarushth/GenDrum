import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# === Load audio ===
filename = "new.wav"  # change to your file
y, sr = librosa.load(filename, sr=None, mono=True)

# === Detect onsets ===
onset_samples = onset_samples = librosa.onset.onset_detect(
    y=y, sr=sr, hop_length=128, backtrack=True, units='samples'
)
onset_times = librosa.samples_to_time(onset_samples, sr=sr)
print(onset_times)
print(f"Detected {len(onset_samples)} onsets")

# === Plot waveform ===
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, color='steelblue')
plt.title("Waveform with Onset Markers")

# Mark onsets
for onset_time in onset_times:
    plt.axvline(onset_time, color='red', linestyle='--', alpha=0.8)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("onset_waveform.png", dpi=300)
plt.show()
