import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load wav file
sample_rate, data = wavfile.read("new.wav")  # replace with your file

# If stereo, take only one channel
if len(data.shape) > 1:
    data = data[:, 0]

# Create time axis
times = np.arange(len(data)) / float(sample_rate)

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(times, data)
plt.title("Waveform of example.wav")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.xlim(0, times[-1])
plt.savefig("waveform.png", dpi=300)
plt.show()
