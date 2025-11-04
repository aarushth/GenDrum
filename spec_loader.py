import torch
import torchaudio.transforms as T
import torch.nn.functional as F
from torchvision.transforms import Resize
from model import AudioClassifier
import matplotlib.pyplot as plt


def show_spectrogram(spec: torch.Tensor, title="Mel Spectrogram"):
    # Remove channel dimension if present
    if spec.ndim == 3:
        spec = spec.mean(dim=0)

    # Convert to NumPy
    spec_np = spec.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_np, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.tight_layout()
    plt.show()

def peak_normalize(waveform: torch.Tensor):
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak 
    y = waveform[0]
    # plt.figure(figsize=(12, 4))
    # plt.plot(y.numpy())
    # plt.title("Waveform")
    # plt.xlabel("Sample")
    # plt.ylabel("Amplitude")
    # plt.show()
    return waveform

def load_model(model_path):
    model = AudioClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_spec(waveform:torch.Tensor, sample_rate, n_mels=40, n_fft=1024, hop_length=256):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = peak_normalize(waveform)
    spec = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=100,
        f_max=sample_rate/2
    )(waveform)
    # spec = T.Spectrogram(
    #         n_fft=n_fft,
    #         win_length=n_fft,
    #         hop_length=hop_length,
    #         power=2.0
    #     )(waveform)
    spec = Resize((256, 256))(spec)
    spec = 10 * torch.log10(spec + 1e-10)
    # spec = (spec - spec.mean()) / spec.std()
    # show_spectrogram(spec, title="Spectrogram")
    return spec




