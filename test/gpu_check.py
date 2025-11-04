# !pip install torch torchvision torchaudio # Use GPU to run
import torch
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device) 
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device index:", torch.cuda.current_device())
else:
    print("GPU is not available.") 
x = torch.tensor(np.array([1.0, 2.0])).to('cuda' if torch.cuda.is_available else "cpu")