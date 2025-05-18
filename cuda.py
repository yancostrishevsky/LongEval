import torch

print("Torch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available!")
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
else:
    print("CUDA is NOT available. Check your installation.")
