import torch
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version detected by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}") # Nom du GPU 0
    print(f"Current CUDA device index: {torch.cuda.current_device()}")
else:
    print("CUDA is NOT available to PyTorch.")


print("PyTorch TEST___________________________")



print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0))