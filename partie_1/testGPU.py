import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version detected by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}") # Nom du GPU 0
    print(f"Current CUDA device index: {torch.cuda.current_device()}")
else:
    print("CUDA is NOT available to PyTorch.")

    # test_alb.py
    import albumentations as A
    import numpy as np  # Just for dummy data if needed

    print(f"Using Albumentations version: {A.__version__}")  # Verify version again

    try:
        transform = A.Compose(
            [A.HorizontalFlip(p=0.5)],  # Une transformation simple
            polygon_params={'format': 'coco'}  # L'argument qui pose problème
        )
        print("Minimal A.Compose with polygon_params SUCCEEDED!")

    except TypeError as e:
        print(f"Minimal test FAILED with TypeError: {e}")
    except Exception as e:
        print(f"Minimal test FAILED with other Exception: {e}")


# test_alb.py
import albumentations as A
import numpy as np # Just for dummy data if needed

print(f"Using Albumentations version: {A.__version__}") # Verify version again

try:
    transform = A.Compose(
        [A.HorizontalFlip(p=0.5)], # Une transformation simple
        polygon_params={'format': 'coco'} # L'argument qui pose problème
    )
    print("Minimal A.Compose with polygon_params SUCCEEDED!")

except TypeError as e:
    print(f"Minimal test FAILED with TypeError: {e}")
except Exception as e:
    print(f"Minimal test FAILED with other Exception: {e}")