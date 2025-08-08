# hello_h100.py
import torch

print("Hello from the H100 test script!")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is NOT available.")

print("Python script ran successfully!")
