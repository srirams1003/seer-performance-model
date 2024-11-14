import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the GPU
    print("Using CUDA!")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")

# Create tensors on the GPU (if available)
x = torch.randn(2, 3, device=device)
y = torch.randn(3, 5, device=device)

# Perform a matrix multiplication using cuDNN (if available)
z = torch.matmul(x, y)

print(z)
