import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
device = (
"cuda" if torch.cuda.is_available()
else "mps" if torch.backends.mps.is_available()
else "cpu"
)
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device=device)
y = x @ x
print(f"Selected device: {device}")
print(f"Tensor shape: {x.shape}")
print(y)