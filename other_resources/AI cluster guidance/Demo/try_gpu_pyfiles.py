import torch
from time import sleep
# 设置是否使用CUDA（如果可用的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA is not available. Running on CPU...")

# 确保持续计算，以保持GPU占用
while True:
    a = torch.randn(1200, 1200, device=device)
    b = torch.randn(1200, 1200, device=device)
    c = torch.matmul(a, b)
    d = torch.matmul(a, b)
    sleep(0.01)