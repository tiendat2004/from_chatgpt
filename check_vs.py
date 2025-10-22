import torch
print(torch.cuda.is_available())   # True nếu đã nhận GPU
print(torch.cuda.get_device_name(0))  # Tên GPU, ví dụ 'NVIDIA GeForce RTX 3050'
print(torch.__version__)  # Phiên bản PyTorch
print(torch.version.cuda)  # Phiên bản CUDA