import torch

x1 = torch.tensor(0.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)
f = torch.exp(x1) * (x1**2 + x2**2)**3
f.backward()

gradient = [x1.grad.item(), x2.grad.item()]
print("Gradient computed by PyTorch:", gradient)
