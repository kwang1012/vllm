import torch

x = torch.Tensor([1])
x2 = torch.Tensor([2])
x3 = torch.Tensor([3])
out = torch.cat((x, x2))
print(out)
out = torch.cat((out, x3))
print(out)
