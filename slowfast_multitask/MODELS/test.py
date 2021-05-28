import torch

a = []
for _ in range(10):
    a.append(torch.rand((2,4)))
print(a)
print(torch.cat(a).shape)
print(torch.stack(a).shape)
