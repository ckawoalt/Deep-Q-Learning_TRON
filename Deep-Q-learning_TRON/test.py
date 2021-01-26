import torch

a=torch.tensor([0.25,0.25,0.25,0.25])
b=torch.tensor([[2,0,3],[4,8,0]])

print(a.multinomial(num_samples=1))


