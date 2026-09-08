import torch

p=torch.FloatTensor(1,5,3)
print(p)
non_zeros = torch.sum(p, dim=2).ne(0)
print(non_zeros)
p = p[non_zeros].unsqueeze(dim=0)
print(p)