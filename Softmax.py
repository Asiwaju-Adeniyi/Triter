import torch 
import triton
import triton.language as tl
import torch.nn.functional as F


union = torch.tensor([[12,34,35,35,36,70], [1,23,46,90,43,35]], dtype=torch.float32, device = 'cuda')
sam = F.softmax(union, dim = 1)
print(F"{sam}")
