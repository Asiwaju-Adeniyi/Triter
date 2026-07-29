import torch 
import triton
import triton.language as tl
import torch.nn.functional as F

def naive_softmax(x: torch.Tensor)-> torch.Tensor: 
  """ eager mode softmax """

  x_max = x.max(dim=1)[0]
  safe_x = x - x_max[:, None]
  numerator = torch.exp(safe_x)
  denominator = numerator.sum(dim=1)
  sm_out = numerator/denominator[:, None]
  return sm_out

union = torch.tensor([[12,34,35,35,36,70], [1,23,46,90,43,35]], dtype=torch.float32, device = 'cuda')
sam = F.softmax(union, dim = 1)
print(F"{sam}")


eager_out = naive_softmax(sam)
print(f"{eager_out=}")
