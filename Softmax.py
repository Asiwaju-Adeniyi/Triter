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

  @triton.jit 

  def _softmax_fwd_kernel(
      output_ptr,
      stride_output_row, 
      input_ptr,
      stride_input_row,
      num_cols,
      block_size: tl.constexpr,
  ): 
  #setup input ptrs
    row_index = tl.program_id(0) 
    row_start_ptr = input_ptr + (row_index * stride_input_row)
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols
    #move to SRAM

    row = tl.load(input_pointers,mask = row_mask, other = float("-inf"))

    #softmax itself
    safe_row = row - row_max  
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator 

    #write back to HBM 

    output_row_ptr = output_ptr + (row_index * strdie_output_row)
    output_pointers = output_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)




  def softmax(x:torch.Tensor)->torch.Tensor: 
    """ Triton impl of Softmax, fwd pass only """

     rows, cols = x.shape 
     assert x.dim() ==2, f"only accepts 2D tensors for now"
     block_size = triton.next_power_of_2(cols)
     num_warps = 4  # *32
     if block_size > 2047: #2048
          num_warps = 8
    if block_size > 4095: #4096
          num_warps = 16
    
    grid = (rows, )

    #allocate out output buffer 
    sm_out = torch.empty_like(x)


     _softmax_fwd_kernel[grid] (
         sm_out, 
         sm_out.stride(0), 
         x, 
         x.stride(0), 
         cols, 
         block_size=block_size,
         num_warps, =num_warps

     )

     return sm_out


union = torch.tensor([[12,34,35,35,36,70], [1,23,46,90,43,35]], dtype=torch.float32, device = 'cuda')
sam = F.softmax(union, dim = 1)
print(F"{sam}")


eager_out = naive_softmax(sam)
print(f"{eager_out=}")
