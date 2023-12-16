import torch
from pydantic import BaseModel
from typing import Tuple

# Output of `_utils.find_optimal_scales_and_zeros`
# `scales` and `rounded_zeros` are tensors of shape `(n_rows,)`
class ScalesAndZeros(BaseModel):
    scales:torch.Tensor
    rounded_zeros:torch.Tensor

# Output of `exact_obc._handle_zeros`
Weights_Hessian_Tuple = Tuple[torch.Tensor, torch.Tensor]

# Output of `exact_obc._get_inverse_hessian_and_mask_per_row`
Hessian_Mask_MinZeros_Tuple = Tuple[torch.Tensor, torch.Tensor, int]

# Output for quantized weights
class QuantizedMatrix(BaseModel):
    quantized_matrix:torch.Tensor
    scales:torch.Tensor
    rounded_zeros:torch.Tensor