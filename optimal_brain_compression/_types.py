import torch
from pydantic import BaseModel, ConfigDict
from typing import Tuple

# Output of `_utils.find_optimal_scales_and_zeros`
# `scales` and `rounded_zeros` are tensors of shape `(n_rows,)`
class ScalesAndZeros(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scales:torch.Tensor
    rounded_zeros:torch.Tensor

# Output of `exact_obc._handle_zeros`
Weights_Hessian_Tuple = Tuple[torch.Tensor, torch.Tensor]

# Output of `exact_obc._get_inverse_hessian_and_mask_per_row`
Hessian_Mask_MinZeros_Tuple = Tuple[torch.Tensor, torch.Tensor, int]

# Output for quantized weights
class QuantizedMatrix(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    quantized_matrix:torch.Tensor
    scales:torch.Tensor
    rounded_zeros:torch.Tensor