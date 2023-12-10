import torch
from _checks import check_if_square, check_if_ndim
from _utils import invert_spd_matrix, find_optimal_scales_and_zeros, get_top_n_nonzero_indices
from _types import Weights_Hessian_Tuple, QuantizedMatrix

def _handle_zeros(weights:torch.Tensor, hessian:torch.Tensor) -> Weights_Hessian_Tuple:
    '''
    - `weights` has shape `(n_rows, n_cols)`
    - `hessian` has shape `(n_cols, n_cols)`
    1. Find the diagonal entries of `hessian` which are zero.
    2. Set them to `1`.
    3. Set the corresponding columns in `weights` matrix to 0. This is because if the Hessian diagonal entries are zero, it means the inputs are always zero. Then we might as well set the weights to be zero.
    4. Return the modified `weights` and `hessian` matrices.
    '''
    check_if_ndim(weights, 2)
    check_if_ndim(hessian, 2)
    check_if_square(hessian)
    # TODO: introduce relative damping as an option
    # The hessian is calculated in high precision (fp64). We convert it back to fp32.
    hessian = hessian.float()
    indices_of_zero_diagonal = hessian.diag() == 0
    hessian[indices_of_zero_diagonal, indices_of_zero_diagonal] = 1
    weights[:, indices_of_zero_diagonal] = 0
    return weights, hessian

def _update_hessian_at_zeros_and_return_inverse(is_zero:torch.Tensor, hessian:torch.Tensor):
    '''
    Input:
    - `is_zero`: boolean tensor of shape `(n_cols,)`
    - `hessian`: tensor of shape `(n_cols, n_cols)`
    Output:
    - `inverse_hessian`: tensor of shape `(n_cols, n_cols)`
    '''
    copy_of_hessian = hessian.clone()
    copy_of_hessian[:, is_zero] = 0
    copy_of_hessian[is_zero, :] = 0
    copy_of_hessian[is_zero, is_zero] = 1
    return invert_spd_matrix(copy_of_hessian)

def _exact_obc_batched(batch_of_rows:torch.Tensor, hessian:torch.Tensor):
    '''
    Input:
    - `batch_of_rows`: tensor of shape `(batch_size, n_cols)`
    - `hessian`: tensor of shape `(n_cols, n_cols)`

    Output:
    - `quantized_rows`: tensor of shape `(batch_size, n_cols)`
    '''
    batch_size = batch_of_rows.size(0)
    # create a copy of `hessian` for each row in `batch_of_rows`
    # `batch_of_hessians.shape` is `(batch_size, n_cols, n_cols)`
    batch_of_hessians = hessian.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_of_masks = torch.BoolTensor(batch_of_rows.shape, device=batch_of_rows.device)
    # `min_zeros_per_row` has shape `(batch_size,)`
    min_zeros_per_row = (batch_of_rows == 0).float().sum(dim=1).min().item()
    '''
    At this point, we don't have _the_ hessian matrix, but _a_ hessian matrix for each row in `batch_of_rows`. 
    We modify each hessian matrix based on each row. 
    In particular, for each `i` in a row for which `row[i]` is 0, we set the `i`th row and `i`th column of the corresponding hessian matrix to 0. 
    '''
    batch_of_inverse_hessians = []
    for (row_index, row) in enumerate(batch_of_rows):
        is_zero = row == 0
        inverse_hessian_of_row = _update_hessian_at_zeros_and_return_inverse(
            is_zero, 
            batch_of_hessians[row_index]
        )
        batch_of_inverse_hessians.append(inverse_hessian_of_row)
        # We only select the top `min_zeros_per_row` indices for which `row[i]` is 0.
        top_n_nonzero_indices = get_top_n_nonzero_indices(is_zero, top_n=min_zeros_per_row)
        batch_of_masks[row_index, top_n_nonzero_indices] = True
    # `batch_of_masks` has shape `(batch_size, n_cols, n_cols)`
    batch_of_inverse_hessians = torch.stack(batch_of_inverse_hessians)

def exact_obc(original_weights:torch.Tensor, hessian:torch.Tensor, n_bits:int):
    '''
    `weights` is a tensor of shape (n_rows, n_cols)
    `hessian` is a tensor of shape (n_cols, n_cols)
    '''
    weights = original_weights.float().clone()
    weights, hessian = _handle_zeros(weights, hessian)

    quantized_weights = torch.zeros_like(weights)
    max_quantized_value = 2**n_bits - 1

    # `scale.shape` = `rounded_zeros.shape` = `(n_rows)`
    optimal_scales_and_zeros = find_optimal_scales_and_zeros(weights, max_quantized_value)