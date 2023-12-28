import torch
from ._checks import check_if_square, check_if_ndim
from ._utils import invert_spd_matrix, find_optimal_scales_and_zeros, get_top_n_nonzero_indices, quantize
from ._types import Weights_Hessian_Tuple, QuantizedMatrix, Hessian_Mask_MinZeros_Tuple

def _handle_zeros(weights:torch.Tensor, hessian:torch.Tensor) -> Weights_Hessian_Tuple:
    '''
    - `weights` has shape `(n_rows, n_cols)`
    - `hessian` has shape `(n_cols, n_cols)`
    1. Find the diagonal entries of `hessian` which are zero.
    2. Set them to `1`.
    3. Set the corresponding columns in `weights` matrix to 0. This is because if the Hessian diagonal entries are zero, it means the inputs are always zero. Then we might as well set the weights to be zero.
    4. Return the modified `weights` and `hessian` matrices.
    '''
    # TODO: introduce relative damping as an option
    # The hessian is calculated in high precision (fp64). We convert it back to fp32.
    hessian = hessian.float()
    indices_of_zero_diagonal = hessian.diag() == 0
    hessian[indices_of_zero_diagonal, indices_of_zero_diagonal] = 1
    weights[:, indices_of_zero_diagonal] = 0
    return weights, hessian

def _update_hessian_at_zeros_and_return_inverse(is_zero:torch.Tensor, hessian:torch.Tensor) -> torch.Tensor:
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

def _get_inverse_hessian_and_mask_per_row(batch_of_rows:torch.Tensor, hessian:torch.Tensor) -> Hessian_Mask_MinZeros_Tuple:
    '''
    Input:
    - `batch_of_rows`: tensor of shape `(batch_size, n_cols)`
    - `hessian`: tensor of shape `(n_cols, n_cols)`

    Output:
    A tuple of
    - `batch_of_inverse_hessians`: tensor of shape `(batch_size, n_cols, n_cols)`
    - `batch_of_masks`: tensor of shape `batch_of_rows.shape`
    - `min_zeros_per_row`: int
    '''
    batch_size = batch_of_rows.size(0)
    # create a copy of `hessian` for each row in `batch_of_rows`
    # `batch_of_hessians.shape` is `(batch_size, n_cols, n_cols)`
    batch_of_hessians = hessian.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_of_masks = torch.zeros_like(batch_of_rows, dtype=torch.bool, device=batch_of_rows.device)
    # `min_zeros_per_row` has shape `(batch_size,)`
    min_zeros_per_row = (batch_of_rows == 0).float().sum(dim=1).min().long().item()
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
    return batch_of_inverse_hessians, batch_of_masks, min_zeros_per_row

def exact_obc(original_weights:torch.Tensor, hessian:torch.Tensor, n_bits:int, batch_size:int=32):
    '''
    `original_weights` is a tensor of shape (n_rows, n_cols)
    `hessian` is a tensor of shape (n_cols, n_cols)
    '''
    check_if_ndim(original_weights, 2)
    check_if_ndim(hessian, 2)
    check_if_square(hessian)

    n_rows, n_cols = original_weights.shape
    weights = original_weights.float().clone()
    weights, hessian = _handle_zeros(weights, hessian)

    quantized_weights = torch.zeros_like(weights, device=weights.device)
    max_quantized_value = 2**n_bits - 1

    # `scale.shape` = `rounded_zeros.shape` = `(n_rows)`
    optimal_scales_and_zeros = find_optimal_scales_and_zeros(weights, max_quantized_value)
    # `sclaes_per_row.shape` = `rounded_zeros_per_row.shape` = `outlier_thresholds_per_row.shape` = `n_rows, 1`
    scales_per_row = optimal_scales_and_zeros.scales.unsqueeze(-1)
    rounded_zeros_per_row = optimal_scales_and_zeros.rounded_zeros.unsqueeze(-1)
    outlier_thresholds_per_row = 0.25 * (scales_per_row ** 2)

    for row_start_index in range(0, n_rows, batch_size):
        row_end_index = min(row_start_index + batch_size, n_rows)
        # `batch_of_rows.shape` = `(batch_size, n_cols)`
        batch_of_rows = weights[row_start_index:row_end_index]
        # `batch_of_scales_per_row.shape` = `batch_of_rounded_zeros_per_row.shape` = `batch_of_outlier_thresholds_per_row.shape` = `(batch_size, 1)`
        batch_of_scales_per_row = scales_per_row[row_start_index:row_end_index]
        batch_of_rounded_zeros_per_row = rounded_zeros_per_row[row_start_index:row_end_index]
        batch_of_outlier_thresholds_per_row = outlier_thresholds_per_row[row_start_index:row_end_index]
        # `batch_of_inverse_hessians.shape` = `(batch_size, n_cols, n_cols)`, `batch_of_masks.shape` = `(batch_size, n_cols)`
        batch_of_inverse_hessians, batch_of_masks, min_zeros_per_row = _get_inverse_hessian_and_mask_per_row(batch_of_rows, hessian)

        for _ in range(min_zeros_per_row, n_cols):
            batch_of_rows_quantized = quantize(batch_of_rows, batch_of_scales_per_row, batch_of_rounded_zeros_per_row, max_quantized_value)
            squared_differences = (batch_of_rows_quantized - batch_of_rows).pow(2)
            # `batch_of_inverse_hessian_diagonals.shape` = `(batch_size, n_cols)`
            batch_of_inverse_hessian_diagonals = batch_of_inverse_hessians.diagonal(dim1=1, dim2=2)
            # first part of equation 7 in the paper
            # `batch_of_scores.shape` = `(batch_size, n_cols)`
            batch_of_scores = squared_differences.div(batch_of_inverse_hessian_diagonals)
            # set the scores of the masked elements to infinity so that they are not selected
            batch_of_scores[batch_of_masks] = float('inf')
            # `index_to_quantize_per_row.shape` = `(batch_size)`
            index_to_quantize_per_row = batch_of_scores.argmin(dim=1)
            # set the squared differences of the masked elements to 0 so that they are not considered while selecting outliers
            squared_differences[batch_of_masks] = 0
            '''
            The paper says "we quantize outliers as soon as they appear".
            Select the rows which contain outliers and for which the weight to be quantized is non-zero.
            '''
            # `row_contains_outlier.shape` = `row_minimum_is_nonzero.shape` = `(batch_size)`
            row_contains_outlier = (squared_differences > batch_of_outlier_thresholds_per_row).any(dim=1)
            arange = torch.arange(batch_of_rows.size(0), device=batch_of_rows.device)
            row_minimum_is_nonzero = batch_of_rows[arange, index_to_quantize_per_row] != 0
            outlier_rows = row_contains_outlier & row_minimum_is_nonzero
            index_to_quantize_per_row[outlier_rows] = squared_differences[outlier_rows].argmax(dim=1)

            quantized_element_per_row = batch_of_rows_quantized[arange, index_to_quantize_per_row]
            quantized_weights[arange + row_start_index, index_to_quantize_per_row] = quantized_element_per_row
            
            # second part of equation 7 in the paper - update the remaining weights in each row
            # `i_th_column_per_hessian.shape` = `(batch_size, n_cols)`
            i_th_column_per_hessian = batch_of_inverse_hessians[arange, :, index_to_quantize_per_row]
            # `i_th_diagonal_per_hessian.shape` = `(batch_size)`
            i_th_diagonal_per_hessian = batch_of_inverse_hessian_diagonals[arange, index_to_quantize_per_row]
            non_quantized_element_per_row = batch_of_rows[arange, index_to_quantize_per_row]
            # `coefficients_per_row.shape` = `(batch_size, 1)` after unsqueezing
            coefficients_per_row = (non_quantized_element_per_row - quantized_element_per_row).div(i_th_diagonal_per_hessian)
            coefficients_per_row.unsqueeze_(-1)
            # `to_substract.shape` = `(batch_size, n_cols)`
            to_substract = coefficients_per_row * i_th_column_per_hessian
            batch_of_rows -= to_substract
            
            batch_of_masks[arange, index_to_quantize_per_row] = True

            # Same as the below line but faster since we have the intermediate results already:
            # batch_of_inverse_hessians = calculate_inverse_after_row_column_removal(batch_of_inverse_hessians, index_to_quantize_per_row)
            i_th_column_per_hessian /= i_th_diagonal_per_hessian.sqrt().unsqueeze(-1)
            # (batch_size, n_cols, 1) @ (batch_size, 1, n_cols) -> (batch_size, n_cols, n_cols)
            batch_of_inverse_hessians -= i_th_column_per_hessian.unsqueeze(2) @ i_th_column_per_hessian.unsqueeze(1)

    return QuantizedMatrix(quantized_matrix=quantized_weights, scales=scales_per_row, rounded_zeros=rounded_zeros_per_row)

