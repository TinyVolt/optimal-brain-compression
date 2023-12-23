import torch
from _types import ScalesAndZeros

from _checks import check_if_first_n_dims_match, check_if_square, check_if_ndim, check_if_same_ndim

def invert_spd_matrix(matrix:torch.Tensor) -> torch.Tensor:
    '''
    `matrix` is a (potentially batched) symmetric positive (semi-)definite matrix of shape `(*, n, n)`.
    If it is semi-definite, we add a diagonal matrix of ones to make it definite.
    '''
    check_if_square(matrix)
    try:
        inverse_matrix = torch.cholesky_inverse(torch.linalg.cholesky(matrix))
    except RuntimeError:
        print(f'Warning: matrix is not positive definite. Adding a diagonal matrix of ones.')
        # diagonal_matrix.shape = (n, n)
        diagonal_matrix = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
        # Add batch dimensions to diagonal matrix
        # If matrix has shape (d_1, d_2, ..., d_{n-2}, n, n), diagonal_matrix should have shape (1, 1, ..., 1, n, n)
        new_shape = (1,) * (matrix.ndim - 2) + diagonal_matrix.shape
        diagonal_matrix = diagonal_matrix.view(new_shape)
        inverse_matrix = torch.cholesky_inverse(torch.linalg.cholesky(matrix + diagonal_matrix))
    return inverse_matrix

def calculate_inverse_after_row_column_removal(
    inverse_matrix:torch.Tensor, 
    indices:torch.LongTensor, 
    is_symmetric:bool=True
) -> torch.Tensor:
    '''
    `inverse_matrix` is the inverse of some matrix with shape `(bs,n,n)`
    `indices` is a tensor of shape `(bs,)

    Implementation of Lemma 1 (equation 4) of the paper `Optimal Brain Compression` by Frantar et al.

    Problem description:
    We are given a matrix `matrix` and its inverse `inverse_matrix`. Now suppose we remove the i-th row and column of `matrix` to get a new matrix `new_matrix`. We would like to calculate the inverse of `new_matrix` using `inverse_matrix`.
    This is faster than calculating the inverse of `new_matrix` from scratch.

    Note that we want to calculate the inverse of `new_matrix` only from `inverse_matrix`. This is why the original matrix `matrix` is not an input to this function.
    Also note that we don't actually remove the rows and columns from the inverse that we calculate.

    If `is_symmetric` is True, we only need to calculate either the rows or the columns, not both.
    '''
    check_if_ndim(inverse_matrix, 3)
    check_if_square(inverse_matrix)
    check_if_ndim(indices, 1)
    check_if_first_n_dims_match(inverse_matrix, indices, 1)

    batch_arange = torch.arange(inverse_matrix.size(0), device=inverse_matrix.device)

    # (bs,n,n) -> (bs,n) -> (bs,) -> (bs,1)
    diagonal_entries = inverse_matrix.diagonal(dim1=-2, dim2=-1)[batch_arange, indices].unsqueeze(-1)
    # (bs,1) -> (bs,1,1)
    diagonal_entries_signs = diagonal_entries.sign().unsqueeze(-1)
    # (bs,1)
    diagonal_entries_sqrt = diagonal_entries.abs().sqrt()
    # `columns` has shape (bs, n)
    columns = inverse_matrix[batch_arange, :, indices].div(diagonal_entries_sqrt)

    if is_symmetric:
        # `columns.unsqueeze(-1) @ columns.unsqueeze(-2)`: (bs,n,1) @ (bs,1,n) -> (bs,n,n)
        to_substract = columns.unsqueeze(-1) @ columns.unsqueeze(-2)
        to_substract.mul_(diagonal_entries_signs)
        return inverse_matrix - to_substract
    else:
        rows = inverse_matrix[batch_arange, indices, :].div(diagonal_entries_sqrt)
        # `columns.unsqueeze(-1) @ rows.unsqueeze(-2)`: (bs,n,1) @ (bs,1,n) -> (bs,n,n)
        to_substract = columns.unsqueeze(-1) @ rows.unsqueeze(-2)
        to_substract.mul_(diagonal_entries_signs)
        return inverse_matrix - to_substract

def quantize(x:torch.Tensor, scales:torch.Tensor, rounded_zeros:torch.Tensor, max_quantized_value:int) -> torch.Tensor:
    '''
    1. scale, round, shift then clamp
    2. undo shift and scale operations
    3. return the final output

    `x` is a tensor of shape (n_rows, n_cols)
    `scales` is a tensor of shape (n_rows, 1)
    `rounded_zeros` is a tensor of shape (n_rows, 1)
    Output is a tensor of shape (n_rows, n_cols)
    '''
    check_if_same_ndim(x, scales)
    check_if_same_ndim(x, rounded_zeros)
    rounded_x = torch.round(x / scales)
    quantized_x = (rounded_x + rounded_zeros).clamp(0, max_quantized_value)
    return scales * (quantized_x - rounded_zeros)

def find_optimal_scales_and_zeros(
    matrix:torch.Tensor, 
    max_quantized_value:int, 
    *, 
    max_shrink:float = 0.8,
    grid_range:int=100, 
    norm=2.4
) -> ScalesAndZeros:
    '''
    `matrix` is a tensor of shape (n_rows, n_cols)
    `max_quantized_value` is an integer = 2 ** n_bits - 1

    Outputs:
    - `scales` is a tensor of shape (n_rows,)
    - `rounded_zeros` is a tensor of shape (n_rows,)

    Implementation of [this code](https://github.com/yhhhli/BRECQ/blob/e455d62e93c70351961f8991c913b59435bd165f/quant/quant_layer.py#L116-L130) from the BRECQ repository. But it calculates the scales and zeros for all the rows at once.
    '''
    check_if_ndim(matrix, 2)
    n_rows = matrix.shape[0]
    zero_num_rows = torch.zeros(n_rows, device=matrix.device)
    # min_values_per_row and max_values_per_row are tensors of shape (n_rows,)
    min_values_per_row = torch.minimum(matrix.min(dim=1).values, zero_num_rows)
    max_values_per_row = torch.maximum(matrix.max(dim=1).values, zero_num_rows)
    indices_of_all_zero_rows = (min_values_per_row == 0) & (max_values_per_row == 0)
    min_values_per_row[indices_of_all_zero_rows] = -1
    max_values_per_row[indices_of_all_zero_rows] = 1

    scales = (max_values_per_row - min_values_per_row) / max_quantized_value
    rounded_zeros = torch.round(-min_values_per_row / scales)

    distances_quantized_and_full_precision = torch.full((n_rows,), float('inf'), device=matrix.device)

    for i in range( int(max_shrink * grid_range) ):
        shrink_factor = 1 - i / grid_range
        shrunken_min_values_per_row = min_values_per_row * shrink_factor
        shrunken_max_values_per_row = max_values_per_row * shrink_factor
        shrunken_scales = (shrunken_max_values_per_row - shrunken_min_values_per_row) / max_quantized_value
        shrunken_rounded_zeros = torch.round(-shrunken_min_values_per_row / shrunken_scales)
        # quantized_matrix has shape (n_rows, n_cols)
        quantized_matrix = quantize(
            matrix, 
            shrunken_scales.unsqueeze(-1), 
            shrunken_rounded_zeros.unsqueeze(-1), 
            max_quantized_value
        )
        # shrunken_distances has shape (n_rows,)
        shrunken_distances = torch.norm(quantized_matrix - matrix, p=norm, dim=1)
        # Update the scales and rounded_zeros if shrunked_distances is smaller
        rows_to_update = shrunken_distances < distances_quantized_and_full_precision
        distances_quantized_and_full_precision[rows_to_update] = shrunken_distances[rows_to_update]
        scales[rows_to_update] = shrunken_scales[rows_to_update]
        rounded_zeros[rows_to_update] = shrunken_rounded_zeros[rows_to_update]

    return ScalesAndZeros(scales=scales, rounded_zeros=rounded_zeros)

def get_top_n_nonzero_indices(one_dim_tensor:torch.BoolTensor, top_n:int) -> torch.Tensor:
    '''
    - Input `one_dim_tensor` is a booleam tensor of shape `(n,)`
    - Output is a tensor containing the indices of the top `top_n` elements of `one_dim_tensor` that are True.
    Example: 
    - if `one_dim_tensor` is `[True, False, True, True, False]` and `top_n` is 2, 
    - the output is `[0, 2]` (top 2 indices of `one_dim_tensor` that are True)
    '''
    check_if_ndim(one_dim_tensor, 1)
    return one_dim_tensor.nonzero(as_tuple=True)[0][:top_n]