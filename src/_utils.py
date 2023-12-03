import torch
from typing import Union

from _checks import check_if_first_n_dims_match, check_if_square, check_if_ndim

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

    # (bs,n,n) -> (bs,n) -> (bs,) -> sqrt -> (bs,1)
    diagonal_entries_sqrt = inverse_matrix.diagonal(dim1=-2, dim2=-1)[batch_arange, indices].sqrt().unsqueeze(-1)
    # `columns` has shape (bs, n)
    columns = inverse_matrix[batch_arange, :, indices].div(diagonal_entries_sqrt)

    if is_symmetric:
        # `columns.unsqueeze(-1) @ columns.unsqueeze(-2)`: (bs,n,1) @ (bs,1,n) -> (bs,n,n)
        return inverse_matrix - columns.unsqueeze(-1) @ columns.unsqueeze(-2)
    else:
        rows = inverse_matrix[batch_arange, indices, :].div(diagonal_entries_sqrt)
        # `columns.unsqueeze(-1) @ rows.unsqueeze(-2)`: (bs,n,1) @ (bs,1,n) -> (bs,n,n)
        return inverse_matrix - columns.unsqueeze(-1) @ rows.unsqueeze(-2)

def quantize(x:torch.Tensor, scales:torch.Tensor, zeros:torch.Tensor, maxq:Union[torch.Tensor,int]) -> torch.Tensor:
    '''
    1. scale, round, shift then clamp
    2. undo shift and scale operations
    3. return the final output
    '''
    rounded_x = torch.round(x / scales)
    quantized_x = (rounded_x + zeros).clamp(0, maxq)
    return scales * (quantized_x - zeros)
