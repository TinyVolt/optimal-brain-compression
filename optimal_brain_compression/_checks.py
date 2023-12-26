import torch

def check_if_atleast_two_dim(x:torch.Tensor):
    assert x.ndim >= 2, f'Expected `x` to have at least 2 dimensions, but `x` has {x.ndim} dimension(s) only.'

def check_if_ndim(x:torch.Tensor, ndim:int):
    assert x.ndim == ndim, f'Expected `x` to have {ndim} dimensions, but `x` has {x.ndim} dimension(s).'

def check_if_square(x:torch.Tensor):
    check_if_atleast_two_dim(x)
    assert x.shape[-1] == x.shape[-2], f'Expected `x` to be square, but `x` has shape {x.shape}.'

def check_if_same_shape(x:torch.Tensor, y:torch.Tensor):
    assert x.shape == y.shape, f'Expected `x` and `y` to have the same shape, but got {x.shape} and {y.shape}.'

def check_if_square_and_same_shape(x:torch.Tensor, y:torch.Tensor):
    check_if_square(x)
    check_if_same_shape(x, y)

def check_if_first_n_dims_match(x:torch.Tensor, y:torch.Tensor, n:int):
    assert x.shape[:n] == y.shape[:n], f'Expected the first {n} dimensions of `x` and `y` to match, but got {x.shape[:n]} and {y.shape[:n]}.'

def check_if_same_ndim(x:torch.Tensor, y:torch.Tensor):
    assert x.ndim == y.ndim, f'Expected `x` and `y` to have the same number of dimensions, but got {x.ndim} and {y.ndim}.'