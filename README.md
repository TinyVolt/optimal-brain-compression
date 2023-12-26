# optimal-brain-compression
This is an unofficial implementation of `ExactOBC` algorithm introduced in the paper [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) packed into a module. The official implementation can be found [here](https://github.com/IST-DASLab/OBC).

- Install using `ssh`:
```sh
pip install git+ssh://git@github.com/tinyvolt/optimal-brain-compression.git
```
- Install using `https`:
```sh
pip install git+https://github.com/tinyvolt/optimal-brain-compression.git
```

### Difference between the official implementation and this one
- The official implementation focuses on running various _experiments_, completeness and reproducibility of results. Its purpose is not reusability, readability or writing good _software_.
- This implementation focuses on reusability, readability and good programming practices. It only implements _one_ algorithm - `ExactOBS` for quantization - while leaving out other implementations like unstructured pruning and N:M pruning.
- This implementation focuses on modularity and not completeness. To be more precise, I did not implement the logic to calculate the Hessian by adding hooks and updating the (unnormalized) covariance matrix for each batch of data. __As long as you have a matrix and a Hessian, you can use this module to quantize the matrix based on the Hessian properties__. The onus of calculating the Hessian and storing the quantized matrix is on the user as of now.

### How to use
Its usage is super simple:
```python
from optimal_brain_compression import exact_obc
quantized_matrix = exact_obc(matrix, hessian, n_bits=4)
```

A complete working example is shown below:

```python
import torch
from optimal_brain_compression import exact_obc

torch.manual_seed(0)
n_rows = 24
n_cols = 5

matrix = torch.randn(n_rows, n_cols)
xs = torch.randn(100, n_cols)
hessian = (xs.t() @ xs).div(xs.shape[0] - 1)
quantized_matrix = exact_obc(matrix, hessian, n_bits=4)
# you can use a smaller batch size if needed
quantized_matrix = exact_obc(matrix, hessian, n_bits=4, batch_size=8)
```

### Directory structure
```
.
├── LICENSE
├── README.md
├── optimal_brain_compression
│   ├── __init__.py
│   ├── _checks.py
│   ├── _types.py
│   ├── _utils.py
│   └── exact_obc.py
└── setup.py
```

### Deep-dive into key ideas
I think the paper has a bunch of interesting ideas. 
- I got interested in the problem described in `Lemma 1` (equation 4) in the paper. I [wrote an article](https://www.linearalgebraforprogrammers.com/blog/inverse_row_col_removed) with a generalized form of this problem with the proofs and code.
- Proof for equations 3 and 7 in the paper:

If $f$ if a function parameterized by a vector of weights $W$, Taylor's expansion gives us (assuming $\nabla W = 0$ for a trained model):

$$ f(W + dW) - f(W) = \nabla W.dW + \frac{1}{2}(dW)^TH(dW) =  \frac{1}{2}(dW)^TH(dW) $$

where $1_p$ is the one hot vector for index $p$.

Let's say you want to set the element at index $p$ of $W$ denoted by $w_p$ to a constant $c$. In other words we want $(dW)_p + w_p = c$ which can be re-written as:

$$ 1_p^T.(dW) + w_p = c $$

We want to minimize $f(W + dW) - f(W) = \frac{1}{2}(dW)^TH(dW) \coloneqq F$ subject to the constraint $G = 1_p^T.(dW) + w_p - c = 0$. Using Lagrange multipliers, we have:

$$ \nabla F = H.(dW), \nabla G = 1_p $$

$$ \nabla F = \lambda \nabla G \Rightarrow H.(dW) = \lambda 1_p \Rightarrow dW = \lambda H^{-1}1_p $$

Using this value in $G$ gives us:

$$ 1_p^T(\lambda)H^{-1}1_p + w_p - c = 0 \Rightarrow \lambda = \frac{c - w_p}{H^{-1}_{pp}} $$

where ${H^{-1}_{pp}}$ is the $p$-th diagonal value of $H^{-1}$. 

This gives us:

$$ dW = \frac{c - w_p}{H^{-1}_{pp}} H^{-1}1_p $$

- Setting $c = 0$ gives equation 3.
- Setting $c = \text{quant}(w_p)$ gives equation 7.

Finally, using this value of $dW$ in the equation $f(W + dW) - f(W) =  \frac{1}{2}(dW)^TH(dW)$ gives us:

$$ f(W + dW) - f(W) = \frac{1}{2}\frac{(c - w_p)^2}{H^{-1}_{pp}} $$
