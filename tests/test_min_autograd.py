# tests/test_min_autograd.py
import numpy as np
from numpy.testing import assert_allclose
from autodiff.tensor import Tensor

def numerical_grad_vec(f, x, eps=1e-6):
    g = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        x_pos = x.copy(); x_pos[i] += eps
        x_neg = x.copy(); x_neg[i] -= eps
        g[i] = (f(x_pos) - f(x_neg)) / (2*eps)
    return g

def test_sum_of_squares_grad():
    rng = np.random.default_rng(0)
    x_np = rng.normal(size=7)
    x = Tensor(x_np, requires_grad=True)
    y = (x * x).sum()
    y.backward()
    assert x.grad is not None
    assert_allclose(x.grad, 2.0 * x_np, rtol=1e-6, atol=1e-8)

def test_matvec_mse_grad_w():
    rng = np.random.default_rng(1)
    n, d = 8, 4
    X = rng.normal(size=(n, d))
    w_np = rng.normal(size=(d,))
    y = X @ w_np + rng.normal(scale=0.01, size=n)  # no bias term

    X_t = Tensor(X)
    w = Tensor(w_np, requires_grad=True)
    y_t = Tensor(y)

    pred = X_t @ w             # (n,)
    loss = ((pred - y_t) * (pred - y_t)).mean()
    loss.backward()
    assert w.grad is not None

    def f_w(wv):
        p = X @ wv
        return ((p - y)**2).mean()

    g_num = numerical_grad_vec(f_w, w_np)
    assert_allclose(w.grad, g_num, rtol=1e-5, atol=1e-7)

def test_scalar_mul_grad():
    """Tests element-wise multiplication on scalar tensors."""
    x = Tensor(5.0, requires_grad=True)
    y = x * x
    y.backward()
    assert x.grad is not None
    assert_allclose(x.grad, 10.0, rtol=1e-6, atol=1e-8)

def test_subtraction_grad():
    """Tests gradients for subtraction."""
    x_np = np.array([1., 2.])
    y_np = np.array([3., 4.])
    x = Tensor(x_np, requires_grad=True)
    y = Tensor(y_np, requires_grad=True)
    z = x - y
    loss = z.sum()
    loss.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert_allclose(x.grad, np.array([1., 1.]), rtol=1e-6, atol=1e-8)
    assert_allclose(y.grad, np.array([-1., -1.]), rtol=1e-6, atol=1e-8)

def test_combined_reductions_grad():
    """Tests gradients for a combination of sum() and mean()."""
    rng = np.random.default_rng(0)
    x_np = rng.normal(size=5)
    x = Tensor(x_np, requires_grad=True)
    y = (x * x).sum() + x.mean()
    y.backward()
    assert x.grad is not None
    # Analytical gradient: d(x^2)/dx + d(x/N)/dx = 2x + 1/N
    n = x_np.size
    analytical_grad = 2 * x_np + (1.0 / n)
    assert_allclose(x.grad, analytical_grad, rtol=1e-6, atol=1e-8)

def test_composed_ops_grad_concise():
    """
    Tests composition of operations:
    Loss = sum((A * B) - A).
    """
    A_np = np.array([1., 2.])
    B_np = np.array([3., 4.])
    A = Tensor(A_np, requires_grad=True)
    B = Tensor(B_np, requires_grad=True)

    # Composition: (Element-wise Multiplication) -> (Subtraction) -> (Sum Reduction)
    loss = (A * B - A).sum()
    
    loss.backward()

    # Expected Analytical Gradients:
    # d(Loss)/dA = B - 1
    analytical_grad_A = B_np - 1
    # d(Loss)/dB = A
    analytical_grad_B = A_np

    assert_allclose(A.grad, analytical_grad_A, rtol=1e-6, atol=1e-8)
    assert_allclose(B.grad, analytical_grad_B, rtol=1e-6, atol=1e-8)

