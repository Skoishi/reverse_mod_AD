# autodiff/tensor.py
"""
Mini reverse-mode autograd for teaching (no broadcasting).

This file provides a tiny Tensor class and the minimal machinery to build a
computational graph and run reverse-mode automatic differentiation
("backpropagation").

✔ What’s supported in this assignment
-------------------------------------
- Data type: float64 NumPy arrays only.
- Device: CPU only.
- Elementwise ops (same shape ONLY): +, -, unary -, *
- Matrix @ vector: (n, d) @ (d,) -> (n,)
- Reductions to scalar: .sum(), .mean()
- Reverse-mode AD via Tensor.backward() on SCALAR outputs.

✘ What’s NOT supported (on purpose, to keep it simple)
------------------------------------------------------
- No general broadcasting rules for binary ops.
  → All elementwise binary ops require identical shapes.
- No GPU, no external autograd libs (no torch/jax/tf/autograd).
- No higher-order gradients.

How to use (once implemented)
-----------------------------
>>> from autodiff.tensor import Tensor
>>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
>>> y = (x * x).sum()
>>> y.backward()
>>> x.grad    # -> array([2., 4., 6.])

Design notes
------------
Each operation creates a new Tensor `out` and:
  • Saves parents in `out._prev` (to trace the graph).
  • Installs a closure `out._backward()` that knows how to push the gradient
    from `out` back to its parents (using the chain rule).
`Tensor.backward()`:
  • Builds a topological ordering of nodes ending at `self`.
  • Seeds `self.grad` with 1 (for scalar outputs).
  • Walks nodes in reverse topo order, calling each node’s `_backward()`.

Implementation strategy for TODOs
---------------------------------
1) Copy the pattern shown in `__add__`.
2) In each op’s `_backward`, write down dL/d(parent) using local derivatives:
   - For z = x + y:         dz/dx = 1,     dz/dy = 1
   - For z = x - y:         dz/dx = 1,     dz/dy = -1
   - For z = x * y:         dz/dx = y,     dz/dy = x
   - For y = A @ b:         dL/dA = outer(dL/dy, b),   dL/db = A^T @ (dL/dy)
   - For s = x.sum():       dL/dx = ones_like(x) * (dL/ds)
   - For m = x.mean():      dL/dx = (1/N) * ones_like(x) * (dL/dm)
3) ALWAYS accumulate into `.grad` (+=), because the same tensor may feed multiple
   children in the graph.

Caveat about broadcasting
-------------------------
We disallow general broadcasting in binary ops to reduce complexity. Internally,
NumPy may "broadcast" a scalar into an array (e.g., multiplying a scalar by a
vector). That’s fine here and doesn’t count as "broadcasting" for this
assignment.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, List, Set
import numpy as np

Array = np.ndarray


def _to_array(x) -> Array:
    """
    Convert input to a float64 NumPy array.

    Accepts Python scalars, lists, tuples, or NumPy arrays.
    We standardize on float64 for predictable numerical behavior and
    easier comparison with finite-difference checks in tests.
    """
    return np.asarray(x, dtype=np.float64)


class Tensor:
    """
    A tiny reverse-mode autodiff tensor (no broadcasting).

    Attributes
    ----------
    data : np.ndarray
        The actual numeric values (float64).
    grad : np.ndarray | None
        The gradient d(output)/d(self) after calling .backward() on a scalar
        output node. None if gradient has not been computed or does not flow
        to this tensor (e.g., requires_grad=False).
    requires_grad : bool
        If False, gradients need not be tracked for this tensor.
    _prev : tuple[Tensor, ...]
        Parent tensors that were used to compute this tensor.
    _backward : Callable[[], None]
        A closure that, when called, propagates the gradient from this tensor
        to its parents using the chain rule. Installed by each op.

    Notes
    -----
    - We only call .backward() when this Tensor is a scalar.
    - Graph is a DAG for the purposes of this assignment (no cycles / in-place).
    """

    def __init__(self, data, *, requires_grad: bool = False):
        self.data: Array = _to_array(data)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[Array] = None

        # Autograd bookkeeping
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple[Tensor, ...] = ()
        self._op: Optional[str] = None  # for debugging/printing if desired

    # -------- convenience --------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the underlying data array."""
        return self.data.shape

    def zero_grad(self) -> None:
        """
        Set .grad to None.

        Call this before a new backward pass to avoid mixing gradients
        from different passes (e.g., between training steps).
        """
        self.grad = None

    # -------- core engine --------

    def backward(self) -> None:
        """
        Reverse-mode autodiff: populate .grad in leaf tensors.

        Preconditions
        -------------
        - This tensor must be a scalar (shape == () or (1,)). If you need a
          scalar from a vector/array, call .sum() or .mean() first.
        - The computational graph must be buildable via this tensor's _prev links.

        Algorithm
        ---------
        1) Build a topological ordering of all nodes that influence `self`
           using depth-first search (DFS) over `._prev`.
        2) Seed the output gradient: if `self.grad` is None, set it to 1.0
           (same shape as self.data).
        3) Walk nodes in reverse topological order and call each node’s
           `_backward()` to push gradients to its parents.
        4) Each parent accumulates into `.grad` (+=), because multiple children
           may contribute to the same parent.

        Raises
        ------
        RuntimeError
            If called on a non-scalar tensor.
        """
        # TODO:
        #  - Assert `self` is scalar.
        #  - Build topological order via depth-first search over ._prev.
        #  - Seed self.grad with ones_like(self.data) if None.
        #  - Iterate reversed(topo) and call v._backward().
        raise NotImplementedError

    # -------- operations --------
    # All elementwise binary ops require identical shapes (no broadcasting).

    # Implemented exemplar: addition
    def __add__(self, other: Tensor) -> Tensor:
        """
        Elementwise addition (same shape only).

        Forward
        -------
        out.data = self.data + other.data

        Backward
        --------
        If L is the final loss and out = self + other:
            dL/dself  += dL/dout * 1
            dL/dother += dL/dout * 1
        """
        if not isinstance(other, Tensor):
            raise TypeError("Only Tensor + Tensor is supported (no broadcasting).")
        assert self.data.shape == other.data.shape, "shape mismatch (no broadcasting)"

        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        out._prev = (self, other)
        out._op = "add"

        def _backward():
            if self.requires_grad:
                g = out.grad  # same shape as self.data
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = out.grad  # same shape as other.data
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        """
        Unary negation.

        Forward
        -------
        out.data = -self.data

        Backward
        --------
        dL/dself += dL/dout * (-1)
        """
        # TODO: create out, set parents/op, implement _backward multiplying by -1
        raise NotImplementedError

    def __sub__(self, other: Tensor) -> Tensor:
        """
        Elementwise subtraction (same shape only).

        Forward
        -------
        out.data = self.data - other.data

        Backward
        --------
        dL/dself  += dL/dout * 1
        dL/dother += dL/dout * (-1)

        Tip
        ---
        You can implement as: return self + (-other)
        """
        # TODO
        raise NotImplementedError

    def __mul__(self, other: Tensor) -> Tensor:
        """
        Elementwise multiplication (same shape only).

        Forward
        -------
        out.data = self.data * other.data

        Backward
        --------
        Using z = x * y:
          dL/dx += dL/dz * y
          dL/dy += dL/dz * x
        """
        # TODO: check types/shapes, create out, set parents/op, implement local grads
        raise NotImplementedError

    def __matmul__(self, other: Tensor) -> Tensor:
        """
        Matrix @ vector multiply, restricted to (n, d) @ (d,) -> (n,).

        Forward
        -------
        y = A @ b

        Backward
        --------
        Let g = dL/dy  (shape (n,))
        Then:
          dL/dA = outer(g, b)      # shape (n, d)
          dL/db = A.T @ g          # shape (d,)

        Raises
        ------
        AssertionError if shapes are not (n, d) @ (d,)
        """
        # TODO: check shapes, compute out, set parents/op, implement grads as above
        raise NotImplementedError

    def sum(self) -> Tensor:
        """
        Sum of all elements -> scalar.

        Forward
        -------
        s = Σ_i x_i

        Backward
        --------
        For every element i:
          dL/dx_i += dL/ds * 1

        Note
        ----
        Multiplying the scalar incoming grad into an array of ones is fine.
        """
        # TODO: create scalar out, set parent, implement backward with ones_like
        raise NotImplementedError

    def mean(self) -> Tensor:
        """
        Mean of all elements -> scalar.

        Forward
        -------
        m = (1/N) * Σ_i x_i

        Backward
        --------
        For every element i:
          dL/dx_i += (1/N) * dL/dm

        Tip
        ---
        Equivalent to sum() / N, but implement directly for clarity.
        """
        # TODO: like sum(), but divide by N (self.data.size)
        raise NotImplementedError

    # -------- debug/pretty --------

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data!r}, grad={self.grad!r}, "
            f"requires_grad={self.requires_grad})"
        )
