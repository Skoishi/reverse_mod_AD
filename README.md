# Mini Autograd (No Broadcasting) + Linear Regression

Build a tiny reverse-mode automatic differentiation engine and use it to fit **least-squares linear regression**.

## Learning goals

* Understand and implement **reverse-mode automatic differentiation** (backprop).
* Work with a minimal **Tensor** API.
* Train a linear regression model using your own autograd.
* Practice clean code, determinism, and basic testing.

---

## Scope & constraints

* Python **3.11**, CPU only.
* Allowed: **NumPy**, standard library, **pytest**.
* Disallowed: torch/jax/tf/autograd libs; GPU; general broadcasting.
* Data type: **float64** everywhere.
* Elementwise ops require **identical shapes**.
* `@` supports only `(n, d) @ (d,) → (n,)`.
* `.backward()` is called **only on scalars**.

---

## Repository layout

```
autodiff/
  __init__.py        # allows importing as a package
  tensor.py          # backprop implementation TODOs here

experiments/
  linreg.py          # training script with four TODOs

tests/
  conftest.py        # resolves some path issues with pytest
  test_min_autograd.py

requirements.txt
README.md
```

Install:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Run the experiment (shows loss at step 0, 50, 100, …):

```bash
python -m experiments.linreg --steps 200 --lr 0.1 --seed 0
```

---

## What you will implement

### In `autodiff/tensor.py`

We provide a documented starter with **`__add__` implemented** as an example. Implement the remaining **TODOs**:

* `Tensor.backward(self)`
  Backprop through the graph (details in the theory section below).
* Ops (same-shape only for elementwise):

  * `__neg__(self)`
  * `__sub__(self, other)`
  * `__mul__(self, other)`
  * `__matmul__(self, other)` restricted to `(n, d) @ (d,) → (n,)`
  * `sum(self)` (scalar)
  * `mean(self)` (scalar)

**Public API must remain:** `from autodiff.tensor import Tensor`.

### In `experiments/linreg.py`

A small linear regression using your `Tensor`:

* Intercept is handled by **adding a column of ones** to the features (so no bias broadcasting is needed).
* You will connect your ops in the training loop (prediction, loss, backward, update).
* Logging prints **step 0** (initial loss) before any update.

---

## Theory: how `.backward()` works (DFS + topological order)

Reverse-mode AD computes gradients by walking the computation graph **from outputs back to inputs**. A `Tensor` remembers:

* its **parents** (`._prev`) used to compute it, and
* a **local backward function** (`._backward`) that knows how to push the gradient from this node to its parents (via the chain rule).

### 1) Build a topological order (using DFS)

* Think of the graph as a DAG where edges point **from parents to child**.
* Starting from the output tensor (the one you call `.backward()` on), perform a **depth-first search** over `._prev`.
* After visiting a node’s parents, **append** the node to a list.
  Doing this ensures parents appear **before** their children in the list.
* The result is a **topological ordering** of all tensors that influence the output.

Why we need this: during backprop, a node’s gradient depends on gradients that have already flowed into its **children**. Processing nodes in **reverse topological order** guarantees that when you backprop through a node, the gradient **from all its uses** has already been accumulated at the child.

### 2) Seed the output gradient

* If the output is a **scalar**, the derivative of the output with respect to itself is **1**.
  Initialize the output tensor’s `.grad` to an array of ones with the same shape as the output (scalar shape).

### 3) Walk nodes in reverse order and apply local chain rule

* Iterate over the nodes in **reverse** topological order.
* For each node `v`, call its `v._backward()`. This closure:

  * reads `v.grad` (the gradient of the loss w\.r.t. `v`)
  * computes contributions to each parent’s gradient using the **local derivative** of `v` with respect to its parents
  * **adds** those contributions into each parent’s `.grad` (accumulation matters because the same tensor can be used by multiple children)

> Important: `.backward()` here assumes the output is scalar. If your expression produces a vector/array, first reduce to a scalar with `.sum()` or `.mean()`.

### 4) Gradient accumulation (+=)

If a parent is used more than once downstream, gradients from different paths **add**. Therefore, each parent does:

```
parent.grad = (parent.grad if already set else 0) + new_contribution
```

This is why we **accumulate**, not assign.

### 5) No broadcasting

Because all elementwise ops require identical shapes, you do **not** need to “un-broadcast” gradients. Shapes simply match on both sides of each local rule. (The only special case is reductions: pushing a scalar gradient back to an array, which you do by filling an array of ones and scaling appropriately.)

---

## Linear regression task

You’ll train a linear model:

* Dataset is synthetic and standardized.
* We add a column of ones to features to absorb the intercept into the weight vector.
* Model: `ŷ = X̃ @ w̃`
* Loss: **mean squared error** over residuals.

Your responsibilities in `experiments/linreg.py` (where marked):

1. **Parameters**: create a weight vector with `requires_grad=True`.
2. **Prediction**: compute `ŷ` using your `@` op (result shape `(n,)`).
3. **Loss**: compute the MSE using your elementwise ops and `.mean()`.
4. **Backward & update**:

   * zero any existing grads on parameters
   * call `.backward()` on the scalar loss
   * update parameters with a simple SGD step (learning rate provided)

**Logging**: The script prints **step 0** loss **before** the update so you can see the initial value. Expect the loss to decrease over time with reasonable hyperparameters.

---

## Tips

* Implement `.backward()` first; many bugs in ops are easier to see once backprop runs.
* For reductions:

  * `sum()`: each input element gets the same scalar incoming gradient.
  * `mean()`: like `sum()` but scaled by `1/N` where `N` is the number of elements.
* For `@` (matrix @ vector):

  * Check the shapes you’re promised (`(n,d) @ (d,) → (n,)`) and be strict.
  * Write the local derivatives on paper before you code the closure.
* Keep everything `float64` for stable comparisons against numerical checks.

---

## Determinism & hygiene

* Use float64 throughout.
* Don’t mutate `.data` **in-place** on tensors that still need gradients (except for the explicit parameter update step after `.backward()`).
* Seeded RNG in the experiment ensures repeatability.

---

## Allowed / disallowed

**Allowed**: NumPy, standard library, pytest, small helper functions in `tensor.py`.
**Disallowed**: Anything else including torch/jax/tf/autograd; GPU; adding broadcasting behavior.

---

## Grading (summary)

* **Correctness (65%)**: reverse-mode engine; gradients for `+`, `-`, unary `-`, `*`, `@ (n,d)@(d,)`, `sum`, `mean`; loss decreases in linreg.
* **Code quality (25%)**: clarity, comments, shape checks, determinism; passes provided tests; add one tiny test of your own.
* **Hygiene (10%)**: repo clean; instructions followed.

---

## Troubleshooting

* **“backward() requires scalar output”**: reduce with `.sum()` or `.mean()` first.
* **Loss not decreasing**: check your `mean()` scaling; verify `@` gradients; try a smaller `--lr`.
* **None gradients**: ensure parameters have `requires_grad=True`; seed the output grad to ones in `.backward()`; call `zero_grad()` before each step.

---

## Submission checklist

* [ ] `autodiff/tensor.py` compiles; all TODOs implemented.
* [ ] `pytest -q` passes locally.
* [ ] `python -m experiments.linreg --steps 200 --lr 0.1 --seed 0` prints step 0 loss and shows a downward trend.
* [ ] No external DL/autograd libs; no broadcasting added; float64 used.

Submit your modified files (tensor.py, test_min_autograd.py, linreg.py) as a zip file on Carmen.

good luck — and have fun building your own autograd!
