# experiments/linreg.py
import argparse, json, os
import numpy as np
from autodiff.tensor import Tensor

def make_data(n=200, d=5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d,))
    b_true = 0.5
    y = X @ w_true + b_true + rng.normal(scale=noise, size=n)
    # standardize features for stability
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    # add column of ones to absorb the intercept (no broadcasting needed)
    X_tilde = np.concatenate([X, np.ones((n, 1))], axis=1)
    return X_tilde, y  # y shape: (n,)

def train(n=200, d=5, noise=0.1, lr=0.01, steps=200, seed=0):
    X_np, y_np = make_data(n=n, d=d, noise=noise, seed=seed)
    X = Tensor(X_np)                          # data tensors: no grads
    y = Tensor(y_np)

    w = Tensor(np.random.randn(d+1,) *0.1 , requires_grad=True)
    # initialize parameters (requires_grad=True).

    log_every = 50  # print step 0, 50, 100, ...

    for t in range(steps):
        # prediction using your Tensor ops (shape: (n,))
        y_predict=X @ w

        # mean squared error loss using your Tensor ops (scalar)
        loss = ((y-y_predict) * (y-y_predict)).mean()

        # log BEFORE the update so step 0 shows the initial loss
        if t % log_every == 0:
            print(f"step {t:3d} | loss={loss.data:.6f}")

        # zero grads, backprop, and SGD step
        w.zero_grad()
        loss.backward()
        w.data -= lr * np.asarray(w.grad.data)

    # metrics back on NumPy
    mse = ((X_np @ w.data - y_np) ** 2).mean()
    return {"mse": float(mse), "w": w.data.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    res = train(n=args.n, d=args.d, noise=args.noise, lr=args.lr,
                steps=args.steps, seed=args.seed)
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f)

if __name__ == "__main__":
    main()
