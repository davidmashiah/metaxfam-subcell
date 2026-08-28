"""
mininet.py
==========
A minimal convolutional neural network implemented from scratch in NumPy.

Why: PyTorch/TensorFlow are unavailable in this environment (no network access),
but convolutional networks are THE standard surrogate architecture in the
metamaterial ML literature. Omitting a CNN would leave the central question
("does a CNN generalize across design families?") untested. So we build one.

Layers: Conv2D (im2col), ReLU, MaxPool2D, Flatten, Dense.
Loss:   MSE.  Optimizer: Adam.

Correctness is established by finite-difference gradient checking (see
__main__), not by assertion.
"""

import numpy as np


# ---------------------------------------------------------------------------
# im2col helpers
# ---------------------------------------------------------------------------
def get_im2col_indices(x_shape, k, stride=1, pad=0):
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - k) // stride + 1
    out_w = (W + 2 * pad - k) // stride + 1
    i0 = np.repeat(np.arange(k), k)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(k), k * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    c = np.repeat(np.arange(C), k * k).reshape(-1, 1)
    return c, i, j, out_h, out_w


def im2col(X, k, stride=1, pad=0):
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    c, i, j, out_h, out_w = get_im2col_indices(X.shape, k, stride, pad)
    cols = Xp[:, c, i, j]                       # (N, C*k*k, out_h*out_w)
    C = X.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(k * k * C, -1)
    return cols, out_h, out_w


def col2im(cols, X_shape, k, stride=1, pad=0):
    N, C, H, W = X_shape
    Hp, Wp = H + 2 * pad, W + 2 * pad
    Xp = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)
    c, i, j, out_h, out_w = get_im2col_indices(X_shape, k, stride, pad)
    cols_r = cols.reshape(C * k * k, -1, N).transpose(2, 0, 1)
    np.add.at(Xp, (slice(None), c, i, j), cols_r)
    if pad == 0:
        return Xp
    return Xp[:, :, pad:-pad, pad:-pad]


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class Conv2D:
    def __init__(self, in_ch, out_ch, k, pad=1, rng=None):
        rng = rng or np.random.default_rng(0)
        self.k, self.pad = k, pad
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = rng.standard_normal((out_ch, in_ch, k, k)) * scale
        self.b = np.zeros(out_ch)
        self.params = ["W", "b"]

    def forward(self, X):
        self.X_shape = X.shape
        cols, out_h, out_w = im2col(X, self.k, 1, self.pad)
        self.cols = cols
        N = X.shape[0]
        out_ch = self.W.shape[0]
        Wr = self.W.reshape(out_ch, -1)
        out = Wr @ cols + self.b.reshape(-1, 1)
        out = out.reshape(out_ch, out_h, out_w, N).transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        out_ch = self.W.shape[0]
        dout_r = dout.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        self.dW = (dout_r @ self.cols.T).reshape(self.W.shape)
        self.db = dout_r.sum(axis=1)
        Wr = self.W.reshape(out_ch, -1)
        dcols = Wr.T @ dout_r
        return col2im(dcols, self.X_shape, self.k, 1, self.pad)


class ReLU:
    params = []

    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D:
    params = []

    def __init__(self, k=2):
        self.k = k

    def forward(self, X):
        N, C, H, W = X.shape
        k = self.k
        self.X_shape = X.shape
        Xr = X.reshape(N, C, H // k, k, W // k, k)
        out = Xr.max(axis=(3, 5))
        self.Xr = Xr
        self.out = out
        return out

    def backward(self, dout):
        N, C, H, W = self.X_shape
        k = self.k
        mask = (self.Xr == self.out[:, :, :, None, :, None])
        # split ties evenly
        counts = mask.sum(axis=(3, 5), keepdims=True)
        dX = mask * dout[:, :, :, None, :, None] / counts
        return dX.reshape(N, C, H, W)


class Flatten:
    params = []

    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, n_in, n_out, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)
        self.params = ["W", "b"]

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout):
        self.dW = self.X.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


# ---------------------------------------------------------------------------
# Network + Adam
# ---------------------------------------------------------------------------
class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for L in self.layers:
            X = L.forward(X)
        return X

    def backward(self, dout):
        for L in reversed(self.layers):
            dout = L.backward(dout)
        return dout

    def param_list(self):
        out = []
        for L in self.layers:
            for p in getattr(L, "params", []):
                out.append((L, p))
        return out


class Adam:
    def __init__(self, net, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps = net, lr, b1, b2, eps
        self.t = 0
        self.m = {}
        self.v = {}
        for L, p in net.param_list():
            key = (id(L), p)
            self.m[key] = np.zeros_like(getattr(L, p))
            self.v[key] = np.zeros_like(getattr(L, p))

    def step(self):
        self.t += 1
        for L, p in self.net.param_list():
            key = (id(L), p)
            g = getattr(L, "d" + p)
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * g * g
            mh = self.m[key] / (1 - self.b1 ** self.t)
            vh = self.v[key] / (1 - self.b2 ** self.t)
            setattr(L, p, getattr(L, p) - self.lr * mh / (np.sqrt(vh) + self.eps))


def make_cnn(seed=0, n_out=1, in_size=32):
    """Three conv/pool blocks; the dense layer adapts to the input resolution."""
    rng = np.random.default_rng(seed)
    s = in_size // 8          # after three 2x max-pools
    return Net([
        Conv2D(1, 8, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
        Conv2D(8, 16, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
        Conv2D(16, 32, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
        Flatten(),
        Dense(32 * s * s, 64, rng=rng), ReLU(),
        Dense(64, n_out, rng=rng),
    ])


def train(net, X, y, epochs=60, bs=32, lr=1e-3, verbose=False, seed=0):
    """X: (N,1,H,W) float, y: (N,1) float."""
    opt = Adam(net, lr=lr)
    rng = np.random.default_rng(seed)
    N = len(X)
    for ep in range(epochs):
        idx = rng.permutation(N)
        tot = 0.0
        for s in range(0, N, bs):
            b = idx[s:s + bs]
            pred = net.forward(X[b])
            diff = pred - y[b]
            loss = np.mean(diff ** 2)
            dout = 2.0 * diff / len(b) / y.shape[1]
            net.backward(dout)
            opt.step()
            tot += loss * len(b)
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"      epoch {ep:3d}  mse {tot/N:.5f}")
    return net


# ---------------------------------------------------------------------------
# Gradient check -- this is what proves the implementation is correct
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Finite-difference gradient check of the from-scratch CNN\n")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3, 1, 8, 8))
    y = rng.standard_normal((3, 1))

    net = Net([
        Conv2D(1, 4, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
        Flatten(), Dense(4 * 4 * 4, 5, rng=rng), ReLU(), Dense(5, 1, rng=rng),
    ])

    def loss_fn():
        pred = net.forward(X)
        return np.mean((pred - y) ** 2)

    # analytic gradients
    pred = net.forward(X)
    diff = pred - y
    net.backward(2.0 * diff / len(X))

    eps = 1e-5
    max_err = 0.0
    for L, p in net.param_list():
        P = getattr(L, p)
        G = getattr(L, "d" + p)
        flat = P.ravel()
        gflat = G.ravel()
        idxs = rng.choice(len(flat), size=min(8, len(flat)), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss_fn()
            flat[i] = orig - eps
            lm = loss_fn()
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            denom = max(1e-8, abs(num) + abs(ana))
            err = abs(num - ana) / denom
            max_err = max(max_err, err)
    print(f"  max relative gradient error: {max_err:.3e}")
    print("  " + ("PASS -- backprop is correct" if max_err < 1e-5
                  else "FAIL -- gradients are wrong"))
