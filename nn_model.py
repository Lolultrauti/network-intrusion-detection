"""
nn_model.py
-----------
A PyTorch feed-forward neural network (multi-layer perceptron) wrapped in a
scikit-learn-style interface so it drops straight into the existing pipeline:
it exposes .fit(), .predict(), .predict_proba() and .classes_, just like the
RandomForest / XGBoost models. That means train_model.py can compare it on
macro-F1 and predict.py can call .predict_proba() with no special handling.

This is the "deep learning" model: a real neural net trained with backprop +
Adam, as opposed to the tree-based classical-ML models.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _MLP(nn.Module):
    """Simple MLP: Linear -> BatchNorm -> ReLU -> Dropout, stacked, then logits."""

    def __init__(self, in_dim, out_dim, hidden=(128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))  # output logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier:
    """sklearn-style wrapper around the PyTorch MLP.

    Parameters mirror typical NN hyperparameters. Designed to be pickled via
    joblib (CPU tensors pickle cleanly).
    """

    def __init__(self, hidden=(128, 64), dropout=0.3, epochs=30, lr=1e-3,
                 batch_size=512, weight_decay=1e-5, random_state=42, verbose=True):
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed; cannot use TorchMLPClassifier.")
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.classes_ = None

    # -- training ------------------------------------------------------------
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        # Map raw class labels -> 0..C-1 contiguous indices for CrossEntropy.
        self._cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self._cls_to_idx[v] for v in y], dtype=np.int64)

        in_dim = X.shape[1]
        out_dim = len(self.classes_)
        self.model = _MLP(in_dim, out_dim, self.hidden, self.dropout)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for ep in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            if self.verbose and (ep == 0 or (ep + 1) % 5 == 0):
                print(f"  [NN] epoch {ep+1:3d}/{self.epochs}  "
                      f"loss={total/len(ds):.4f}")
        return self

    # -- inference -----------------------------------------------------------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X))
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

    # -- pickling: keep CPU state, drop nothing special ----------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            state["_model_state"] = self.model.state_dict()
            state["_model_arch"] = (self.model.net[0].in_features,
                                    len(self.classes_))
            state["model"] = None  # rebuilt on load
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if state.get("_model_state") is not None:
            in_dim, out_dim = state["_model_arch"]
            self.model = _MLP(in_dim, out_dim, self.hidden, self.dropout)
            self.model.load_state_dict(state["_model_state"])
            self.model.eval()
