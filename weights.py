"""
Turn a trained SB3 policy into a pure-numpy MLP.

Two reasons this exists:

1. Speed. The tournament runs 190 pairings simultaneously as batched matrix
   multiplies. Calling model.predict() one observation at a time would take
   ~10 minutes; this takes ~3 seconds.
2. Safety. Submissions are .npz files containing nothing but float arrays, so
   loading 20 files from 20 strangers cannot execute code. SB3's own .zip
   format is pickle-based; don't collect those from a room full of people.
"""

import numpy as np

ACTIVATIONS = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0)}


class NumpyMLP:
    """Feed-forward net: [Linear, act] * n + Linear. Batched over rows."""

    def __init__(self, layers, activation="tanh"):
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32))
                       for W, b in layers]
        self.activation = activation
        self.act_fn = ACTIVATIONS[activation]

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < len(self.layers) - 1:
                x = self.act_fn(x)
        return x

    def save(self, path):
        arrs = {}
        for i, (W, b) in enumerate(self.layers):
            arrs[f"W{i}"] = W
            arrs[f"b{i}"] = b
        np.savez(path, n_layers=np.int32(len(self.layers)),
                 activation=np.array(self.activation), **arrs)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        n = int(d["n_layers"])
        layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)]
        return cls(layers, str(d["activation"]))


def extract_mlp(model):
    """Pull the actor pathway out of an SB3 ActorCriticPolicy."""
    import torch.nn as nn

    policy = model.policy
    modules = list(policy.mlp_extractor.policy_net) + [policy.action_net]

    layers, activation = [], "tanh"
    for m in modules:
        if isinstance(m, nn.Linear):
            layers.append((m.weight.detach().cpu().numpy(),
                           m.bias.detach().cpu().numpy()))
        elif isinstance(m, nn.Tanh):
            activation = "tanh"
        elif isinstance(m, nn.ReLU):
            activation = "relu"
    if not layers:
        raise RuntimeError("no Linear layers found -- is this an MlpPolicy?")
    return NumpyMLP(layers, activation)


def check_matches_sb3(model, mlp, n=200, seed=0):
    """Sanity check: numpy and torch must pick the same deterministic action."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, size=(n, mlp.layers[0][0].shape[1])).astype(np.float32)
    torch_actions, _ = model.predict(obs, deterministic=True)
    numpy_actions = np.argmax(mlp(obs), axis=1)
    agree = float(np.mean(np.asarray(torch_actions).ravel() == numpy_actions))
    return agree
