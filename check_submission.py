"""
Validate a submission before it reaches the leaderboard.

Run:  python check_submission.py submissions/YOURNAME.npz

Checks the file loads, the shapes line up with the arena's observation and
action spaces, the outputs are finite, and the agent actually plays. The last
check is the one that matters: it catches anyone who quietly changed the
observation encoding, which is the failure that otherwise surfaces halfway
through the results.
"""

import sys

import numpy as np

from core import OBS_DIM, N_ACTIONS
from weights import NumpyMLP
import engine


def check(path, verbose=True):
    problems = []

    try:
        mlp = NumpyMLP.load(path)
    except Exception as exc:
        return [f"could not load {path}: {exc}"], None

    in_dim = mlp.layers[0][0].shape[1]
    out_dim = mlp.layers[-1][0].shape[0]
    if in_dim != OBS_DIM:
        problems.append(f"expects {in_dim} observation features, arena provides {OBS_DIM}")
    if out_dim != N_ACTIONS:
        problems.append(f"outputs {out_dim} actions, arena has {N_ACTIONS}")

    if not problems:
        probe = np.random.default_rng(0).uniform(-1, 1, (256, OBS_DIM)).astype(np.float32)
        logits = mlp(probe)
        if not np.all(np.isfinite(logits)):
            problems.append("policy produced non-finite outputs (NaN or inf)")
        elif len(np.unique(np.argmax(logits, axis=1))) == 1:
            problems.append("policy picks the same action in every situation "
                            "-- it collapsed; try a higher ent_coef")

    catches = None
    if not problems:
        try:
            quick = engine.Benchmark(n_episodes=20)
            catches = float(engine.evaluate(mlp, quick)[0].mean())
        except Exception as exc:
            problems.append(f"crashed during play: {exc}")

    if verbose:
        print(f"checking {path}")
        for p in problems:
            print(f"  FAIL  {p}")
        if not problems:
            print(f"  ok    {catches:.1f} catches per episode")
            if catches < 1.0:
                print("        (valid, but barely catching anything -- "
                      "check ent_coef and shaping_coef)")
            print("  ready to upload")
    return problems, catches


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python check_submission.py path/to/submission.npz")
    problems, _ = check(sys.argv[1])
    sys.exit(1 if problems else 0)
