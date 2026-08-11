"""
Tests. Run: python test_arena.py

Includes a minimal gymnasium stub so the env's *rules* can be verified even
where gymnasium isn't installed. If gymnasium is present, the real one is used.
"""

import sys
import types

import numpy as np

try:
    import gymnasium  # noqa: F401
    STUBBED = False
except ImportError:
    STUBBED = True
    g = types.ModuleType("gymnasium")
    sp = types.ModuleType("gymnasium.spaces")

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Env:
        def reset(self, seed=None, options=None):
            return None

    sp.Box, sp.Discrete = _Box, _Discrete
    g.Env, g.spaces = _Env, sp
    sys.modules["gymnasium"] = g
    sys.modules["gymnasium.spaces"] = sp

import core
import engine
from env import HuntEnv
from weights import NumpyMLP

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(label)


def test_geometry():
    print("geometry")
    check("walls are inside the grid",
          all(0 <= r < core.GRID and 0 <= c < core.GRID for r, c in core.WALLS))
    check("free cells exclude walls", not (set(core.FREE) & set(core.WALLS)))
    seen, stack, free = {core.FREE[0]}, [core.FREE[0]], set(core.FREE)
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (r + dr, c + dc)
            if n in free and n not in seen:
                seen.add(n)
                stack.append(n)
    check("grid is fully connected", len(seen) == len(core.FREE),
          f"{len(seen)}/{len(core.FREE)} reachable")
    check("spawn pairs all far enough apart",
          bool(np.all(np.maximum(
              np.abs(engine.START_PAIRS[:, 0] - engine.START_PAIRS[:, 2]),
              np.abs(engine.START_PAIRS[:, 1] - engine.START_PAIRS[:, 3]),
          ) >= core.MIN_SPAWN_DIST)))
    # the respawn bug that bites at the centre cell: every free cell must have
    # somewhere legal to put the prey, or the env loops forever
    check("every cell has a legal prey respawn",
          int(engine.RESPAWN_COUNT.min()) > 0,
          f"min {int(engine.RESPAWN_COUNT.min())} options")


def test_movement_rules():
    print("movement rules")
    open_cell = next(p for p in core.FREE
                     if all((p[0] + d[0], p[1] + d[1]) in core.FREE for d in core.DELTAS))
    hunter_moves = {core.resolve(open_cell, a, True) for a in range(core.N_ACTIONS)}
    prey_moves = {core.resolve(open_cell, a, False) for a in range(core.N_ACTIONS)}
    check("hunter has 8 moves plus stay", len(hunter_moves) == 9, f"{len(hunter_moves)}")
    check("prey has 4 moves plus stay", len(prey_moves) == 5, f"{len(prey_moves)}")
    check("walls block movement",
          all(core.resolve(p, a, True) not in core.WALLS
              for p in core.FREE for a in range(core.N_ACTIONS)))
    check("cannot leave the grid",
          all(0 <= core.resolve(p, a, True)[0] < core.GRID
              and 0 <= core.resolve(p, a, True)[1] < core.GRID
              for p in core.FREE for a in range(core.N_ACTIONS)))


def test_env_matches_engine():
    """The gym env and the vectorized evaluator must implement the same game."""
    print("env / engine agreement")
    rng = np.random.default_rng(0)
    for _ in range(400):
        pos = core.FREE[rng.integers(len(core.FREE))]
        act = int(rng.integers(core.N_ACTIONS))
        diag = bool(rng.integers(2))
        single = core.resolve(pos, act, diag)
        batch = engine.resolve_batch(np.array([pos]), np.array([act]), diag)[0]
        if tuple(batch) != tuple(single):
            check("resolve() agrees with resolve_batch()", False,
                  f"{pos} a={act} diag={diag}: {single} vs {tuple(batch)}")
            return
    check("resolve() agrees with resolve_batch()", True, "400 random cases")

    bad = None
    for _ in range(400):
        prey = core.FREE[rng.integers(len(core.FREE))]
        hunter = core.FREE[rng.integers(len(core.FREE))]
        single = core.prey_action(prey, hunter)
        batch = int(engine.prey_actions_batch(np.array([prey]), np.array([hunter]))[0])
        if single != batch:
            bad = f"prey={prey} hunter={hunter}: {single} vs {batch}"
            break
    check("prey_action() agrees with prey_actions_batch()", bad is None, bad or "400 cases")

    obs_single = core.make_obs((1, 2), (4, 3))
    obs_b = engine.obs_batch(np.array([[1, 2]]), np.array([[4, 3]]))[0]
    check("make_obs() agrees with obs_batch()", np.allclose(obs_single, obs_b))


def test_env_rollout():
    print("env rollout")
    env = HuntEnv(shaping_coef=0.05, seed=0)
    obs, _ = env.reset(seed=0)
    check("observation shape", obs.shape == (core.OBS_DIM,), str(obs.shape))
    rng = np.random.default_rng(0)
    steps, catches = 0, 0
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(int(rng.integers(core.N_ACTIONS)))
            steps += 1
            done = term or trunc
        catches += info["catches"]
    check("episodes are the right length", steps == 5 * core.MATCH_STEPS, str(steps))
    check("observations stay in range", bool(np.all(np.abs(obs) <= 2.0)))
    check("a random agent still catches sometimes", catches > 0,
          f"{catches} catches in 5 episodes")
    check("agent never sits on a wall", env.me not in core.WALLS)

    # the centre-cell respawn hazard, exercised directly
    env.me = (core.GRID // 2, core.GRID // 2)
    check("respawn terminates from the centre cell",
          core.chebyshev(env.me, env._respawn_prey()) >= core.MIN_RESPAWN_DIST)


def test_shaping_is_exploitable():
    """The follow-but-never-catch exploit must exist -- it's the lecture's payoff."""
    print("reward shaping")
    T = core.MATCH_STEPS

    def hover(coef):                       # sit at distance 1, never catch
        return coef * (1 - 1 / core.MAX_D) * T

    def hunt(coef, per=8):                 # catch every ~8 steps
        return T / per + coef * 0.5 * T    # plus shaping earned on approach

    check("high shaping_coef makes hovering pay better",
          hover(0.5) > hunt(0.5), f"hover={hover(0.5):.0f} vs hunt={hunt(0.5):.0f}")
    check("low shaping_coef does not",
          hover(0.05) < hunt(0.05), f"hover={hover(0.05):.0f} vs hunt={hunt(0.05):.0f}")


def test_submission_roundtrip():
    print("submissions")
    rng = np.random.default_rng(0)
    layers = [(rng.normal(0, .5, (32, core.OBS_DIM)), np.zeros(32)),
              (rng.normal(0, .5, (core.N_ACTIONS, 32)), np.zeros(core.N_ACTIONS))]
    mlp = NumpyMLP(layers, "tanh")
    mlp.save("/tmp/_t.npz")
    back = NumpyMLP.load("/tmp/_t.npz")
    probe = rng.uniform(-1, 1, (16, core.OBS_DIM)).astype(np.float32)
    check("save/load round trip", np.allclose(mlp(probe), back(probe)))
    d = np.load("/tmp/_t.npz", allow_pickle=False)
    check("file contains no pickled objects", all(d[k].dtype != object for k in d.files))
    small = engine.Benchmark(n_episodes=4, steps=50)
    check("policy can play the benchmark",
          engine.evaluate(back, small)[0].shape == (4,))


def test_benchmark_is_fair():
    print("benchmark")
    b = engine.Benchmark(n_episodes=30)
    a1 = engine.evaluate(engine.greedy_hunter, b)[0]
    a2 = engine.evaluate(engine.greedy_hunter, b)[0]
    check("same agent scores identically twice", bool(np.array_equal(a1, a2)),
          f"{a1.sum()} then {a2.sum()}")
    r = engine.evaluate(engine.random_hunter, b)[0]
    check("skill shows up as score", a1.sum() > 5 * max(1, r.sum()),
          f"greedy {a1.sum()} vs random {r.sum()}")
    _, tr = engine.evaluate(engine.greedy_hunter, b, record_episode=0)
    check("replay trace has one frame per step",
          len(tr["hunter"]) == b.steps + 1 and len(tr["caught"]) == b.steps,
          f"{len(tr['hunter'])} frames")


if __name__ == "__main__":
    if STUBBED:
        print("(gymnasium not installed -- using stub; SB3 paths untested here)\n")
    for fn in [test_geometry, test_movement_rules, test_env_matches_engine,
               test_env_rollout, test_shaping_is_exploitable,
               test_submission_roundtrip, test_benchmark_is_fair]:
        fn()
    print(f"\n{len(FAILURES)} failures" if FAILURES else "\nall tests passed")
    sys.exit(1 if FAILURES else 0)
