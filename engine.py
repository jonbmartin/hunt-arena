"""
Vectorized Hunt Arena evaluator.

Plays many episodes in parallel with numpy. Dynamics are identical to
env.HuntEnv (there is a test that asserts this); this version exists so that
scoring twenty submissions finishes while people are still looking at the
screen.

Fairness
--------
Every agent is scored on the SAME benchmark: the same start positions, and the
same pre-drawn stream of prey randomness and respawn choices. Two agents that
act identically get identical scores, and nobody's rank depends on who else
entered. That is the whole reason for the single-agent redesign, so don't
replace the pre-drawn streams with live rng calls.
"""

import numpy as np

from core import (GRID, WALLS, FREE, DELTAS, N_ACTIONS, PREY_ACTIONS, OBS_DIM,
                  MAX_D, MIN_SPAWN_DIST, MIN_RESPAWN_DIST, MATCH_STEPS,
                  chebyshev)

WALLGRID = np.zeros((GRID, GRID), bool)
for _r, _c in WALLS:
    WALLGRID[_r, _c] = True

FREE_ARR = np.array(FREE, np.int64)                      # (F, 2)
CELL_INDEX = np.full((GRID, GRID), -1, np.int64)
for _i, (_r, _c) in enumerate(FREE):
    CELL_INDEX[_r, _c] = _i

DELTA_ARR = DELTAS.astype(np.int64)
IS_DIAG = np.array([4 <= a <= 7 for a in range(N_ACTIONS)])

# All legal (hunter, prey) start pairs at Chebyshev distance >= MIN_SPAWN_DIST.
START_PAIRS = np.array(
    [(a[0], a[1], b[0], b[1]) for a in FREE for b in FREE
     if chebyshev(a, b) >= MIN_SPAWN_DIST],
    dtype=np.int64,
)

# For each free cell, the free cells a prey may respawn into. Precomputed as a
# padded table so a respawn is an O(1) lookup instead of rejection sampling --
# which also makes it reproducible from a pre-drawn stream.
_valid = [[j for j, b in enumerate(FREE) if chebyshev(a, b) >= MIN_RESPAWN_DIST]
          for a in FREE]
RESPAWN_COUNT = np.array([len(v) for v in _valid], np.int64)
RESPAWN_TABLE = np.zeros((len(FREE), int(RESPAWN_COUNT.max())), np.int64)
for _i, _v in enumerate(_valid):
    RESPAWN_TABLE[_i, :len(_v)] = _v
assert RESPAWN_COUNT.min() > 0, "some cell has nowhere to respawn the prey"

# The benchmark the leaderboard uses. Participants may train against anything;
# they are always scored here.
EVAL_EPISODES = 200
EVAL_PREY_NOISE = 0.1
EVAL_SEED = 20260809


# --------------------------------------------------------------------------
# batched rules
# --------------------------------------------------------------------------

def obs_batch(me, prey):
    """me, prey: (M, 2) int arrays -> (M, 6) float32."""
    me = me.astype(np.float32)
    prey = prey.astype(np.float32)
    return np.stack([
        me[:, 0] / MAX_D * 2 - 1,
        me[:, 1] / MAX_D * 2 - 1,
        prey[:, 0] / MAX_D * 2 - 1,
        prey[:, 1] / MAX_D * 2 - 1,
        (prey[:, 0] - me[:, 0]) / MAX_D,
        (prey[:, 1] - me[:, 1]) / MAX_D,
    ], axis=1).astype(np.float32)


def resolve_batch(pos, actions, can_diag):
    """Apply actions to (M, 2) positions, honouring walls and the 8-vs-4 rule."""
    delta = DELTA_ARR[actions].copy()
    if not can_diag:
        delta[IS_DIAG[actions]] = 0
    nxt = pos + delta
    inside = np.all((nxt >= 0) & (nxt < GRID), axis=1)
    clipped = np.clip(nxt, 0, GRID - 1)
    ok = inside & ~WALLGRID[clipped[:, 0], clipped[:, 1]]
    return np.where(ok[:, None], nxt, pos)


def prey_actions_batch(prey, hunter):
    """Vectorized scripted prey, matching core.prey_action including tie-break."""
    M = len(prey)
    best = np.full(M, PREY_ACTIONS[0], np.int64)
    best_d = np.full(M, -1.0)
    for a in PREY_ACTIONS:
        nxt = resolve_batch(prey, np.full(M, a, np.int64), False)
        d = np.max(np.abs(nxt - hunter), axis=1).astype(np.float64)
        better = d > best_d                      # strict: first action wins ties
        best = np.where(better, a, best)
        best_d = np.where(better, d, best_d)
    return best


def greedy_hunter(obs):
    """House reference policy: step to whichever legal cell is nearest the prey.

    Takes the same (M, 6) observation a submission sees and returns (M, 9)
    scores, so it plugs into the same argmax path as a NumpyMLP.
    """
    me = np.stack([(obs[:, 0] + 1) / 2 * MAX_D,
                   (obs[:, 1] + 1) / 2 * MAX_D], 1).round().astype(np.int64)
    prey = np.stack([(obs[:, 2] + 1) / 2 * MAX_D,
                     (obs[:, 3] + 1) / 2 * MAX_D], 1).round().astype(np.int64)
    scores = np.zeros((len(obs), N_ACTIONS), np.float32)
    for a in range(N_ACTIONS):
        nxt = resolve_batch(me, np.full(len(obs), a, np.int64), True)
        scores[:, a] = -np.max(np.abs(nxt - prey), axis=1)
    return scores


def random_hunter(obs, _rng=np.random.default_rng(0)):
    return _rng.random((len(obs), N_ACTIONS)).astype(np.float32)


# A deliberately imperfect greedy, as the middle rung of the leaderboard.
# Its mistakes are a pure function of the state rather than live rng draws, so
# the bot scores the same every time it is run -- the benchmark would not be
# reproducible otherwise.
_HASH_N = 4096
_NOISE_TABLE = np.random.default_rng(12345).random(_HASH_N)
_ACTION_TABLE = np.random.default_rng(54321).integers(N_ACTIONS, size=_HASH_N)


def _state_hash(me, prey):
    a = CELL_INDEX[me[:, 0], me[:, 1]]
    b = CELL_INDEX[prey[:, 0], prey[:, 1]]
    return (a * 131 + b * 17 + 7) % _HASH_N


def sloppy_hunter(obs, noise=0.08):
    """Greedy, but blunders in a fixed fraction of states.

    Tuned to ~15 catches per episode: clearly above a flailing agent, clearly
    below a tuned one. Note the noise bites far harder than a live coin flip
    would -- a state-determined blunder repeats every time that state comes
    round, so the bot loops instead of shrugging it off. 0.08 here is roughly
    as damaging as 0.35 of live randomness.
    """
    me = np.stack([(obs[:, 0] + 1) / 2 * MAX_D,
                   (obs[:, 1] + 1) / 2 * MAX_D], 1).round().astype(np.int64)
    prey = np.stack([(obs[:, 2] + 1) / 2 * MAX_D,
                     (obs[:, 3] + 1) / 2 * MAX_D], 1).round().astype(np.int64)
    scores = greedy_hunter(obs)
    h = _state_hash(me, prey)
    blunder = _NOISE_TABLE[h] < noise
    if blunder.any():
        scores = scores.copy()
        scores[blunder] = -1e3
        scores[blunder, _ACTION_TABLE[h[blunder]]] = 1e3
    return scores


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------

class Benchmark:
    """A fixed set of episodes, with all randomness drawn up front."""

    def __init__(self, n_episodes=EVAL_EPISODES, steps=MATCH_STEPS,
                 seed=EVAL_SEED, prey_noise=EVAL_PREY_NOISE):
        rng = np.random.default_rng(seed)
        self.n_episodes = int(n_episodes)
        self.steps = int(steps)
        self.starts = START_PAIRS[rng.integers(len(START_PAIRS), size=n_episodes)]
        self.noise_use = rng.random((n_episodes, steps)) < prey_noise
        self.noise_act = rng.integers(N_ACTIONS, size=(n_episodes, steps))
        # one respawn draw per step is more than any episode can consume
        self.respawn = rng.integers(1 << 30, size=(n_episodes, steps))


_DEFAULT = None


def benchmark():
    """The shared default benchmark, built once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Benchmark()
    return _DEFAULT


def _respawn_prey(hunter, draws):
    """Pick each episode's next prey cell from its pre-drawn number."""
    idx = CELL_INDEX[hunter[:, 0], hunter[:, 1]]
    pick = draws % RESPAWN_COUNT[idx]
    return FREE_ARR[RESPAWN_TABLE[idx, pick]]


def evaluate(policy, bench=None, record_episode=None):
    """Run every benchmark episode in parallel.

    Returns (catches_per_episode, trace). `trace` is None unless
    record_episode is an episode index, in which case it is a dict of that
    episode's hunter/prey positions and catch flags, for the replay.
    """
    bench = bench or benchmark()
    M, T = bench.n_episodes, bench.steps

    me = bench.starts[:, 0:2].copy()
    prey = bench.starts[:, 2:4].copy()
    catches = np.zeros(M, np.int32)
    ptr = np.zeros(M, np.int64)          # how many respawn draws each has used

    trace = None
    if record_episode is not None:
        e = int(record_episode)
        trace = {"hunter": [me[e].tolist()], "prey": [prey[e].tolist()],
                 "caught": []}

    for t in range(T):
        act = np.argmax(policy(obs_batch(me, prey)), axis=1)
        pa = np.where(bench.noise_use[:, t], bench.noise_act[:, t],
                      prey_actions_batch(prey, me))

        old_me, old_prey = me, prey
        me = resolve_batch(me, act, True)
        prey = resolve_batch(prey, pa, False)

        collided = np.all(me == prey, axis=1)
        swapped = np.all(me == old_prey, axis=1) & np.all(prey == old_me, axis=1)
        got = collided | swapped
        catches += got.astype(np.int32)

        k = int(got.sum())
        if k:
            prey = prey.copy()
            prey[got] = _respawn_prey(me[got], bench.respawn[got, ptr[got] % T])
            ptr = ptr.copy()
            ptr[got] += 1

        if trace is not None:
            trace["hunter"].append(me[e].tolist())
            trace["prey"].append(prey[e].tolist())
            trace["caught"].append(bool(got[e]))

    return catches, trace


def score(policy, bench=None):
    """Total catches across the benchmark -- the number on the leaderboard."""
    return int(evaluate(policy, bench)[0].sum())
