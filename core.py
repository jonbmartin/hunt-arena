"""
Hunt Arena rules and geometry. Pure numpy, no gymnasium dependency, so the
vectorized evaluator and the tests can import it anywhere.

The task is single-agent: one hunter chases one scripted prey. There is no
opponent policy to learn against, so a submission is judged on an absolutely
fixed benchmark rather than on who else showed up.
"""

import numpy as np

GRID = 7
WALLS = frozenset({(1, 2), (1, 4), (3, 1), (3, 5), (5, 2), (5, 4)})
FREE = tuple(sorted(
    (r, c) for r in range(GRID) for c in range(GRID) if (r, c) not in WALLS
))

# 0-3 cardinal, 4-7 diagonal, 8 stay
DELTAS = np.array([
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
    (0, 0),
], dtype=np.int8)
N_ACTIONS = 9
DIAGONALS = frozenset({4, 5, 6, 7})

# The hunter moves in 8 directions, the prey in 4. Without that asymmetry the
# gap between them is invariant in open space and the prey is never caught.
PREY_ACTIONS = (0, 1, 2, 3, 8)

OBS_DIM = 6
MAX_D = GRID - 1

# Initial spawn draws BOTH positions, so 4 is always satisfiable. A respawn
# holds the hunter fixed and only moves the prey -- and from the centre cell
# nothing is 4 away (max Chebyshev distance from (3,3) is 3), so requiring 4
# there would loop forever. 3 is reachable from every free cell: the four
# corners are never walls and are at distance >= 3 from anywhere.
MIN_SPAWN_DIST = 4
MIN_RESPAWN_DIST = 3

MATCH_STEPS = 200


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def make_obs(me, prey):
    """Observation from the hunter's point of view, roughly in [-1, 1]."""
    return np.array([
        me[0] / MAX_D * 2 - 1,
        me[1] / MAX_D * 2 - 1,
        prey[0] / MAX_D * 2 - 1,
        prey[1] / MAX_D * 2 - 1,
        (prey[0] - me[0]) / MAX_D,
        (prey[1] - me[1]) / MAX_D,
    ], dtype=np.float32)


def resolve(pos, action, can_diag):
    """Apply an action, respecting the 8-vs-4 asymmetry and walls."""
    if not can_diag and action in DIAGONALS:
        return pos  # prey cannot move diagonally
    dr, dc = DELTAS[action]
    nr, nc = pos[0] + int(dr), pos[1] + int(dc)
    if not (0 <= nr < GRID and 0 <= nc < GRID):
        return pos
    if (nr, nc) in WALLS:
        return pos
    return (nr, nc)


def prey_action(prey, hunter):
    """Scripted prey: step to whichever legal cell is furthest from the hunter.

    Deterministic, including the tie-break (first action in PREY_ACTIONS order
    wins). engine.prey_actions_batch must reproduce this exactly.
    """
    best, best_d = PREY_ACTIONS[0], None
    for a in PREY_ACTIONS:
        d = chebyshev(resolve(prey, a, False), hunter)
        if best_d is None or d > best_d:
            best, best_d = a, d
    return best


def caught(old_hunter, old_prey, new_hunter, new_prey):
    """A catch is landing on the prey, or swapping cells with it."""
    return new_hunter == new_prey or (new_hunter == old_prey and new_prey == old_hunter)
