"""
Hunt Arena: a 7x7 grid pursuit task for a 1-hour RL workshop.

Single-agent Gymnasium env. The learner is the hunter; the prey is a fixed
script, not a policy, so the environment is stationary and a submission can be
scored on an absolutely fixed benchmark.

Rules
-----
- 7x7 grid, 6 wall cells, no wraparound.
- The hunter moves in 8 directions, the prey in 4 (diagonals resolve to STAY).
  This asymmetry is what makes the prey catchable -- with equal move sets the
  gap between them is invariant in open space.
- A catch is landing on the prey, or swapping cells with it.
- On a catch: +1, and the prey respawns at least 4 cells away. The hunter does
  NOT move, so a catch costs you your proximity -- which is exactly what makes
  the shaping reward exploitable (see shaping_coef).
- An episode is 200 steps. Score = catches.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core import (GRID, WALLS, FREE, N_ACTIONS, OBS_DIM, MAX_D, MIN_SPAWN_DIST,
                  MIN_RESPAWN_DIST, MATCH_STEPS, chebyshev, make_obs, resolve,
                  prey_action, caught)


class HuntEnv(gym.Env):
    """Gymnasium env. The learner hunts; the prey is scripted.

    Parameters
    ----------
    shaping_coef : float
        Per-step reward for being close to the prey. Speeds learning up a lot.
        It is also not free: a catch teleports the prey away and cuts off the
        income, so a large coefficient pays you to follow and never catch.
    step_penalty : float
        Constant per-step reward. Negative values pressure you to hurry.
    prey_noise : float
        Probability the prey moves at random instead of fleeing. This is a
        TRAINING knob only -- the leaderboard always evaluates at
        EVAL_PREY_NOISE, so training against a dopey prey is your problem.
    match_steps : int
        Episode length. Keep at 200 to match the benchmark.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, shaping_coef=0.0, step_penalty=0.0, prey_noise=0.1,
                 match_steps=MATCH_STEPS, seed=None):
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.shaping_coef = float(shaping_coef)
        self.step_penalty = float(step_penalty)
        self.prey_noise = float(prey_noise)
        self.match_steps = int(match_steps)
        self.rng = np.random.default_rng(seed)

    # -- core ---------------------------------------------------------------

    def _spawn_pair(self):
        while True:
            i, j = self.rng.integers(len(FREE), size=2)
            if chebyshev(FREE[i], FREE[j]) >= MIN_SPAWN_DIST:
                return FREE[i], FREE[j]

    def _respawn_prey(self):
        while True:
            p = FREE[self.rng.integers(len(FREE))]
            if chebyshev(self.me, p) >= MIN_RESPAWN_DIST:
                return p

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.me, self.prey = self._spawn_pair()
        self.t = 0
        self.catches = 0
        return make_obs(self.me, self.prey), {}

    def step(self, action):
        if self.rng.random() < self.prey_noise:
            pa = int(self.rng.integers(N_ACTIONS))
        else:
            pa = prey_action(self.prey, self.me)

        old_me, old_prey = self.me, self.prey
        self.me = resolve(self.me, int(action), True)
        self.prey = resolve(self.prey, pa, False)

        reward = self.step_penalty
        if caught(old_me, old_prey, self.me, self.prey):
            reward += 1.0
            self.catches += 1
            self.prey = self._respawn_prey()
        elif self.shaping_coef:
            reward += self.shaping_coef * (1.0 - chebyshev(self.me, self.prey) / MAX_D)

        self.t += 1
        truncated = self.t >= self.match_steps
        return (make_obs(self.me, self.prey), reward, False, truncated,
                {"catches": self.catches})

    def render(self):
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if (r, c) in WALLS:
                    row += "#"
                elif (r, c) == self.me:
                    row += "H"
                elif (r, c) == self.prey:
                    row += "p"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)
