"""
HUNT ARENA -- train your agent.

Everything you need to change is in CONFIG below. You should not need to read
past the line that says STOP READING HERE, though you're welcome to.

Run:    python train.py YOURNAME
Output: submissions/YOURNAME.npz  <- upload this file
"""

import os
import sys
import time

import numpy as np

# ==========================================================================
# CONFIG -- this is your playground
# ==========================================================================

CONFIG = dict(
    # The comment after each knob is the range worth exploring, not a hint at
    # the answer. The ends are not automatically better -- some are much worse.

    # ---- how long you train ---------------------------------------------
    # More is better but you only have so many minutes. 1M takes a couple of
    # minutes on a Colab CPU, and this task is still improving at 2M. The
    # SCORE column is often still climbing when training ends.
    total_timesteps=1_000_000,   # 200_000 ... 3_000_000

    # ---- reward ----------------------------------------------------------
    # shaping_coef pays you every step for being close to the prey, on top of
    # the +1 for an actual catch. It makes learning much faster. It is also
    # not free -- catching the prey teleports it away from you. Try 0.0, try
    # 0.05, try 0.5, and watch what your agent actually does.
    shaping_coef=0.05,   # 0.0 ... 0.5

    # Constant per-step reward. Negative values pressure you to hurry.
    step_penalty=0.0,    # -0.01 ... 0.0

    # ---- who you train against -------------------------------------------
    # Probability the prey ignores you and moves at random. You are ALWAYS
    # scored against a prey with noise 0.1, whatever you train against.
    prey_noise=0.1,      # 0.0 ... 0.5

    # ---- PPO -------------------------------------------------------------
    # These are a working point, not an optimum. The defaults below train an
    # agent that visibly hunts and beats two of the three house bots. Getting
    # past the third means changing something here, and the knobs are not
    # equally powerful -- one of them is worth more than all the others put
    # together on this task.
    learning_rate=3e-4,  # 1e-4 ... 1e-2   above that it stops converging
    n_steps=2048,        # 512 ... 8192    total rollout length before update
    batch_size=512,      # 64 ... 2048     keep it a divisor of n_steps
    n_epochs=10,         # 3 ... 20        passes over each batch of data
    gamma=0.95,          # 0.90 ... 0.995  how far ahead you care
    gae_lambda=0.95,     # 0.8 ... 1.0     bias/variance in the advantage
    clip_range=0.2,      # 0.1 ... 0.4     how far one update may move you
    ent_coef=0.01,       # 0.0 ... 0.05    exploration pressure
    net_arch=[64, 64],   # [32,32] ... [256,256]

    n_envs=8,
    seed=0,
)

# ==========================================================================
# STOP READING HERE (unless you want to)
# ==========================================================================

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from core import MATCH_STEPS
from env import HuntEnv
from weights import extract_mlp, check_matches_sb3
import engine

SUBMISSION_DIR = "submissions"

# PPO diagnostics worth watching, mapped to the names SB3 logs them under.
LOSS_KEYS = (
    ("value_loss",     "train/value_loss"),
    ("policy_loss",    "train/policy_gradient_loss"),
    ("entropy_loss",   "train/entropy_loss"),
    ("approx_kl",      "train/approx_kl"),
    ("clip_fraction",  "train/clip_fraction"),
    ("explained_var",  "train/explained_variance"),
)


class TrainingLog(BaseCallback):
    """Snapshots episode reward, catches, and PPO losses as training runs.

    Rows land in `.rows`; `plot_training` turns them into curves. The loss
    numbers come from SB3's logger, which is written by the previous rollout's
    `train()` call, so they lag the reward by one update -- close enough to
    read as a trend.
    """

    def __init__(self, every, quiet=False, score=True):
        super().__init__(verbose=0)
        self.every = max(1, int(every))
        self.quiet = quiet
        self.score = score       # also run the real benchmark (costs ~0.1s)
        self.rows = []
        self._next = self.every

    def _on_training_start(self):
        if not self.quiet:
            print(f"{'steps':>9}  {'reward':>8}  {'catches':>8}  "
                  f"{'SCORE':>7}  {'v_loss':>9}")

    def _on_step(self):
        if self.num_timesteps < self._next:
            return True
        self._next += self.every

        buf = self.model.ep_info_buffer
        if not buf:                       # no episode has finished yet
            return True

        row = {
            "steps":   self.num_timesteps,
            "reward":  float(np.mean([e["r"] for e in buf])),
            "catches": float(np.mean([e.get("catches", np.nan) for e in buf])),
        }
        for name, key in LOSS_KEYS:
            v = self.model.logger.name_to_value.get(key)
            row[name] = float(v) if v is not None else float("nan")

        # The number the leaderboard uses. The rows above measure the policy
        # while it samples its actions; the benchmark takes its best action
        # every time, which is a different agent. The benchmark is vectorised
        # and costs about 0.1s, so it is cheap to track alongside.
        row["score"] = float("nan")
        if self.score:
            row["score"] = float(engine.evaluate(extract_mlp(self.model))[0].mean())

        self.rows.append(row)

        if not self.quiet:
            print(f"{row['steps']:>9,}  {row['reward']:>8.2f}  "
                  f"{row['catches']:>8.1f}  {row['score']:>7.1f}  "
                  f"{row['value_loss']:>9.4f}")
        return True


def plot_training(log, title=None):
    """Draw the curves recorded by a TrainingLog. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    rows = log.rows if isinstance(log, TrainingLog) else list(log)
    if not rows:
        raise ValueError("nothing logged -- train for more steps than log_every")

    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    steps = col("steps")

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.suptitle(title or "training progress", fontsize=12)

    # What you optimise (reward) against what you are scored on (benchmark
    # catches). Two things pull them apart: a large shaping_coef makes reward
    # climb while catches do not, and the sampled policy always looks better
    # than the one you submit.
    a = ax[0, 0]
    a.plot(steps, col("reward"), color="#2b7bba", label="reward")
    a.set_ylabel("reward", color="#2b7bba")
    a.tick_params(axis="y", labelcolor="#2b7bba")
    a.set_title("reward vs. catches")
    a.set_xlabel("timesteps")

    b = a.twinx()
    b.plot(steps, col("catches"), color="#d1495b", ls="--",
           label="catches (sampled, training)")
    if np.isfinite(col("score")).any():
        b.plot(steps, col("score"), color="#d1495b", lw=2,
               label="SCORE (argmax, benchmark)")
    b.set_ylabel("catches", color="#d1495b")
    b.tick_params(axis="y", labelcolor="#d1495b")
    b.legend(fontsize=7, frameon=False, loc="lower right")

    a = ax[0, 1]
    a.plot(steps, col("value_loss"), color="#5f8a3a")
    a.set_title("value loss")
    a.set_xlabel("timesteps")
    a.set_yscale("log")

    a = ax[1, 0]
    a.plot(steps, col("policy_loss"), color="#8a5fa8", label="policy grad")
    a.plot(steps, col("entropy_loss"), color="#c98a2b", label="entropy")
    a.axhline(0, color="0.8", lw=.8)
    a.set_title("policy & entropy loss")
    a.set_xlabel("timesteps")
    a.legend(fontsize=8, frameon=False)

    a = ax[1, 1]
    a.plot(steps, col("explained_var"), color="#2b7bba", label="explained var")
    a.plot(steps, col("clip_fraction"), color="#d1495b", label="clip fraction")
    a.axhline(0, color="0.8", lw=.8)
    a.set_title("critic fit & clipping")
    a.set_xlabel("timesteps")
    a.legend(fontsize=8, frameon=False)

    for a in ax.ravel():
        a.grid(alpha=.25)
    fig.tight_layout()
    return fig


def make_env(cfg, rank):
    def _init():
        # info_keywords pulls the env's per-episode catch count into the
        # Monitor record, so the callback below can plot the thing you are
        # actually scored on next to the reward you are actually optimising.
        return Monitor(
            HuntEnv(
                shaping_coef=cfg["shaping_coef"],
                step_penalty=cfg["step_penalty"],
                prey_noise=cfg["prey_noise"],
                match_steps=MATCH_STEPS,
                seed=cfg["seed"] + rank,
            ),
            info_keywords=("catches",),
        )
    return _init


def build(cfg):
    venv = DummyVecEnv([make_env(cfg, i) for i in range(cfg["n_envs"])])
    return PPO(
        "MlpPolicy", venv,
        learning_rate=cfg["learning_rate"],
        n_steps=max(1, cfg["n_steps"] // cfg["n_envs"]),
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        policy_kwargs=dict(net_arch=cfg["net_arch"]),
        seed=cfg["seed"],
        verbose=0,
        # A tiny MLP on a 6-float observation is slower on a GPU than on a CPU,
        # and Colab hands out GPU runtimes freely. Pin it, or half the room
        # trains at half speed and wonders why.
        device="cpu",
    )


def make_logger(cfg, quiet=False, points=40):
    """A TrainingLog that samples ~`points` times over the whole run."""
    every = max(cfg["n_steps"], cfg["total_timesteps"] // points)
    return TrainingLog(every=every, quiet=quiet)


def main(name, cfg=None, progress_bar=False, plot=False):
    cfg = cfg or CONFIG
    model = build(cfg)

    log = make_logger(cfg)
    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"],
                progress_bar=progress_bar, callback=log)
    print(f"trained {cfg['total_timesteps']:,} steps in {time.time() - t0:.0f}s")

    if plot:
        import matplotlib.pyplot as plt
        plot_training(log, title=name)
        plt.savefig(f"{name}_training.png", dpi=120)
        print(f"wrote {name}_training.png")

    mlp = extract_mlp(model)
    agree = check_matches_sb3(model, mlp)
    if agree < 0.99:
        raise RuntimeError(f"weight extraction mismatch ({agree:.1%} agreement)")

    catches, _ = engine.evaluate(mlp)
    house, _ = engine.evaluate(engine.greedy_hunter)
    print(f"your score : {catches.sum():5d}  ({catches.mean():.1f} catches per episode)")
    print(f"house bot  : {house.sum():5d}  ({house.mean():.1f} catches per episode)")
    if catches.sum() <= house.sum():
        print("  ...the scripted bot is still ahead of you.")

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    path = os.path.join(SUBMISSION_DIR, f"{name}.npz")
    mlp.save(path)
    print(f"\nsaved {path} -- upload this file")
    return mlp


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python train.py YOURNAME [--plot]")
    main(sys.argv[1].strip().lower().replace(" ", "_"),
         plot="--plot" in sys.argv)
