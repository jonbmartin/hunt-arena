"""Generate train.ipynb for the workshop.

The notebook is generated, not hand-edited. Change it here and re-run:

    python make_notebook.py
"""
import json, os

# Where participants upload their .npz. Make a Google Form with a single
# File upload question, then paste its link here and re-run this script.
# Until it is set, the notebook tells people to ask you for the link.
FORM_URL = "https://forms.gle/omakKjbGMsEQjTQSA"

MD = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
CODE = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
MD("""
# Hunt Arena

You are a hunter on a 7x7 grid. A scripted prey runs away from you. Catch it as
many times as you can in 200 steps.

You will not write an RL algorithm. You will change numbers in **cell 2** and
find out what happens.

Run the cells in order. Cell 1 takes a minute; after that, cell 2 is the only
one you need to keep editing.
"""),

CODE("""
#@title 1. Setup — run this first { display-mode: "form" }
!pip -q install "stable-baselines3[extra]>=2.0" "gymnasium>=0.29"

REPO = "https://github.com/jonbmartin/hunt-arena"   #@param {type:"string"}

import os, sys, subprocess

def _arena_here():
    \"\"\"Are the workshop modules already importable? (local Jupyter, or a re-run)\"\"\"
    try:
        import engine  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False

if not _arena_here():
    if not os.path.isdir("arena"):
        # GIT_TERMINAL_PROMPT=0 so a bad URL fails immediately instead of
        # blocking on a username prompt that Colab cannot answer.
        r = subprocess.run(
            ["git", "clone", "--depth", "1", REPO, "arena"],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise SystemExit(
                f"Could not clone {REPO}\\n"
                f"  git said: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no output'}\\n\\n"
                "The REPO box above has to point at a PUBLIC GitHub repo.\\n"
                "An authentication prompt means the URL is wrong, or the repo is private."
            )
    os.chdir("arena")
    sys.path.insert(0, os.getcwd())

import engine
print("setup ok — the prey is waiting")
"""),

MD("""
## 2. The config — this is your playground

This is the only cell you need to edit. Everything else just runs.

A few things worth knowing before you start turning knobs:

* `shaping_coef` pays you every step for being **near** the prey, on top of the
  `+1` for catching it. It makes learning much faster. It is also not free.
* Catching the prey **teleports it away from you**. You do not move.
* You are always scored against a prey with `prey_noise = 0.1`, whatever you
  train against.
* These defaults work. They are a starting point, not a good answer — they
  will beat two of the three house bots and lose to the third. The knobs are
  not equally powerful; one of them is worth more than all the rest together.
"""),

CODE("""
CONFIG = dict(
    # ---- how long you train ------------------------------------------------
    total_timesteps=1_000_000,   # ~1.5 min. More is better, up to a point.

    # ---- reward ------------------------------------------------------------
    shaping_coef=0.05,   # per-step reward for being close. Try 0.0, 0.05, 0.5.
    step_penalty=0.0,    # constant per-step reward. Negative = hurry up.

    # ---- who you practise against -----------------------------------------
    prey_noise=0.1,      # chance the prey moves at random instead of fleeing

    # ---- PPO ---------------------------------------------------------------
    learning_rate=3e-4,
    n_steps=2048,        # rollout length before each update
    batch_size=512,
    n_epochs=10,
    gamma=0.95,          # how far ahead you care
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,       # exploration pressure; 0 tends to collapse early
    net_arch=[64, 64],

    n_envs=8,
    seed=0,
)

NAME = "yourname"        # no spaces — this becomes your submission filename
print(f"{CONFIG['total_timesteps']:,} steps, lr={CONFIG['learning_rate']}")
"""),

MD("""
## 3. Train

Watch the **SCORE** column. Three numbers are printed and they do not agree:

* `reward` is what PPO is actually maximising.
* `catches` is how many catches that policy gets **while training**, where it
  picks actions by sampling.
* `SCORE` is the benchmark your submission is graded on. It takes the policy's
  single best action every time — a different, harder agent.

`reward` and `catches` flatten out early. `SCORE` keeps climbing for a long
time afterwards. **If SCORE is still rising when training stops, you stopped
too early** — go back to cell 2 and raise `total_timesteps`.

A million steps takes a couple of minutes on a Colab CPU.
"""),

CODE("""
import importlib, train
importlib.reload(train)          # picks up CONFIG edits without a restart

model = train.build(CONFIG)
log   = train.make_logger(CONFIG)   # records reward, catches, SCORE, losses
model.learn(total_timesteps=CONFIG["total_timesteps"], callback=log)
print("done")
"""),

MD("""
## 4. How did training go?

The first plot is the one that matters. The dashed line is the policy PPO
trained; the solid line is what you are actually scored on. A gap between them
is normal — a solid line that is still climbing means you should train longer.

The other three are diagnostics:

* **value loss** — the critic learning to predict returns.
* **policy & entropy loss** — entropy climbing toward zero means the policy is
  getting decisive. If it slams into zero early, it collapsed before it
  learned anything.
* **critic fit & clipping** — `explained var` near 1 is a critic that
  understands the task. A large `clip fraction` means most of your updates are
  being clipped away, so lower `learning_rate`.
"""),

CODE("""
fig = train.plot_training(log, title=NAME)
"""),

MD("""
## 5. Score it and save your submission

`mlp` is your policy converted to plain numpy arrays — that is what you
upload. Three house bots are shown for scale.
"""),

CODE("""
import os
import engine
from weights import extract_mlp, check_matches_sb3
from check_submission import check

if NAME.strip().lower() in ("", "yourname"):
    raise SystemExit("Set NAME in cell 2 to your own name first — "
                     "twenty files called yourname.npz all collide.")
NAME = NAME.strip().lower().replace(" ", "_")

mlp = extract_mlp(model)
assert check_matches_sb3(model, mlp) > 0.99, "weight extraction mismatch"

you, _    = engine.evaluate(mlp)
greedy, _ = engine.evaluate(engine.greedy_hunter)
sloppy, _ = engine.evaluate(engine.sloppy_hunter)
rand, _   = engine.evaluate(engine.random_hunter)

print(f"  you            {you.mean():5.1f} catches per episode")
print(f"  [bot] greedy   {greedy.mean():5.1f}")
print(f"  [bot] sloppy   {sloppy.mean():5.1f}")
print(f"  [bot] random   {rand.mean():5.1f}")

os.makedirs("submissions", exist_ok=True)
path = f"submissions/{NAME}.npz"
mlp.save(path)
check(path)

try:                              # hand the file to your browser
    from google.colab import files
    files.download(path)
    print(f"\\n{NAME}.npz is in your Downloads. Upload it — link in the cell below.")
except Exception:
    print(f"(not on Colab) submission is at {path}")
"""),


MD("""
## 6. Watch it play

The blue circle is your agent. The amber diamond is the prey. If your agent
shadows the prey without ever closing, that is not a bug — work out what you
paid it to do.
"""),

CODE("""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

import engine
from core import GRID, WALLS

bench = engine.Benchmark(n_episodes=1)
_, tr = engine.evaluate(mlp, bench, record_episode=0)
H, P = np.array(tr["hunter"]), np.array(tr["prey"])

fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.set_xlim(-.5, GRID - .5); ax.set_ylim(GRID - .5, -.5)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
for r, c in WALLS:
    ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="0.78"))
hunter, = ax.plot([], [], "o", ms=19, color="#2b7bba", label="hunter (you)")
prey,   = ax.plot([], [], "D", ms=13, color="#ffb300", label="prey")
title = ax.set_title("")
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.01), ncol=2, frameon=False)

def frame(t):
    hunter.set_data([H[t, 1]], [H[t, 0]])
    prey.set_data([P[t, 1]], [P[t, 0]])
    title.set_text(f"step {t}    catches {int(np.sum(tr['caught'][:t]))}")
    return hunter, prey, title

anim = animation.FuncAnimation(fig, frame, frames=len(H), interval=110)
plt.close(fig)
HTML(anim.to_jshtml())
"""),

MD("""
## 7. Submit

Cell 5 downloaded **`<yourname>.npz`** to your machine. Upload it here:

### __FORM_URL__

Submit when you are happy with your agent. If you submit more than once we
take your most recent one.

Then go back to **cell 2** and try to beat it — you can keep training and
resubmit right up to the deadline.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.ipynb")
text = json.dumps(nb, indent=1)
if FORM_URL.startswith("PASTE_"):
    text = text.replace("__FORM_URL__", "(ask your host for the upload link)")
    print("!! FORM_URL is not set — the notebook has no upload link in it")
else:
    text = text.replace("__FORM_URL__", FORM_URL)
with open(out, "w") as f:
    f.write(text)
print("wrote", out, len(cells), "cells")
