# Hunt Arena

A competitive reinforcement learning exercise. You train an agent to catch a
fleeing prey on a 7×7 grid, submit it, and watch every agent in the room run
the same episode side by side.

You do not write an RL algorithm. You tune a configuration block, look at what
your agent learned, and try to do better.

## The task

A hunter and a scripted prey share a 7×7 grid with six walls. You control the
hunter.

- The hunter moves in **8 directions**, the prey in **4**. Without that
  asymmetry the gap between them never closes and the prey is never caught.
- A catch is landing on the prey, or swapping places with it.
- On a catch you score **+1** and the prey respawns at least 3 cells away.
  **You do not move** — so catching costs you your proximity.
- An episode is **200 steps**. Your score is the number of catches.

The prey is a fixed script rather than a learning opponent, so the environment
is stationary and every submission is scored on the same fixed benchmark: the
same 200 starting positions and the same pre-drawn prey randomness. Two agents
that behave identically score identically, and your rank never depends on who
else entered.

**Observation** — 6 floats: your position, the prey's position, and the delta
between them.
**Actions** — 9: eight directions plus stay.

## Getting started

Open `train.ipynb` in Google Colab and run the cells in order. Nothing to
install.

To run it on your own machine instead:

```bash
pip install "stable-baselines3[extra]>=2.0" "gymnasium>=0.29"
python train.py yourname
python check_submission.py submissions/yourname.npz
```

## Reading your training run

Training prints a table and then draws four plots. The first column pair is
the one to watch:

| column | what it is |
|---|---|
| `reward` | what PPO is maximising, including any shaping you added |
| `catches` | catches while training, where the policy **samples** its actions |
| `SCORE` | the benchmark you are graded on, where the policy takes its **best** action |

`reward` and `catches` flatten out early. `SCORE` keeps improving long after
they do, so **if `SCORE` is still rising when training stops, train longer.**

The remaining plots are diagnostics: value loss (the critic learning to predict
returns), entropy (falling toward zero as the policy becomes decisive), and
explained variance and clip fraction (how well the critic fits, and how much of
each update is being clipped away).

## Submitting

The notebook scores your agent, validates the file, and downloads
`yourname.npz`. Upload that file wherever your host has asked you to.

You can submit as often as you like — the most recent one counts. Go back to
the config cell and try to beat yourself.

Submissions are `.npz` files containing only float arrays. Nothing is unpickled
and no submitted code is executed.

## Reference points

Three scripted bots sit on the leaderboard as fixed rungs:

| bot | catches per episode |
|---|---|
| `[bot] greedy` | 24.5 |
| `[bot] sloppy` | 15.3 |
| `[bot] random` | 0.6 |

Getting past `greedy` means genuinely tuning something.

Scoring is noisy at the margin: per-episode standard deviation is 8–10 catches
for a learned agent, so a mean over 200 episodes carries a standard error of
about 0.6. Gaps above ~2 catches per episode are real; ties inside 1 are not.

## The race

Every agent runs the *identical* episode — same start, same prey, same
randomness — each on its own small board, all animating at once. The agents
never interact; you are watching the difference between policies, not a fight.
Boards re-sort live by score.

The prey is an amber diamond on every board; only the hunter takes the agent's
colour. Click any board to open it full screen.

## Files

| file | what it does |
|---|---|
| `train.ipynb` | the Colab notebook — the main way to do the exercise |
| `train.py` | the same thing as a command-line script |
| `check_submission.py` | validates a submission before you send it |
| `leaderboard.py` | scores every submission and builds the race |
| `race.py` | builds the race animation |
| `env.py` | the Gymnasium environment |
| `engine.py` | vectorised benchmark and the scripted bots |
| `core.py` | rules and geometry |
| `weights.py` | policy → numpy conversion, and the submission format |
| `make_notebook.py` | regenerates `train.ipynb` |

`train.ipynb` is generated. Edit `make_notebook.py` and re-run it rather than
editing the notebook by hand.

## Running the session (hosts)

Collect the `.npz` files into a folder and run:

```bash
python leaderboard.py submissions/ --race race.html
```

Then open `race.html` — every submission runs the same episode side by side.

## Verifying an install

```bash
python test_arena.py
```

Checks the rules, the engine, the submission format, and benchmark fairness.
It runs without `gymnasium` installed.
