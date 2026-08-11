"""
Score every submission and build the ghost race.

    python leaderboard.py submissions/                  # table
    python leaderboard.py submissions/ --race race.html # + the animation
    python leaderboard.py submissions/ --csv out.csv

Every agent plays the identical benchmark -- same starts, same prey
randomness -- so the ranking depends only on the policies, not on who entered
or in what order.
"""

import argparse
import glob
import os
import re
import time

import numpy as np

from weights import NumpyMLP
from check_submission import check
import engine
import race as race_mod

# Three fixed reference points, so the leaderboard reads as a ladder rather
# than a list: beat random, then beat sloppy, then beat greedy.
HOUSE_BOTS = {
    "[bot] greedy": engine.greedy_hunter,
    "[bot] sloppy": engine.sloppy_hunter,
    "[bot] random": engine.random_hunter,
}


def safe_name(path):
    """Agent name from the filename, stripped to characters we can trust.

    The name is the only participant-controlled string that reaches the race
    HTML, so it is sanitised here rather than escaped at every use site. Also
    keeps a pathological filename from wrecking the grid layout.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    clean = re.sub(r"[^A-Za-z0-9 _-]", "", stem).strip()[:24]
    return clean or "anon"


def load_all(folder):
    entrants, rejected = {}, []
    seen = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        name = safe_name(path)
        if name in seen or name in HOUSE_BOTS:   # sanitising can create collisions
            seen[name] = seen.get(name, 1) + 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        problems, _ = check(path, verbose=False)
        if problems:
            rejected.append((name, problems[0]))
        else:
            entrants[name] = NumpyMLP.load(path)
    return entrants, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--episodes", type=int, default=engine.EVAL_EPISODES)
    ap.add_argument("--race", default=None, help="write the ghost race here")
    ap.add_argument("--race-episode", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    entrants, rejected = load_all(args.folder)
    for name, why in rejected:
        print(f"  REJECTED {name}: {why}")
    field = {**entrants, **HOUSE_BOTS}
    print(f"{len(entrants)} submissions + {len(HOUSE_BOTS)} house bots")

    bench = engine.Benchmark(n_episodes=args.episodes)
    t0 = time.time()
    results, traces = {}, {}
    for name, pol in field.items():
        catches, tr = engine.evaluate(pol, bench, record_episode=args.race_episode)
        results[name] = catches
        traces[name] = tr
    print(f"scored {len(field)} agents on {args.episodes} episodes "
          f"in {time.time() - t0:.1f}s")

    rows = sorted(results, key=lambda n: (-results[n].sum(), n))
    width = max(len(r) for r in rows) + 2
    print(f"\n{'#':>3}  {'agent':<{width}}{'total':>8}{'per ep':>9}{'sd':>7}")
    print("-" * (27 + width))
    for i, name in enumerate(rows, 1):
        c = results[name]
        print(f"{i:>3}  {name:<{width}}{c.sum():>8d}{c.mean():>9.1f}{c.std():>7.1f}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("rank,agent,total_catches,mean_per_episode,sd\n")
            for i, n in enumerate(rows, 1):
                c = results[n]
                f.write(f"{i},{n},{c.sum()},{c.mean():.3f},{c.std():.3f}\n")
        print(f"wrote {args.csv}")

    if args.race:
        entries = [{
            "name": n,
            "bot": n in HOUSE_BOTS,
            "per_ep": round(float(results[n].mean()), 2),
            "hunter": traces[n]["hunter"],
            "prey": traces[n]["prey"],
            "caught": traces[n]["caught"],
        } for n in rows]
        race_mod.build_html(entries, args.race, args.episodes)
        print(f"wrote {args.race} -- open it in a browser")


if __name__ == "__main__":
    main()
