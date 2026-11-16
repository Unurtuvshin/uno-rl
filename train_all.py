"""Train the whole model library once:  python train_all.py [--jobs 6] [--sizes 10000,20000,...]
5 methods x 2 training opponents x 7 sizes -> models/<algo>_<opp>_<episodes>.json (resumable: existing files are skipped).
Each size is its own run (the exploration schedule ends at that size).  Each file holds the Q table (int16, Q*10000, for
bucket brigade the bid c*Q) and the greedy win rate vs both bots.  uno_player.html is built from these with build_page.py."""
import argparse
import base64
import json
import os
import sys
import time
from array import array
from concurrent.futures import ProcessPoolExecutor

from algos import ALGOS
from algos.common import evaluate, train
from env import UnoEnv
from game import Rules
from opponents import OPPONENTS

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
EVAL = 2000


def pack(ag, ranges):
    idx, val = array("H"), array("h")
    for s, row in ag.Q.items():
        if not any(row):
            continue
        i = 0
        for x, m in zip(s, ranges):
            i = i * m + x
        idx.append(i)
        val.extend(max(-32000, min(32000, round(v * ag.scale * 10000))) for v in row)
    if sys.byteorder == "big":
        idx.byteswap()
        val.byteswap()
    return base64.b64encode(idx.tobytes()).decode(), base64.b64encode(val.tobytes()).decode()


def job(cfg):
    algo, opp, n, out = cfg
    path = os.path.join(out, f"{algo}_{opp}_{n}.json")
    if os.path.exists(path):
        return path, "skipped"
    rules, seed = Rules(), list(ALGOS).index(algo)
    ag, _, secs = train(ALGOS[algo], UnoEnv(rules, OPPONENTS[opp], seed), n, seed, max(n // 10, 1))
    ev = {}
    for name, o in OPPONENTS.items():
        w, d, l = evaluate(ag, UnoEnv(rules, o, seed + 10_000), EVAL)
        ev[name] = dict(win=w, draw=d, loss=l)
    ranges = UnoEnv(rules).abs.ranges
    idx, val = pack(ag, ranges)
    with open(path, "w") as f:
        json.dump(dict(algo=algo, opp=opp, episodes=n, n_states=len(ag.Q), secs=round(secs, 1), eval=ev, eval_games=EVAL,
                       ranges=list(ranges), n_actions=ag.n_actions, idx=idx, val=val), f)
    return path, f"{secs:.0f}s, vs random {ev['random']['win']:.2f}, vs rule {ev['rule']['win']:.2f}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--sizes", default=",".join(map(str, SIZES)))
    p.add_argument("--algo", default="all")
    a = p.parse_args()
    out = os.path.join(HERE, "models")
    os.makedirs(out, exist_ok=True)
    algos = list(ALGOS) if a.algo == "all" else a.algo.split(",")
    cfgs = [(al, o, int(n), out) for n in sorted(map(int, a.sizes.split(",")), reverse=True) for al in algos for o in OPPONENTS]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (path, msg) in enumerate(ex.map(job, cfgs), 1):
            print(f"[{i}/{len(cfgs)}] {os.path.basename(path)}  {msg}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
