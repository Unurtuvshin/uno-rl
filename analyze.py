"""Behaviour of the trained policies.   python analyze.py [--vs rule] [--games 3000] [--glob "results/*_q.json"]
Loads every results/<tag>_q.json (+ its .json for rules), plays greedy games and counts what the agent does in the
situations that matter (slide 11).  Prints a table and writes results/analysis.csv.  Percentages are over decisions."""
import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict

from algos import ALGOS
from algos.common import Agent
from env import UnoEnv
from game import Rules
from opponents import OPPONENTS

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = [("d4_any", "+4 played when legal (%)"), ("d4_opp_low", "  ... opponent <=2 cards (%)"),
        ("d4_opp_high", "  ... opponent >2 cards (%)"), ("d4_best_col", "+4 names own most-held colour (%)"),
        ("stack", "stacked when allowed (%)"), ("draw_free", "drew although a card was playable (%)"),
        ("play_drawn", "played the card just drawn (%)"), ("hand_ge8", "decisions with >=8 cards (%)")]


def load_rules(d):
    t = lambda x: tuple(map(t, x)) if isinstance(x, list) else x     # json lists -> tuples (stacking pairs etc.)
    return Rules(**{k: t(v) for k, v in d.items()})


def analyse(q_path, vs, games, seed=123):
    meta = json.load(open(q_path.replace("_q.json", ".json")))
    sm = meta["summary"]
    rules = load_rules(meta["rules"])
    ag = Agent(ALGOS[sm["algo"]], UnoEnv(rules).n_actions, seed, **meta["hp"])
    ag.Q = {eval(k): v for k, v in json.load(open(q_path)).items()}     # keys were written with repr(); our own files
    env = UnoEnv(rules, OPPONENTS[vs], seed)
    ab, nc = env.abs, env.abs.nc
    d4 = ab.kinds.index("D4") if "D4" in ab.kinds else None
    cnt = defaultdict(lambda: [0, 0])                                   # name -> [hits, opportunities]

    def tally(name, hit, ok=True):
        if ok:
            cnt[name][0] += bool(hit)
            cnt[name][1] += 1

    for _ in range(games):
        s, legal = env.reset()
        done = False
        while not done:
            a = ag.select(s, legal, greedy=True)
            own, opp, _, pend, drawn, bits = s
            low = opp <= (1 if ab.lim else 1)                           # bucket 0/1 = opponent holds 1-2 cards (lim: o-1)
            has_d4 = d4 is not None and any(x // nc == d4 for x in legal if x < ab.DRAW)
            play_d4 = a < ab.DRAW and d4 is not None and a // nc == d4
            tally("d4_any", play_d4, has_d4 and not pend)
            tally("d4_opp_low", play_d4, has_d4 and not pend and low)
            tally("d4_opp_high", play_d4, has_d4 and not pend and not low)
            tally("d4_best_col", play_d4 and a % nc == 0, play_d4)
            plays = [x for x in legal if x < ab.DRAW]
            tally("stack", a < ab.DRAW, pend and plays)
            tally("draw_free", a == ab.DRAW, not pend and not drawn and plays and ab.DRAW in legal)
            tally("play_drawn", a < ab.DRAW, drawn)
            tally("hand_ge8", (own + 1) >= 8 if ab.lim else own >= 4, True)
            s, legal, _, done = env.step(a)
    return sm, {k: (100 * h / n if n else None) for k, (h, n) in cnt.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vs", default="rule", choices=OPPONENTS)
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--glob", default=os.path.join(HERE, "results", "*_q.json"))
    a = p.parse_args()
    rows = []
    for f in sorted(glob.glob(a.glob)):
        sm, r = analyse(f, a.vs, a.games)
        rows.append((sm["tag"], r))
    names = [c for c, _ in COLS]
    fmt = lambda v: "   n/a" if v is None else f"{v:6.1f}"
    print(f"greedy play vs {a.vs}, {a.games} games each\n{'tag':40}" + "".join(f"{n:>13}" for n in names))
    for tag, r in rows:
        print(f"{tag:40}" + "".join(f"{fmt(r.get(n)):>13}" for n in names))
    print("\n" + "\n".join(f"{n:13}= {label}" for n, label in COLS))
    with open(os.path.join(HERE, "results", "analysis.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag"] + names)
        w.writerows([[t] + [r.get(n) for n in names] for t, r in rows])
