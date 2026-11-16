"""Proof that web/engine.js behaves exactly like the Python:  python check_js.py   (needs node)
Python plays games (random / rule-based seats, default rules and the full 108-card rules) and records every shuffle, every
random() the rule bot used, every legal-move set, every encoded state and the state after every move.  node replays them."""
import dataclasses
import json
import os
import random
import subprocess
import sys

from env import Abstraction
from game import Game, Rules
from opponents import OPPONENTS

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = dict(n_colors=4, numbers=tuple(range(10)), specials=("SKIP", "REV", "D2", "WILD", "D4"), hand_limit=0)
VARIANTS = {"default": {}, "12-card limit, numbers 1-2": dict(numbers=(1, 2), hand_limit=12), "full": FULL, "no stacking, official +4": dict(stacking=(), plus4_needs_no_color=True, **FULL),
            "no draw if playable": dict(draw_even_if_playable=False)}


class Rec(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.shuffles, self.randoms = [], []

    def shuffle(self, x):
        super().shuffle(x)
        self.shuffles.append([list(c) for c in x])

    def random(self):
        v = super().random()
        self.randoms.append(v)
        return v


def snap(g):
    return dict(hands=[sorted(map(list, h)) for h in g.hands], top=[g.top_color, g.top_value], pend=g.pend, kind=g.pend_kind,
                deck=len(g.deck), disc=len(g.discard), cur=g.cur, drawn=list(g.drawn) if g.drawn else None, done=g.done, winner=g.winner)


def canon(m):
    return json.dumps([m[0]] + ([list(m[1]), m[2]] if m[0] == "play" else []), separators=(",", ":"))


def play(rules, seed, opp1, start):
    rng = Rec(seed)
    g = Game(rules, rng, start)
    ab = Abstraction(rules)
    steps = []
    while not g.done:
        legal = g.legal_moves(g.cur)
        rec = dict(seat=g.cur, legal=sorted(map(canon, legal)), order=list(map(canon, legal)))   # order: Python's set order, which the rule bot's noise follows
        s, by = ab.encode(g, g.cur)                     # the state as seen by the seat to move, either seat
        rec.update(state=list(s), actions=sorted(by))
        if g.cur == 0:
            mv = rng.choice(legal)                      # seat 0 plays randomly
        else:
            n0 = len(rng.randoms)
            mv = OPPONENTS[opp1](g, g.cur, legal, rng)
            rec.update(noise=rng.randoms[n0:], opp=opp1)
        rec["move"] = canon(mv)
        g.apply(mv)
        rec["after"] = snap(g)
        steps.append(rec)
    return dict(seed=seed, start=start, shuffles=rng.shuffles, steps=steps, initial_cards=g.n_cards)


if __name__ == "__main__":
    cases = []
    for name, kv in VARIANTS.items():
        rules = Rules(**kv)
        d = dataclasses.asdict(rules)
        for i in range(60):
            cases.append(dict(variant=name, rules=d, game=play(rules, 1000 + i, ["random", "rule"][i % 2], i % 2)))
    path = os.path.join(HERE, "web", "parity.json")
    json.dump(cases, open(path, "w"))
    r = subprocess.run(["node", os.path.join(HERE, "web", "parity_test.js"), path], capture_output=True, text=True, encoding="utf8", errors="replace")
    print(r.stdout, r.stderr)
    os.remove(path)
    sys.exit(r.returncode)
