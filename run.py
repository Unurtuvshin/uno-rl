"""Train and evaluate.   python run.py --algo all --episodes 20000 --opp random --jobs 5
--opp random+rule trains vs random, then continues vs rule-based.  Rule changes: --rule plus4_needs_no_color=True --rule "stacking=()" --rule n_players=3
Every result is saved as results/<tag>.csv (learning curve) + <tag>.json (rules, hyper-parameters, summary incl. win rate
vs both opponents) + <tag>_q.json (Q table).
The tag contains a hash of the rules, so runs under different rules never overwrite each other."""
import argparse
import ast
import csv
import dataclasses
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

from algos import ALGOS
from algos.common import evaluate, train
from env import UnoEnv
from game import Rules
from opponents import OPPONENTS

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_rules(items):
    fields = {f.name for f in dataclasses.fields(Rules)}
    kv = {}
    for it in items:
        k, v = it.split("=", 1)
        assert k in fields, f"unknown rule {k!r}; choose from {sorted(fields)}"
        kv[k] = ast.literal_eval(v)
    return Rules(**kv)


def run_one(cfg):
    """Top-level (picklable) worker. Returns a plain dict, never the agent."""
    rules = Rules(**{**dataclasses.asdict(Rules()), **cfg["rules"]})
    rules = dataclasses.replace(rules, stacking=tuple(map(tuple, rules.stacking)))
    opps = [OPPONENTS[n] for n in cfg["opp"].split("+")]       # "random+rule" = curriculum
    opp = opps[-1]
    algo = ALGOS[cfg["algo"]]
    seed = cfg["seed"]
    ag, curve, secs = train(algo, [UnoEnv(rules, o, seed + i) for i, o in enumerate(opps)], cfg["episodes"], seed, cfg["window"])
    ev = {name: evaluate(ag, UnoEnv(rules, o, seed + 10_000), cfg["eval"]) for name, o in OPPONENTS.items()}
    w, d, l = ev[cfg["opp"].split("+")[-1]]
    rh = hashlib.md5(json.dumps(dataclasses.asdict(rules), sort_keys=True).encode()).hexdigest()[:6]
    tag = f"{cfg['algo']}_{cfg['opp']}_s{seed}_r{rh}"
    os.makedirs(cfg["out"], exist_ok=True)
    with open(os.path.join(cfg["out"], tag + ".csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["episode", "train_win", "train_loss", "tau", "n_states"])
        wr.writerows(curve)
    summary = dict(tag=tag, algo=cfg["algo"], opp=cfg["opp"], seed=seed, episodes=cfg["episodes"],
                   secs=round(secs, 1), games_per_s=round(cfg["episodes"] * len(opps) / secs),
                   eval_games=cfg["eval"], win=w, draw=d, loss=l, n_states=len(ag.Q),
                   eval_vs={k: dict(win=a, draw=b, loss=c) for k, (a, b, c) in ev.items()})
    with open(os.path.join(cfg["out"], tag + ".json"), "w") as f:
        json.dump(dict(summary=summary, rules=dataclasses.asdict(rules), hp=ag.hp), f, indent=1)
    with open(os.path.join(cfg["out"], tag + "_q.json"), "w") as f:     # learned values, for the behaviour analysis
        json.dump({repr(k): v for k, v in ag.Q.items()}, f)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="sarsa", help="|".join(ALGOS) + "|all")
    p.add_argument("--opp", default="random", help="random | rule | random+rule (train vs random, then continue vs rule; --episodes per stage)")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--eval", type=int, default=10000, help="evaluation games vs EACH opponent")
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--rule", action="append", default=[], help="key=value, repeatable")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    a = p.parse_args()
    assert all(n in OPPONENTS for n in a.opp.split("+")), f"--opp must be made of {list(OPPONENTS)} joined by +"
    rules = dataclasses.asdict(parse_rules(a.rule))
    names = list(ALGOS) if a.algo == "all" else [a.algo]
    cfgs = [dict(algo=n, opp=a.opp, episodes=a.episodes, eval=a.eval, window=a.window, seed=a.seed + i,
                 rules=rules, out=a.out) for i, n in enumerate(names)]   # distinct seed per job
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            out = list(ex.map(run_one, cfgs))
    else:
        out = [run_one(c) for c in cfgs]
    print(f"{'algo':16}{'opp':13}{'train s':>8}{'games/s':>9}{'win':>7}{'draw':>7}{'loss':>7}{'states':>8}")
    for s in out:
        print(f"{s['algo']:16}{s['opp']:13}{s['secs']:8}{s['games_per_s']:9}{s['win']:7.3f}{s['draw']:7.3f}"
              f"{s['loss']:7.3f}{s['n_states']:8}")
