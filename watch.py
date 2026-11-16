"""See what the agent learned after N episodes.
    python watch.py --algo sarsa --episodes 20000                 # learning curve + one greedy game, move by move
    python watch.py --algo sarsa --episodes 200000 --trace-seed 7 # same deal, compare with a smaller --episodes
Per decision it prints your real hand, the board, and the learned value Q(s,a) of every legal action (* = chosen).
Q is the expected final reward (+1 win, -1 loss): along a won game it should climb towards +1, along a lost one fall towards -1.
Unseen / never-tried actions show +0.000.
    python watch.py --algo sarsa --episodes 200000 --html      # animated replay: writes replay_sarsa_200000.html (2 players)"""
import argparse
import json
import os
from collections import Counter

from algos import ALGOS
from algos.common import train
from env import UnoEnv
from opponents import OPPONENTS
from run import parse_rules

VAL = [str(i) for i in range(10)] + ["Skip", "Rev", "+2", "Wild", "+4"]


def card(c):
    return ("RYGB"[c[0]] if c[0] < 4 else "W") + VAL[c[1]]


def record(env, ag, name):
    """Play one greedy game and return the event list the HTML viewer replays (deal / decide / move / end)."""
    ev, hands = [], []
    ab = env.abs

    def snap(g):
        return dict(hand=[list(c) for c in g.hands[0]], opp=[len(h) for h in g.hands[1:]], top=[g.top_color, g.top_value],
                    pend=g.pend, deck=len(g.deck))

    def obs(g, p, mv):
        if mv is None:
            ev.append(dict(t="deal", s=snap(g)))
        else:
            e = dict(t="move", p=p, k=mv[0], pend=0, s=snap(g))
            if mv[0] == "play":
                e.update(card=list(mv[1]), color=mv[2])
            elif mv[0] == "draw":
                new = Counter(map(tuple, g.hands[p])) - Counter(hands[p])
                e.update(drawn=[list(c) for c in new.elements()], n=sum(new.values()), pend=int(hands_pend[0]))
            ev.append(e)
        hands[:] = [list(h) for h in g.hands]
        hands_pend[0] = g.pend

    hands_pend = [0]
    env.observer = obs
    s, legal = env.reset()
    done = False
    while not done:
        row, act = ag.q(s), ag.select(s, legal, greedy=True)
        cq = {}                                   # per hand card: best Q of the moves that play it (wild: best colour)
        for x in legal:
            for m in env.moves[x]:
                if m[0] == "play":
                    k = f"{m[1][0]},{m[1][1]}"
                    if k not in cq or row[x] > cq[k]["q"]:
                        cq[k] = dict(q=round(row[x], 4), label=name(x), col=m[2] if m[1][0] == 4 else None, chosen=x == act)
        ev.append(dict(t="decide", s=snap(env.game), opts=[dict(label=name(x), q=round(row[x], 4), chosen=x == act) for x in legal],
                       cq=cq, drawq=round(row[ab.DRAW], 4) if ab.DRAW in legal else None,
                       passq=round(row[ab.PASS], 4) if ab.PASS in legal else None, drawchosen=act == ab.DRAW, passchosen=act == ab.PASS))
        s, legal, r, done = env.step(act)
    g = env.game
    why = "draw" if g.winner is None else "emptied" if not g.hands[g.winner] else "limit"   # how it ended
    ev.append(dict(t="end", r=r, why=why, s=snap(g)))
    env.observer = None
    return ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="sarsa", choices=ALGOS)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--opp", default="rule", choices=OPPONENTS, help="opponent for training and for the shown game")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace-seed", type=int, default=7, help="which deal to show (same number = same first deal)")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--html", nargs="?", const="", metavar="FILE", help="write an animated replay (default replay_<algo>_<N>.html)")
    p.add_argument("--rule", action="append", default=[])
    a = p.parse_args()
    rules = parse_rules(a.rule)
    opp = OPPONENTS[a.opp]
    ag, curve, secs = train(ALGOS[a.algo], UnoEnv(rules, opp, a.seed), a.episodes, a.seed, max(a.episodes // 10, 1))
    print(f"{a.algo}: {a.episodes} episodes vs {a.opp} in {secs:.1f}s, {len(ag.Q)} states learned, tau={ag.tau:.2f}\n")
    print("learning curve (win rate in training, per block; the agent is still exploring):")
    for ep, w, l, tau, n in curve:
        print(f"  {ep:>8}  {w:5.1%} {'#' * round(w * 40)}")

    env = UnoEnv(rules, opp, a.trace_seed)
    ab = env.abs

    def name(x):
        if x == ab.DRAW:
            return "draw"
        if x == ab.PASS:
            return "pass"
        k, r = divmod(x, ab.nc)
        return f"{ab.kinds[k]} (colour#{r + 1})"       # colour#1 = the colour you hold most

    if a.html is not None:
        assert rules.n_players == 2, "the HTML replay shows one opponent"
        meta = dict(algo=a.algo, episodes=a.episodes, opp=a.opp, states=len(ag.Q),
                    rules=f"{rules.n_colors} colours, numbers {list(rules.numbers)}, {'/'.join(rules.specials)}, hand limit {rules.hand_limit or 'none'}")
        path = a.html or f"replay_{a.algo}_{a.episodes}.html"
        tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_template.html"), encoding="utf8").read()
        data = json.dumps(dict(meta=meta, events=record(env, ag, name)))
        open(path, "w", encoding="utf8").write(tpl.replace("/*DATA*/null", data.replace("</", "<\\/")))
        print("wrote", path)
        return

    for g in range(a.games):
        print(f"\n===== greedy game {g + 1} (no learning) =====")
        s, legal = env.reset()
        done, step, qs = False, 0, []
        while not done:
            gm = env.game
            print(f"\n[{step}] board {card((gm.top_color, gm.top_value))}  pending draw {gm.pend}  "
                  f"opponent holds {[len(h) for h in gm.hands[1:]]}  you hold {len(gm.hands[0])}")
            print("    your hand:", " ".join(card(c) for c in sorted(gm.hands[0])))
            row, act = ag.q(s), ag.select(s, legal, greedy=True)
            for x in sorted(legal, key=lambda x: -row[x]):
                print(f"    {'*' if x == act else ' '} {name(x):22}{row[x]:+.3f}")
            qs.append(row[act])
            s, legal, r, done = env.step(act)
            step += 1
        print(f"\nresult: {'WIN' if r == 1 else 'LOSS' if r == -1 else 'draw'} (reward {r:+d})")
        print("Q of the chosen move at each decision:", " ".join(f"{q:+.2f}" for q in qs))


if __name__ == "__main__":
    main()
