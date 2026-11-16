"""Self-check.  python test_env.py   (about a minute)"""
import dataclasses
import random

from algos import ALGOS
from algos.common import evaluate, train
from env import UnoEnv
from game import Game, Rules
from opponents import OPPONENTS, random_opp

FULL = dict(n_colors=4, numbers=tuple(range(10)), specials=("SKIP", "REV", "D2", "WILD", "D4"), hand_limit=0)
VARIANTS = {
    "default (team's rule sheet)": {},
    "full 108-card UNO": FULL,
    "no hand limit": dict(hand_limit=0),
    "no stacking": dict(stacking=(), **FULL),
    "official +4": dict(plus4_needs_no_color=True, **FULL),
    "no draw if playable": dict(draw_even_if_playable=False),
    "auto-play drawn": dict(auto_play_drawn=True),
    "3 players": dict(n_players=3, **FULL),
    "4 players, no stacking": dict(n_players=4, stacking=(), **FULL),
}


def play_bare(rules, n, seed, policy):
    """Plain game loop, same policy in every seat. Checks the card count after EVERY applied move."""
    rng = random.Random(seed)
    wins = [0] * rules.n_players
    for i in range(n):
        g = Game(rules, rng, start=i % rules.n_players)
        while not g.done:
            g.apply(policy(g, g.cur, g.legal_moves(g.cur), rng))
            assert g.total_cards() == g.n_cards, "cards lost or duplicated"
            assert all(h for h in g.hands) or g.done
        if g.winner is not None:
            wins[g.winner] += 1
    return wins


def test_rules_terminate_and_conserve():
    for name, kv in VARIANTS.items():
        for pol in OPPONENTS.values():
            play_bare(Rules(**kv), 150, 1, pol)
        print("ok  rules:", name)


def test_symmetry():
    w = play_bare(Rules(), 4000, 7, random_opp)       # same policy in both seats, alternating start
    assert 0.46 < w[0] / sum(w) < 0.54, w
    print("ok  random vs random win rate", round(w[0] / sum(w), 3))


def test_mdp_matches_deck():
    r = Rules(**FULL)
    env = UnoEnv(r, random_opp, 3)
    assert env.n_actions == 22 and env.abs.n_states_max == 20480     # slides 5-7 (full deck)
    r = Rules()
    env = UnoEnv(r, random_opp, 3)
    ranges = env.abs.ranges
    rng = random.Random(0)
    seen = set()
    for _ in range(300):
        s, legal = env.reset()
        done = False
        while not done:
            assert all(0 <= x < m for x, m in zip(s, ranges)), s
            assert legal and all(0 <= a < env.n_actions for a in legal)
            seen.add(s)
            s, legal, rew, done = env.step(rng.choice(legal))
        assert s is None and legal == [] and rew in (-1, 0, 1)
    print("ok  MDP matches deck; distinct states seen in 300 random games:", len(seen))


def test_reproducible():
    out = []
    for _ in range(2):
        ag, curve, _ = train(ALGOS["sarsa"], UnoEnv(Rules(), random_opp, 5), 500, 5, 100)
        out.append((curve, evaluate(ag, UnoEnv(Rules(), random_opp, 99), 100)))
    assert [c[:4] for c in out[0][0]] == [c[:4] for c in out[1][0]] and out[0][1] == out[1][1]
    print("ok  same seed, same result")


def test_algorithms_run():
    for name, algo in ALGOS.items():
        for kv in ({}, dict(stacking=(), n_players=3, **FULL)):
            rules = Rules(**kv)
            ag, curve, secs = train(algo, UnoEnv(rules, OPPONENTS["rule"], 1), 1500, 1, 500)
            w, d, l = evaluate(ag, UnoEnv(rules, OPPONENTS["rule"], 2), 200)
            assert abs(w + d + l - 1) < 1e-9 and ag.Q
        print(f"ok  {name:15} trains and evaluates (1500 games, {len(ag.Q)} states)")


if __name__ == "__main__":
    test_rules_terminate_and_conserve()
    test_symmetry()
    test_mdp_matches_deck()
    test_reproducible()
    test_algorithms_run()
    print("ALL OK")
