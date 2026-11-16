"""Opponent policies: f(game, seat, legal_moves, rng) -> move.  Both obey the same rules as the agent."""
from game import KIND


def random_opp(game, p, moves, rng):
    return rng.choice(moves)


def rule_opp(game, p, moves, rng):
    """Greedy heuristic: shed plain cards of the most-held colour, hold wilds, attack when the other side is low."""
    plays = [m for m in moves if m[0] == "play"]
    if not plays:
        return moves[0]
    small = min(len(h) for i, h in enumerate(game.hands) if i != p) <= 2
    best = game.best_color(p)

    def score(m):
        k = KIND[m[1][1]]
        s = {"NUM": 5, "SKIP": 3, "REV": 3, "D2": 3, "WILD": 1, "D4": 0}[k]
        if small and k in ("SKIP", "REV", "D2", "D4"):
            s += 10
        if game.pend:
            s += 20 + (k == "D4")
        return s + 0.5 * (m[2] == best) + rng.random() * 0.1

    return max(plays, key=score)


OPPONENTS = {"random": random_opp, "rule": rule_opp}
