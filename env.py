"""MDP layer: turns the game into (state, legal actions) for seat 0 (the agent). Matches slides 6-7.

State  = (own hand bucket, opponent hand bucket, top-colour rank, pending-penalty bucket, just-drew, playable-kind bits)
Action = card kind x colour rank (colour rank = colours sorted by how many you hold) + draw + pass
Change the state/action design here only; rules live in game.py.
"""
import random
from game import Game, Rules, KIND
from opponents import random_opp


class Abstraction:
    def __init__(self, rules):
        self.kinds = ["NUM"] + [k for k in ("SKIP", "REV", "D2", "WILD", "D4") if k in rules.specials]
        self.two = rules.n_players == 2
        if self.two and "REV" in self.kinds:        # reverse acts as skip with 2 players
            self.kinds.remove("REV")
            if "SKIP" not in self.kinds:
                self.kinds.insert(1, "SKIP")
        self.nc, self.lim = rules.n_colors, rules.hand_limit
        self.DRAW = len(self.kinds) * self.nc
        self.PASS = self.DRAW + 1
        self.n_actions = self.PASS + 1
        self.ranges = (self.lim - 1 if self.lim else 5, self.lim - 1 if self.lim else 4, self.nc, 4, 2, 2 ** len(self.kinds))
        self.n_states_max = 1
        for x in self.ranges:
            self.n_states_max *= x

    def encode(self, game, seat=0):
        hand = game.hands[seat]
        counts = [0] * 4
        for c in hand:
            if c[0] < 4:
                counts[c[0]] += 1
        order = sorted(range(self.nc), key=lambda c: (-counts[c], c))
        rank = {c: i for i, c in enumerate(order)}
        by, bits = {}, 0
        for m in game.legal_moves(seat):
            if m[0] == "draw":
                a = self.DRAW
            elif m[0] == "pass":
                a = self.PASS
            else:
                k = KIND[m[1][1]]
                if self.two and k == "REV":
                    k = "SKIP"
                k = self.kinds.index(k)
                a = k * self.nc + rank[m[2]]
                bits |= 1 << k
            by.setdefault(a, []).append(m)
        n, o = len(hand), min(len(h) for i, h in enumerate(game.hands) if i != seat)
        pend = 0 if not game.pend else 1 if game.pend == 2 else 2 if game.pend == 4 else 3
        if self.lim:                              # one bucket per hand size, so the 11-card danger is visible
            nb, ob = n - 1, min(o, self.lim - 1) - 1
        else:
            nb = 0 if n == 1 else 1 if n == 2 else 2 if n == 3 else 3 if n <= 5 else 4
            ob = 0 if o == 1 else 1 if o == 2 else 2 if o <= 4 else 3
        state = (nb, ob,
                 rank[game.top_color], pend, int(game.drawn is not None), bits)
        return state, by


class UnoEnv:
    """reset() -> (s, legal);  step(a) -> (s', legal', r, done).  r: +1 win, -1 loss, 0 otherwise.
    The opponents move inside step(), so s' is always the agent's next decision (or None when done)."""

    def __init__(self, rules=Rules(), opponent=random_opp, seed=0):
        self.rules, self.opponent, self.rng = rules, opponent, random.Random(seed)
        self.abs = Abstraction(rules)
        self.n_actions = self.abs.n_actions
        self.games = 0
        self.observer = None                      # copied to each new Game; see Game.observer

    def _to_agent(self):
        g = self.game
        while not g.done and g.cur != 0:
            g.apply(self.opponent(g, g.cur, g.legal_moves(g.cur), self.rng))

    def _obs(self):
        if self.game.done:
            self.moves = {}
            return None, []
        s, self.moves = self.abs.encode(self.game)
        return s, list(self.moves)

    def reset(self):
        self.game = Game(self.rules, self.rng, start=self.games % self.rules.n_players)  # alternate who starts
        self.games += 1
        self.game.observer = self.observer
        if self.observer:
            self.observer(self.game, None, None)      # the deal
        self._to_agent()
        return self._obs()

    def step(self, a):
        self.game.apply(self.rng.choice(self.moves[a]))   # random among cards equivalent under the abstraction
        self._to_agent()
        w = self.game.winner
        r = 0 if w is None else (1 if w == 0 else -1)
        s, legal = self._obs()
        return s, legal, r, self.game.done
