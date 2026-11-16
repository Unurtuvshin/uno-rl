"""UNO game engine. Rules live in `Rules`; change a rule by changing a field, not the engine.

Card = (color, value). color 0-3 (R,Y,G,B), 4 = wild. value 0-9 numbers, 10 skip, 11 reverse,
12 draw2, 13 wild, 14 wild+4.  Defaults = the "current proposal" of the team's rule sheet (2 colours, numbers 1-2,
skip/+2/+4, lose at 12 cards).  The full 108-card UNO is Rules(n_colors=4, numbers=tuple(range(10)),
specials=("SKIP","REV","D2","WILD","D4"), hand_limit=0).
"""
from dataclasses import dataclass

SKIP, REV, D2, WILD, D4 = 10, 11, 12, 13, 14
KIND = ["NUM"] * 10 + ["SKIP", "REV", "D2", "WILD", "D4"]


@dataclass(frozen=True)
class Rules:
    n_players: int = 2
    hand_size: int = 7
    n_colors: int = 2                         # colours 0..n_colors-1
    numbers: tuple = (1, 2)                   # number cards per colour (0 would be single, others twice)
    specials: tuple = ("SKIP", "D2", "D4")    # subset of SKIP REV D2 WILD D4 (SKIP/REV/D2: 2 per colour; WILD/D4: 4 total)
    hand_limit: int = 12                      # reaching this many cards loses (0 = no limit); also after a penalty draw
    plus4_needs_no_color: bool = False        # True = official: +4 only if no card of the current colour
    # (pending kind, kind played on top of it): +2 on +2, +4 on +4, +4 on +2.  () = no stacking
    stacking: tuple = (("D2", "D2"), ("D4", "D4"), ("D2", "D4"))
    draw_even_if_playable: bool = True        # False = may only draw when no card is playable
    auto_play_drawn: bool = False             # True = a playable drawn card is played at once
    max_turns: int = 500                      # moves (all players) before the game is a draw


def make_deck(rules):
    deck = []
    for c in range(rules.n_colors):
        for v in rules.numbers:
            deck += [(c, v)] * (1 if v == 0 else 2)
        for s in rules.specials:
            if s in ("SKIP", "REV", "D2"):
                deck += [(c, KIND.index(s))] * 2
    for s in rules.specials:
        if s in ("WILD", "D4"):
            deck += [(4, KIND.index(s))] * 4
    return deck


class Game:
    def __init__(self, rules, rng, start=0):
        self.rules, self.rng = rules, rng
        self.deck = make_deck(rules)
        self.n_cards = len(self.deck)
        rng.shuffle(self.deck)
        n = rules.n_players
        c = self.deck.pop(next(i for i, x in enumerate(self.deck) if x[1] < 10))   # first card: a plain number
        self.hands = [[self.deck.pop() for _ in range(rules.hand_size)] for _ in range(n)]
        self.discard = [c]
        self.top_color, self.top_value = c
        self.dir, self.cur = 1, start % n
        self.pend, self.pend_kind = 0, None      # pending draw penalty
        self.drawn = None                        # card just drawn this turn (play-or-pass decision)
        self.turns, self.done, self.winner = 0, False, None
        self.observer = None                     # optional f(game, seat, move), called after every applied move (replay.html)

    # ---- helpers
    def total_cards(self):
        return len(self.deck) + len(self.discard) + sum(map(len, self.hands))   # == self.n_cards

    def _advance(self, steps=1):
        self.cur = (self.cur + self.dir * steps) % self.rules.n_players
        self.drawn = None

    def _draw(self, p, k):
        got = []
        for _ in range(k):
            if not self.deck and len(self.discard) > 1:
                top = self.discard.pop()
                self.deck, self.discard = self.discard, [top]
                self.rng.shuffle(self.deck)
            if not self.deck:
                break
            card = self.deck.pop()
            self.hands[p].append(card)
            got.append(card)
        lim = self.rules.hand_limit
        if lim and len(self.hands[p]) >= lim:     # too many cards: the next seat wins
            self.done, self.winner = True, (p + 1) % self.rules.n_players
        return got

    def _playable(self, card, hand):
        col, val = card
        if val == WILD:
            return True
        if val == D4:
            return not (self.rules.plus4_needs_no_color and any(c[0] == self.top_color for c in hand))
        return col == self.top_color or val == self.top_value

    def _plays(self, card):
        return [("play", card, c) for c in range(self.rules.n_colors)] if card[0] == 4 else [("play", card, card[0])]

    def best_color(self, p):
        counts = [0] * 4
        for c in self.hands[p]:
            if c[0] < 4:
                counts[c[0]] += 1
        return max(range(self.rules.n_colors), key=lambda c: (counts[c], -c))

    # ---- interface
    def legal_moves(self, p):
        hand = self.hands[p]
        if self.pend:
            moves = [m for c in set(hand) if (self.pend_kind, KIND[c[1]]) in self.rules.stacking for m in self._plays(c)]
            return moves + [("draw",)]
        if self.drawn is not None:
            return self._plays(self.drawn) + [("pass",)]
        moves = [m for c in set(hand) if self._playable(c, hand) for m in self._plays(c)]
        if not moves or self.rules.draw_even_if_playable:
            moves.append(("draw",))
        return moves

    def apply(self, mv):
        p = self.cur
        self._apply(mv)
        if self.observer:
            self.observer(self, p, mv)

    def _apply(self, mv):
        p, kind = self.cur, mv[0]
        self.turns += 1
        if kind == "play":
            _, card, color = mv
            self.hands[p].remove(card)
            self.discard.append(card)
            self.top_color, self.top_value = color, card[1]
            if not self.hands[p]:
                self.done, self.winner = True, p
                return
            v, step = card[1], 1
            if v == SKIP or (v == REV and self.rules.n_players == 2):
                step = 2
            elif v == REV:
                self.dir *= -1
            elif v in (D2, D4):
                self.pend_kind = KIND[v]
                self.pend += 2 if v == D2 else 4
            self._advance(step)
        elif kind == "draw":
            if self.pend:
                self._draw(p, self.pend)
                self.pend, self.pend_kind = 0, None
                if not self.done:
                    self._advance()
            else:
                got = self._draw(p, 1)
                if self.done:
                    return
                if got and self._playable(got[0], self.hands[p]):
                    if self.rules.auto_play_drawn:
                        self._apply(("play", got[0], got[0][0] if got[0][0] < 4 else self.best_color(p)))
                        self.turns -= 1
                    else:
                        self.drawn = got[0]
                else:
                    self._advance()
        else:                                    # pass
            self._advance()
        if not self.done and self.turns >= self.rules.max_turns:
            self.done = True                     # draw: winner stays None
