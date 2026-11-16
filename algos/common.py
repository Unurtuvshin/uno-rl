"""Shared by every algorithm: Boltzmann action selection, training loop, greedy evaluation."""
import math
import random
import time


class Agent:
    def __init__(self, algo, n_actions, seed, **hp):
        self.hp = {**algo.HP, **hp}
        self.n_actions = n_actions
        self.rng = random.Random(seed)            # own RNG: independent of the environment's
        self.scale = self.hp["c_bid"] if getattr(algo, "BID_SCALE", False) else 1.0
        self.tau = self.hp["tau0"]
        self.Q = {}                               # state -> [value per action]
        self.N, self.S = {}, {}                   # Monte Carlo visit counts / reward sums

    def q(self, s):
        row = self.Q.get(s)
        if row is None:
            row = self.Q[s] = [0.0] * self.n_actions
        return row

    def select(self, s, legal, greedy=False):
        """Boltzmann over legal actions (greedy=True: argmax, random ties)."""
        row, sc = self.q(s), self.scale
        if greedy:
            best = max(row[a] for a in legal)
            return self.rng.choice([a for a in legal if row[a] == best])
        vals = [row[a] * sc for a in legal]
        m = max(vals)
        w = [math.exp((v - m) / self.tau) for v in vals]
        return self.rng.choices(legal, w)[0]


def train(algo, env, episodes, seed=0, window=1000, **hp):
    """env may be a list of envs: `episodes` games against each in turn (curriculum), one agent, tau decays over all."""
    envs = env if isinstance(env, (list, tuple)) else [env]
    ag = Agent(algo, envs[0].n_actions, seed, **hp)
    total = episodes * len(envs)
    decay = (ag.hp["tau_min"] / ag.hp["tau0"]) ** (1 / max(total, 1))
    curve, wins, losses, t0 = [], 0, 0, time.time()
    for ep in range(1, total + 1):
        r = algo.episode(ag, envs[(ep - 1) // episodes])
        wins += r == 1
        losses += r == -1
        ag.tau = max(ag.hp["tau_min"], ag.tau * decay)
        if ep % window == 0:
            curve.append((ep, wins / window, losses / window, round(ag.tau, 4), len(ag.Q)))
            wins = losses = 0
    return ag, curve, time.time() - t0


def evaluate(ag, env, n):
    """Greedy play, no learning. Returns (win, draw, loss) fractions."""
    w = l = 0
    for _ in range(n):
        s, legal = env.reset()
        done = False
        while not done:
            s, legal, r, done = env.step(ag.select(s, legal, greedy=True))
        w += r == 1
        l += r == -1
    return w / n, 1 - (w + l) / n, l / n
