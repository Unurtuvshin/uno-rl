"""Simple Profit Sharing, textbook fig 3.11. Rules (s,a) are remembered; when a reward arrives it is shared back.
The slide leaves r(t) undefined: here r(t) = r * gamma^(T-1-t), so the last rule gets the full reward.
A draw (r = 0) is not a reward event, so nothing is updated (as in the slide)."""
HP = dict(c_bid=0.1, gamma=0.95, tau0=1.0, tau_min=0.05)


def episode(ag, env):
    c, gamma = ag.hp["c_bid"], ag.hp["gamma"]
    s, legal = env.reset()
    hist = []
    while True:
        a = ag.select(s, legal)
        hist.append((s, a))
        s, legal, r, done = env.step(a)
        if done:
            break
    if r != 0:
        T = len(hist)
        for t, (st, at) in enumerate(hist):
            row = ag.q(st)
            row[at] += c * (r * gamma ** (T - 1 - t) - row[at])
    return r
