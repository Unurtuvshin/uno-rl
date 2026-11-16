"""Monte Carlo, textbook fig 3.9 (every-visit, undiscounted). Q(s,a) = mean reward of the episodes containing (s,a).
A draw (r = 0) is not a reward event, so nothing is updated (as in the slide)."""
HP = dict(tau0=1.0, tau_min=0.05, c_bid=0.0)       # c_bid unused


def episode(ag, env):
    s, legal = env.reset()
    hist = []
    while True:
        a = ag.select(s, legal)
        hist.append((s, a))
        s, legal, r, done = env.step(a)
        if done:
            break
    if r != 0:
        for st, at in hist:
            k = (st, at)
            n, tot = ag.N.get(k, 0) + 1, ag.S.get(k, 0.0) + r
            ag.N[k], ag.S[k] = n, tot
            ag.q(st)[at] = tot / n
    return r
