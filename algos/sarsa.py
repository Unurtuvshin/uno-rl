"""Sarsa(0), textbook fig 3.5. On-policy: the update uses the action a' actually selected in s'."""
HP = dict(alpha=0.1, gamma=0.99, tau0=1.0, tau_min=0.05)


def episode(ag, env):
    alpha, gamma = ag.hp["alpha"], ag.hp["gamma"]
    s, legal = env.reset()
    a = ag.select(s, legal)
    while True:
        s2, legal2, r, done = env.step(a)
        row = ag.q(s)
        if done:                                   # terminal state: Q(s', .) = 0
            row[a] += alpha * (r - row[a])
            return r
        a2 = ag.select(s2, legal2)                 # a' from the legal actions of s' only
        row[a] += alpha * (r + gamma * ag.q(s2)[a2] - row[a])
        s, a = s2, a2
