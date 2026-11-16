"""Q(0)-learning, textbook fig 3.3. Optional baseline (Sarsa's off-policy twin); not assigned to anyone."""
HP = dict(alpha=0.1, gamma=0.99, tau0=1.0, tau_min=0.05)


def episode(ag, env):
    alpha, gamma = ag.hp["alpha"], ag.hp["gamma"]
    s, legal = env.reset()
    while True:
        a = ag.select(s, legal)
        s2, legal2, r, done = env.step(a)
        row = ag.q(s)
        if done:
            row[a] += alpha * (r - row[a])
            return r
        row[a] += alpha * (r + gamma * max(ag.q(s2)[b] for b in legal2) - row[a])
        s, legal = s2, legal2
