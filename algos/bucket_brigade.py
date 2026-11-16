"""Simple Bucket Brigade, textbook fig 3.7. bid = c_bid * Q; actions are chosen by bid; no alpha/gamma.
Bids settle near the undiscounted return (about -1..+1), so the Boltzmann temperature needs no rescaling."""
HP = dict(c_bid=0.1, tau0=1.0, tau_min=0.05)
BID_SCALE = True                                   # Agent.select ranks actions by c_bid * Q


def episode(ag, env):
    c = ag.hp["c_bid"]
    s, legal = env.reset()
    a = ag.select(s, legal)
    while True:
        s2, legal2, r, done = env.step(a)
        row = ag.q(s)
        if done:
            row[a] += r - c * row[a]
            return r
        a2 = ag.select(s2, legal2)
        row[a] += r + c * ag.q(s2)[a2] - c * row[a]    # Q += [r + bid(s',a') - bid(s,a)]
        s, a = s2, a2
