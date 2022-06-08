def next_state(A, t, x, k):
    a = x
    for h in range(k):
        a -= A[t, h]
        if a < 0:
            return h


def out(B, st, y, p):
    a = y
    for u in range(p):
        a -= B[st, u]
        if a < 0:
            return u


def first_state(pi, x, k):
    a = x[0]
    for h in range(k):
        a -= pi[h]
        if a < 0:
            z1 = h
    return z1


def hmm(A, B, pi, x, y, p, k, l):
    sample = jax.numpy.empty(l)
    outcome = jax.numpy.empty(l)
    z1 = first_state(pi, x, k)
    sample = sample.at[0].set(z1)
    outcome = outcome.at[0].set(out(B, z1, y[0], p))
    z_i = z1
    for j in range(1, l):
        z_i1 = next_state(A, z_i, x[j], k)
        sample = sample.at[j].set(z_i1)
        outcome = outcome.at[j].set(out(B, z_i1, y[j], p))
        z_i = z_i1
    return sample, outcome
