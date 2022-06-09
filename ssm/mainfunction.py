class discreteHMM:
    def next_state(self, A, t, x, k):
        a = x
        for h in range(k):
            a -= A[t, h]
            if a < 0:
                return h

    def outcome(self, B, st, y, p):
        a = y
        for u in range(p):
            a -= B[st, u]
            if a < 0:
                return u

    def first_state(self, pi, x, k):
        a = x[0]
        for h in range(k):
            a -= pi[h]
            if a < 0:
                z1 = h
        return z1

    def hmm(self, A, B, pi, x, y, p, k, l):
        sample = jax.numpy.empty(l)
        outcomes = jax.numpy.empty(l)
        beta = discreteHMM()
        z1 = beta.first_state(pi, x, k)
        sample = sample.at[0].set(z1)
        outcomes = outcomes.at[0].set(beta.outcome(B, z1, y[0], p))
        z_i = z1
        for j in range(1, l):
            z_i1 = beta.next_state(A, z_i, x[j], k)
            sample = sample.at[j].set(z_i1)
            outcomes = outcomes.at[j].set(beta.outcome(B, z_i1, y[j], p))
            z_i = z_i1
        return sample, outcomes


class continuousHMM:
    def state(self, t, key):
        key = jax.random.PRNGKey(key)
        while True:

            key, subkey = jax.random.split(key)
            x = jax.random.uniform(key=subkey, shape=(1,))
            y = jax.random.uniform(key=key, shape=(1,))

            if transition(t, x) >= y:
                return x[0]
                break

    def out(self, st, key):
        key = jax.random.PRNGKey(key)
        while True:
            key, subkey = jax.random.split(key)
            x = jax.random.uniform(key=subkey, shape=(1,))
            y = jax.random.uniform(key=key, shape=(1,))
            if emission(st, x) >= y:
                return x[0]
                break

    def first_state(self):
        key = jax.random.PRNGKey(0)

        while True:
            key, subkey = jax.random.split(key)
            x = jax.random.uniform(key=subkey, shape=(1,))
            y = jax.random.uniform(key=key, shape=(1,), minval=0, maxval=1)

            if prior(x) >= y:
                z1 = x[0]
                break
        return z1

    def hmm(self, l):
        beta = continuousHMM()
        sample = jax.numpy.empty(l)
        outcomes = jax.numpy.empty(l)

        z_i = beta.first_state()

        sample = sample.at[0].set(z_i)
        outcomes = outcomes.at[0].set(beta.out(z_i, 0))
        for j in range(1, l):
            z_i1 = beta.state(z_i, j)
            sample = sample.at[j].set(z_i1)
            outcomes = outcomes.at[j].set(beta.out(z_i1, l + j))
            z_i = z_i1
        return sample, outcomes


      
