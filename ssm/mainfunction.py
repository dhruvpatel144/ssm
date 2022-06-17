class discreteHMM:
    def next_state(self, A, current_state, xkey):
        nextstate = int(
            tfp.distributions.Categorical(
                probs=A[current_state], name="Categorical"
            ).sample(1, seed=xkey)
        )
        return nextstate

    def outcome(self, B, current_state, ykey):
        u = int(
            tfp.distributions.Categorical(
                probs=B[current_state], name="Categorical"
            ).sample(1, seed=ykey)
        )
        return u

    def first_state(self, pi, xkey):
        z1 = int(
            tfp.distributions.Categorical(probs=pi, name="Categorical").sample(
                1, seed=xkey
            )
        )
        return z1

    def hmm(self, A, B, pi, length_of_chain):

        xkey = jax.random.split(jax.random.PRNGKey(0), num=length_of_chain)
        ykey = jax.random.split(jax.random.PRNGKey(1), num=length_of_chain)

        sample = jax.numpy.empty(length_of_chain)
        outcomes = jax.numpy.empty(length_of_chain)

        beta = discreteHMM()
        z1 = beta.first_state(pi, xkey[0])

        sample = sample.at[0].set(z1)
        outcomes = outcomes.at[0].set(beta.outcome(B, z1, ykey[0]))
        z_i = z1
        for j in range(1, length_of_chain):
            z_i1 = beta.next_state(A, z_i, xkey[j])
            sample = sample.at[j].set(z_i1)
            outcomes = outcomes.at[j].set(beta.outcome(B, z_i1, ykey[j]))
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


      
