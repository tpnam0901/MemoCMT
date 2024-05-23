import optax


@optax.inject_hyperparams
def sgd(learning_rate=0.01):
    return optax.sgd(learning_rate=learning_rate)


@optax.inject_hyperparams
def adam(learning_rate=0.01):
    return optax.adam(learning_rate=learning_rate)


@optax.inject_hyperparams
def rmsprop(learning_rate=0.01):
    return optax.rmsprop(learning_rate=learning_rate)


@optax.inject_hyperparams
def adagrad(learning_rate=0.01):
    return optax.adagrad(learning_rate=learning_rate)


@optax.inject_hyperparams
def adafactor(learning_rate=0.01):
    return optax.adafactor(learning_rate=learning_rate)


@optax.inject_hyperparams
def adamw(learning_rate=0.01, weight_decay=0.01):
    return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
