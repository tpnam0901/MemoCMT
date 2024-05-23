from torch import optim


def sgd(params, learning_rate=0.01, momentum=0.9):
    return optim.SGD(params, lr=learning_rate, momentum=momentum)


def adam(params, learning_rate=0.01):
    return optim.Adam(params, lr=learning_rate)


def rmsprop(params, learning_rate=0.01):
    return optim.RMSprop(params, lr=learning_rate)


def adagrad(params, learning_rate=0.01):
    return optim.Adagrad(params, lr=learning_rate)


def adamw(params, learning_rate=0.01, weight_decay=0.01):
    return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
