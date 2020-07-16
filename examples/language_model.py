#!/usr/bin/env python3

# Train and run a character-level language model.  The training set is
# a single ASCII text file.

import numpy as np
import pvml
import sys

# CONFIGURATION
HIDDEN_STATES = [256]
BATCH_SIZE = 16
TRAINING_STEPS = 10000
MAXLEN = 400


# ASCII characters in the 32-126 range are encoded as (code - 32).
# The start of a sequence is encoded as 128 - 32 = 96
# The end of a sequence is encoded as 129 - 32 = 97
# Any other character is encoded as 127 - 32 = 95

REPORT_EVERY = 1000
LEARNING_RATE = 0.01
SOS = 0
EOS = 1
UNK = 2


def encode1(c):
    n = ord(c)
    return (n - 32 if n >= 32 and n < 127 else UNK)


def encode(s):
    codes = [SOS, *(encode1(c) for c in s), EOS]
    return np.array(codes, dtype=np.uint8)


def decode(codes):
    return "".join(chr(32 + c) if c != UNK else "?" for c in codes)


def read_data(filename, seqlen, delimiter=None):
    with open(filename) as f:
        text = f.read()
    alphabet = ["<SOS>", "<EOS>", "<UNK>"] + sorted(set(text))
    encoding = dict((c, n) for n, c in enumerate(alphabet))
    if delimiter is not None:
        text = text.split(delimiter)
    else:
        text = [text]
    data = []
    for seq in text:
        if not seq:
            continue
        codes = [encoding.get(c, UNK) for c in seq]
        extra = len(codes) % seqlen
        codes.extend([EOS] * (seqlen - extra))
        data.append(np.array(codes).reshape(-1, seqlen))
    return np.concatenate(data, 0), alphabet


def one_hot_vectors(X, k):
    U = X.reshape(-1)
    H = np.zeros((U.size, k), dtype=np.uint8)
    H[np.arange(U.size), U] = 1
    return H.reshape(*X.shape, k)


def generate(rnn, maxlen, alphabet):
    codes = [SOS]
    last = SOS
    X = np.zeros((1, 1, len(alphabet)), dtype=np.uint8)
    init = None
    while len(codes) < maxlen and last != EOS:
        X.fill(0)
        X[0, 0, last] = 1
        Hs, P = rnn.forward(X, init)
        init = [H[:, -1, :] for H in Hs[1:]]
        last = np.random.choice(len(alphabet), p=P[0, -1, :])
        codes.append(last)
    return "".join(alphabet[c] for c in codes)
            

def train(training_file, seqlen, delimiter):
    data, alphabet = read_data(sys.argv[1], seqlen, delimiter)
    X = one_hot_vectors(data[:, :-1], len(alphabet))
    Y = data[:, 1:]
    rnn = pvml.RNN([len(alphabet), 256, len(alphabet)])
    steps = 0
    steps_per_call = REPORT_EVERY // BATCH_SIZE
    while steps < TRAINING_STEPS:
        rnn.train(X, Y, lr=LEARNING_RATE, steps=steps_per_call, batch=BATCH_SIZE)
        P = rnn.forward(X)[1]
        loss = rnn.loss(Y, P)
        steps += steps_per_call
        print(steps, loss)
        print(generate(rnn, MAXLEN, alphabet))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("USAGE: ./language_model.py TRAINING_FILE [SEQUENCE_LENGTH] [DELIMITER]")
        sys.exit()
    training_file = sys.argv[1]
    seqlen = (32 if len(sys.argv) < 3 else int(sys.argv[2]))
    delimiter = (None if len(sys.argv) < 4 else sys.argv[3])
    train(training_file, seqlen, delimiter)
