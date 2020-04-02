from .cnn import CNN as _CNN


_LAYERS = [
    (96, 7, 4),
    # (16, 1, 1),
    (128, 3, 1),
    (16, 1, 1),
    (128, 3, 1),
    (32, 1, 1),
    (256, 3, 2),
    (32, 1, 1),
    (256, 3, 1),
    (48, 1, 1),
    (384, 3, 1),
    (48, 1, 1),
    (384, 3, 1),
    (64, 1, 1),
    (512, 3, 2),
    (64, 1, 1),
    (512, 3, 1),
    (1000, 1, 1)
]


def make_pvmlnet(pretrained=False):
    channels = [3] + [x[0] for x in _LAYERS]
    kernels = [x[1] for x in _LAYERS]
    strides = [x[2] for x in _LAYERS]
    net = _CNN(channels, kernels, strides)
    return net


if __name__ == "__main__":
    import numpy as np
    net = make_pvmlnet(pretrained=False)
    parameters = sum(w.size for w in net.weights)
    parameters += sum(b.size for b in net.biases)
    print("{} parameters".format(parameters))
    images = np.empty((1, 224, 224, 3))
    activations = net.forward(images)
    for x in activations:
        print("x".join(map(str, x.shape)))
