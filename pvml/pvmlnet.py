from .cnn import CNN as _CNN




# Inspired by Alexnet, with differences due to the lack of maxpool and
# padding.
_LAYERS = [
    (96, 7, 3),
    (192, 3, 2),
    (192, 3, 1),
    (384, 3, 2),
    (384, 3, 1),
    (512, 3, 2),
    (512, 3, 1),
    (1024, 6, 1),
    (1024, 1, 1),
    (1000, 1, 1)
]


class PVMLNet(_CNN):
    def __init__(self):
        channels = [3] + [x[0] for x in _LAYERS]
        kernels = [x[1] for x in _LAYERS]
        strides = [x[2] for x in _LAYERS]
        super().__init__(channels, kernels, strides)

    def preprocessing(self, X):
        pass
        

if __name__ == "__main__":
    import numpy as np
    net = PVMLNet()
    parameters = sum(w.size for w in net.weights)
    parameters += sum(b.size for b in net.biases)
    print("{} parameters".format(parameters))
    images = np.empty((1, 224, 224, 3))
    activations = net.forward(images)
    for x in activations:
        print("x".join(map(str, x.shape)))
