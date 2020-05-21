#!/usr/bin/env python3

import pvml
import numpy as np
import sys
import PIL.Image


def process_batch(paths, net):
    images = []
    for impath in paths:
        im = PIL.Image.open(impath).convert("RGB")
        im = np.array(im.resize((224, 224), PIL.Image.BILINEAR))
        images.append(im / 255.0)
    images = np.stack(images, 0)

    # Process the images
    activations = net.forward(images)

    # Take the activation before the last convolution
    features = activations[-3].reshape(len(paths), -1)
    return features


# Check the command line
if len(sys.argv) < 2:
    print("USAGE: ./pvmlnet_classify IMAGE1 IMAGE2 IMAGE3 ...")
    sys.exit()

# Load the network
net = pvml.PVMLNet.load("pvmlnet.npz")

# Load the images
filenames = sys.argv[1:]

# Process batches of 16 images
batch_size = 16
for i in range(0, len(filenames), batch_size):
    features = process_batch(filenames[i:i + batch_size], net)
    np.savetxt(sys.stdout, features, fmt="%.5g")
