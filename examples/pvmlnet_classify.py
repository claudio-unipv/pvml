#!/usr/bin/env python3

import pvml
import numpy as np
import sys
import PIL.Image

# Check the command line
if len(sys.argv) < 2:
    print("USAGE: ./pvmlnet_classify IMAGE1 IMAGE2 IMAGE3 ...")
    sys.exit()

# Load the network
net = pvml.PVMLNet.load("pvmlnet.npz")

# Load the images
filenames = sys.argv[1:]
images = []
for impath in filenames:
    im = PIL.Image.open(impath).convert("RGB")
    im = np.array(im.resize((224, 224), PIL.Image.BILINEAR))
    images.append(im / 255.0)
images = np.stack(images, 0)

# Classify the images
labels, probs = net.inference(images)

# Print the results
ii = np.argsort(-probs, 1)
for i in range(len(filenames)):
    for k in range(5):
        p = probs[i, ii[i, k]] * 100
        synset = net.CLASSES[ii[i, k]]
        print("{}. {} ({:.1f}%)".format(k + 1, synset, p))
    print()
