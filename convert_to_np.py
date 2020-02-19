#!/usr/bin/env python3
import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import color

DATADIR = "mat/"
OUTDIR = "npdata/"

RED = 0
GREEN = 1
BLUE = 2

def get_channel(x, channel):
    ret = np.copy(x)
    return np.squeeze(ret[:, :, channel, :])

def grayscale(x):

    red = get_channel(x, RED)
    green = get_channel(x, GREEN)
    blue = get_channel(x, BLUE)

    gray = 0.2125 * red + 0.7154 * green + 0.0721 * blue

    print("red:", red.max())
    print("blue:", blue.max())
    print("green:", green.max())
    print("gray:", gray.max())

    return gray

def apply_threshold(x, thresh):
    ret = np.copy(x)
    indices = x <= thresh
    ret[indices] = 0

    return ret


def _main():
    files = os.listdir(DATADIR)
    extless_files = [os.path.splitext(i)[0] for i in files]
    infiles = [os.path.join(DATADIR, i) for i in files]
    outfiles = [os.path.join(OUTDIR, i) for i in extless_files]

    assert(len(infiles)==len(outfiles))
    print("Files to proc:", infiles)
    print("Files to output:", outfiles)
    input("Press ENTER to continue...")

    for (infile, outfile) in zip(infiles, outfiles):
        video = scipy.io.loadmat(infile)
        common_keys = ["__header__", "__version__", "__globals__"]
        keys = list(video.keys())
        key = [i for i in keys if i not in common_keys]
        if len(key) != 1:
            raise KeyError("Too many keys!")
        key = key[0]
        video = video[key]
        print("Read video mat file:", infile)
        # Swap the axes for easier indexing
        video = np.moveaxis(video, -1, 0)
        np.save(outfile, video)
        print("Wrote video numpy file:", outfile+".npy")

    print("Done!")


if __name__ == "__main__":
    _main()

