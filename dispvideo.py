import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import color

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

def play_video(four_dim_array, threshold):
    """
    Assumes the input is [y, x, RGB, sample], plays the video
    """
    _, _, _, samples = four_dim_array.shape
    gray = grayscale(four_dim_array)
    thresholded = apply_threshold(gray, threshold)
    f = plt.figure(1)
    f.canvas.manager.window.activateWindow()
    f.canvas.manager.window.raise_()
    p = f.add_subplot(111)
    image = p.imshow(thresholded[:, :, 0], cmap="gray", vmin=0, vmax=255)
    for i in range(1, samples):
        tmp = thresholded[:, :, i]
        image.set_data(tmp)
        plt.pause(0.02)

def main(fname, threshold):
    try:
        threshold = int(threshold)
    except TypeError:
        print("Second argument must be convertible to integer!")
        return

    video = scipy.io.loadmat(fname)
    common_keys = ["__header__", "__version__", "__globals__"]
    keys = list(video.keys())
    key = [i for i in keys if i not in common_keys]
    if len(key) != 1:
        raise KeyError("Too many keys!")
    key = key[0]
    video = video[key]
    print("Read video mat file.")

    play_video(video, threshold)

if __name__ == "__main__":
    fname = sys.argv[1]
    try:
        threshold = sys.argv[2]
    except IndexError:
        threshold=0
    main(fname, threshold)
