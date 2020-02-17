import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import color

r_index = 0
g_index = 1
b_index = 2

def get_red(x):
    x[:,:,1,:] = 0
    x[:,:,2,:] = 0
    return x

def get_green(x):
    x[:,:,0,:] = 0
    x[:,:,2,:] = 0
    return x

def get_blue(x):
    x[:,:,0,:] = 0
    x[:,:,1,:] = 0
    return x

def grayscale(x):
    red = np.squeeze(get_red(x)[:, :, r_index, :])
    blue = np.squeeze(get_blue(x)[:, :, g_index, :])
    green = np.squeeze(get_green(x)[:, :, b_index, :])

    gray = 0.2125 * red + 0.7154 * green + 0.0721 * blue

    print("red:", red[:].max())
    print("blue:", blue[:].max())
    print("green:", green[:].max())
    print("gray:", gray[:].max())

    return gray

def play_video(four_dim_array):
    """
    Assumes the input is [y, x, RGB, sample], plays the video
    """
    _, _, _, samples = four_dim_array.shape
    #channel = blue(four_dim_array)
    gray = grayscale(four_dim_array)
    f = plt.figure(1)
    f.canvas.manager.window.activateWindow()
    f.canvas.manager.window.raise_()
    p = f.add_subplot(111)
    print("Gray shape:", gray.shape)
    image = p.imshow(gray[:, :, 0], cmap=plt.get_cmap("gray"))
    for i in range(1, samples):
        tmp = gray[:, :, i]
        image.set_data(tmp)
        plt.pause(0.02)

def main(fname):

    video = scipy.io.loadmat(fname)
    common_keys = ["__header__", "__version__", "__globals__"]
    keys = list(video.keys())
    key = [i for i in keys if i not in common_keys]
    if len(key) != 1:
        raise KeyError("Too many keys!")
    key = key[0]
    video = video[key]
    print("Read video mat file.")

    play_video(video)

if __name__ == "__main__":
    fname = sys.argv[1]
    print("File to play:", fname)
    main(fname)
