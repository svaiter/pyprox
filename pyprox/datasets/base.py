from __future__ import division
from os.path import join, dirname
from matplotlib.image import imread

import numpy as np
from scipy.misc.pilutil import imresize

def load_sample_image(name):
    module_path = join(dirname(__file__), "images")
    filename = join(module_path, name + ".png")
    try:
        im = imread(filename)
    except:
        raise Exception(filename + " does not exists")
    return im