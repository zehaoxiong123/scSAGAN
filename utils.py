import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

def count_z_score(data):
    lenth = len(data)
    total = sum(data)
    ave = float(total) / lenth
    tempsum = sum([pow(data[i] - ave, 2) for i in range(lenth)])
    tempsum = pow(float(tempsum) / lenth, 0.5)
    for i in range(lenth):
        data[i] = (data[i] - ave) / tempsum
    return data

def plot_in_bar():
    cm = plt.cm.get_cmap('PuBu')
    xy = range(50)
    z = xy
    sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=50, s=10, cmap=cm)
    plt.colorbar(sc)
    plt.show()