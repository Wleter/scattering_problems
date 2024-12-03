import shutil
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

def plot() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")

    return fig, ax

def plot_scattering(ax, x, y_re, y_im, label=""):
    lines = ax.plot(x, y_re, label=label)
    ax.plot(x, y_im, linestyle="--", color=lines[0].get_color())

def load_save(filename, delimiter='\t', skiprows=1):
    try:
        data = np.loadtxt(f'../data/{filename}', delimiter=delimiter, skiprows=skiprows)
        shutil.copyfile(f'../data/{filename}', f'../data_saved/{filename}')
    except:
        data = np.loadtxt(f'../data_saved/{filename}', delimiter=delimiter, skiprows=skiprows)
    
    return data

def load(filename, delimiter='\t', skiprows=1):
    data = np.loadtxt(f'../data/{filename}', delimiter=delimiter, skiprows=skiprows)

    return data