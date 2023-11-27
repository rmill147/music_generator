import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import pandas as pd


# read and plot data
def read_csv(filename):
    # with open(filename, 'r') as f:
    #     lines = f.readlines()
    data = pd.read_csv(filename, usecols=[i for i in range(4)])
    print(data)
    # header = lines[0].strip().split(',')
    # data = [line.strip().split(',') for line in lines[1:]]
    data = np.array(data).astype('float')
    return data


def plot(notes, accuracy, color="red"):

    # x = notes / nodes
    x = notes
    y = accuracy

    plt.scatter(x, y, color=color)
    pylab.ylim(0.0,1.1)
    plt.grid(True, linestyle='-.',which='both')
    pylab.xlabel('notes',labelpad=0.5)
    pylab.ylabel('accuracy',labelpad=1.0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()


filename_one = "run_3_10_nodes_600_epochs_clean.csv"
data = read_csv(filename_one)
# print("header")
# print(header)

plot(data[:, 0], data[:, 3])



