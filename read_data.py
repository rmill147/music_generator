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
    num_epochs = 100
    num_nodes = data[0][1]
    print(num_nodes)
    return data, num_epochs,num_nodes


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
    plt.title(f"Accuracy per number of notes ({num_nodes}) nodes")
    plt.show()


filename_one = "test_epochs=100.csv"
data, num_epochs,num_nodes = read_csv(filename_one)
plot(data[:, 0], data[:, 3])



