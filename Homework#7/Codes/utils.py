import matplotlib.pyplot as plt
import numpy as np


def log_plot(x, y, title, x_label, y_label):
    plt.plot(x, y)
    plt.yscale("log")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


if __name__ == '__main__':
    y = [1e3, 1e-1, 1e-4]
    x = np.arange(3)
    log_plot(x, y, "test", "x", "y")
