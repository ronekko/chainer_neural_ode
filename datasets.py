# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 02:32:11 2019

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt


def create_points(n_sqrt=30):
    """Creates 2D points `x` and corresponding 2D points `y`.

    A set of points `x` consists of N points on meshgrid([0, 1), [0, 1)),
    where N = n_sqrt^2. Each point of a set `y` is created by a mapping
    (y_0, y_1) = (r * cos(\\theta), r * sin(\\theta)), where r = x_0
    and \\theta = x_1 * 2 \\pi.
    """

    x = np.meshgrid(np.linspace(0, 1, n_sqrt, endpoint=False),
                    np.linspace(0, 1, n_sqrt, endpoint=False))
    x = np.dstack(x).reshape(-1, 2)

    r, angle = x.T
    r = r + 1
    angle = angle * 2 * np.pi
    y = np.stack((r * np.cos(angle), r * np.sin(angle)), axis=1)
    return x.astype(np.float32), y.astype(np.float32)


def plot_points(x, y):
    # Each point is colored with (R, G, B) = (x_0, 0, x_1)
    colors = np.stack((x[:, 0], np.zeros(len(x)), x[:, 1]), 1)
    colors = np.stack((x[:, 1], np.full(len(x), 0.3), x[:, 0]), 1)

    fig, ax = plt.subplots(1, 2)

    ax[0].scatter(*x.T, c=colors)
    ax[0].set_title('X')
    ax[0].grid()
    ax[0].set_aspect('equal')

    ax[1].scatter(*y.T, c=colors)
    ax[1].set_title('Y')
    ax[1].grid()
    ax[1].set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    n_sqrt = 30
    x, y = create_points(n_sqrt)
    plot_points(x, y)
