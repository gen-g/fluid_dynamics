"""
https://www.cosmo.sci.hokudai.ac.jp/~gfdlab/comptech/resume/210_advdiv/2012_0216-ogihara.pdf
"""

import numpy as np


def plot(n, x):
    import os
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    for t, n0 in enumerate(tqdm(n)):
        fig, ax = plt.subplots()
        ax.plot(x, n0)
        fig.savefig(f"{os.path.dirname(__file__)}/plot/{format(t, '08')}.png")
        plt.clf()
        plt.close(fig)


def derivative(n, dx):
    # dn = (n[2:] - n[:-2]) / (2 * dx)
    # 風上法
    dn = (n[1:-1] - n[0:-2]) / dx
    return dn


def derivative2(n, dx):
    dn = (n[2:] - 2 * n[1:-1] + n[:-2]) / (dx ** 2)
    return dn

def boundary_condition(n):
    n[0] = n[1]
    n[-1] = n[-2]
    return n

def convection_diffusion(n, c, k, dt, dx):
    dndx = derivative(n, dx)
    dn2dx = derivative2(n, dx)
    n0 = n.copy()
    n0[1:-1] = n[1:-1] + dt * (-c * dndx + k * dn2dx)
    n0 = boundary_condition(n0)
    return n0


def main():
    c = -1
    k = 0
    dx = 0.1
    dt = 0.01
    x = np.arange(0, np.pi, dx)

    n0 = np.sin(x)
    n = [n0]
    for t in range(200):
        n.append(convection_diffusion(n[t], c, k, dt, dx))
    plot(n, x)


if __name__ == '__main__':
    main()
