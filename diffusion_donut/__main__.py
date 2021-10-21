""" 2-dimensional diffusion equation
The equation is described as:
    u_xx + u_yy = u_t.
The initial condition:
    u = 1000 at 4<r<5
      = 0    otherwise.
The boundary condition:
    u = 0 x_limit and y_limit
"""
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def create_coordinate(limx, limy, dx):
    x = np.arange(limx[0], limx[1], dx)
    y = np.arange(limy[0], limy[1], dx)
    X = np.meshgrid(x, y)
    return X


def df2dx2(u, dh, axis):
    if axis == 'x':
        dx = dh
        du = u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]
        return du / (dx ** 2)
    elif axis == 'y':
        dy = dh
        du = u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]
        return du / (dy ** 2)
    else:
        return


def boundary_condition(u):
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    return u


def diffusion(u, dx, dy):
    d = 4
    dt = ((dx * dy) ** 2) / (2 * d * (dx ** 2 + dy ** 2))
    # in bulk
    du2dx = df2dx2(u, dx, axis='x')
    du2dy = df2dx2(u, dy, axis='y')
    diffusion_term = d * dt * (du2dx + du2dy)

    u0 = u.copy()
    u0[1:-1, 1:-1] = u0[1:-1, 1:-1] + diffusion_term
    # on bound
    u0 = boundary_condition(u0)
    return u0


def plot(x, u, t):
    fig, ax = plt.subplots()
    cs = ax.pcolormesh(x[0], x[1], u, cmap='hot', shading='auto')
    ax.contour(x[0], x[1], u, cmap='rainbow')
    fig.colorbar(cs, ax=ax)
    fig.savefig(f"{os.path.dirname(__file__)}/plot/{t}.png")
    plt.clf()
    plt.close()


def main():
    dx = dy = 0.1
    lim_x = (-10, 10 + dx)
    lim_y = (-10, 10 + dy)
    x = create_coordinate(lim_x, lim_y, dx)
    u0 = np.zeros(x[0].shape)
    # initial condition
    for i in range(x[0].shape[0]):
        for j in range(x[0].shape[1]):
            r = np.sqrt(x[0][i, j] ** 2 + x[1][i, j] ** 2)
            if 5 > r > 4:
                u0[i, j] = 1000
    u = [u0]
    for t in tqdm(range(100)):
        plot(x, u[t], t)
        u0 = diffusion(u[t], dx, dy)
        u.append(u0)


if __name__ == '__main__':
    main()
