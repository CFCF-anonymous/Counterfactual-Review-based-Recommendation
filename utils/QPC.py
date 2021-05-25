import numpy as np
import torch
import sys
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


def solve_QP(grad_list, c_G, c_h):
    l = len(grad_list)
    c_l = len(c_G)

    x = grad_list.cpu().data.numpy()
    y = np.dot(x, x.transpose())
    y = y.astype(np.double)

    Q = 2 * matrix(y)
    p = matrix([0.0] * l)

    #G = matrix([[-1.0, 0.0], [0.0, -1.0]])
    #h = matrix([0.0, 0.0])

    G_tmp = []
    h_tmp = []
    for i in range(l):
        x = [0.0]*l
        x[i] = -1
        G_tmp.append(x)
        h_tmp.append(0.0)
    for i in range(c_l):
        G_tmp.append(c_G[i])
        h_tmp.append(c_h[i])

    G = matrix(np.array(G_tmp))
    h = matrix(h_tmp)
    A = matrix([1.0] * l, (1, l))
    b = matrix(1.0)
    sol = solvers.qp(Q, p, G, h, A, b)
    return sol['x'], 0.0


if __name__ == "__main__":
    print('ok')
