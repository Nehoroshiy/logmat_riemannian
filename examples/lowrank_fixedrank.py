import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import theano

from scipy.sparse import coo_matrix, random as sparse_rnd
from scipy.sparse.linalg import norm as sparse_norm

import theano.tensor as T
from pymanopt import Problem
from pymanopt.manifolds import FixedRankEmbeeded
from pymanopt.solvers import TrustRegions, ConjugateGradient, SteepestDescent


def _bootstrap_problem(A, k, minstepsize=1e-9):
    m, n = A.shape
    manifold = FixedRankEmbeeded(m, n, k)
    #solver = TrustRegions(maxiter=500, minstepsize=1e-6)
    solver = SteepestDescent(maxiter=500, minstepsize=minstepsize)
    #solver = ConjugateGradient(maxiter=500, minstepsize=minstepsize)
    return manifold, solver


def dense_low_rank_approximation(A, k):
    manifold, solver = _bootstrap_problem(A, k)

    def cost(X):
        XU, XS, XV = X
        return 0.5 * la.norm(XU.dot(XS).dot(XV) - A)**2

    def egrad(X):
        XU, XS, XV = X
        return XU.dot(XS).dot(XV) - A

    def ehess(X, Z):
        XU, XS, XV = X
        ZUp, ZM, ZVp = Z
        U = np.hstack((XU.dot(ZM) + Z.Up, X.U))
        S = np.eye(2*XS.shape[0])
        V = np.vstack((XV, ZVp))
        return U.dot(S).dot(V)

    problem = Problem(man=manifold, cost=cost, egrad=egrad, ehess=ehess)
    return solver.solve(problem)


def low_rank_matrix_approximation_theano(A, k, norm_ord, minstepsize=1e-9):
    manifold, solver = _bootstrap_problem(A, k, minstepsize)

    U, S, V = [T.matrix(sym) for sym in ['U', 'S', 'V']]
    if norm_ord == 'fro':
        cost = T.sum((U.dot(S).dot(V) - A)**2)
    elif norm_ord == 'spectral':
        cost = (U.dot(S).dot(V) - A).norm(2)
    elif norm_ord == 'abs':
        cost = (U.dot(S).dot(V) - A).norm(1)
    else:
        mat = U.dot(S).dot(V) - A
        cost = T.diag(mat.T.dot(mat)).norm(L=norm_ord)#T.sum(T.nlinalg.svd(U.dot(S).dot(V) - A, full_matrices=False)[1])


    problem = Problem(man=manifold, theano_cost=cost, theano_arg=[U, S, V])
    return solver.solve(problem)


def generate_sigma_set(shape, percent):
    return sparse_rnd(*shape, density=percent).nonzero()


if __name__ == "__main__":
    # Generate random problem data.
    m, n = 4, 4
    k = 2
    U = rnd.randn(m, k)
    V = rnd.randn(k, n)
    A = U.dot(V)

    factors = []
    norm_ords = ['fro', 'spectral', 3, 4, 5, 6, 10, 'abs']
    for norm_ord in norm_ords:
        u, s, v = low_rank_matrix_approximation_theano(A, k, norm_ord, minstepsize=1e-16)

        print("for norm ord {} \t || USV - A ||: {}".format(norm_ord, la.norm(u.dot(s).dot(v) - A)))
        factors.append((u, s, v))
    print("Now print norms of factor delta in all listed ords:")

    norms = []
    for factor in factors:
        u, s, v = factor
        full = u.dot(s).dot(v)
        appendable = []
        for norm_ord in norm_ords:
            if norm_ord == 'fro':
                cur_norm = la.norm(full, ord='fro')
            elif norm_ord == 'spectral':
                cur_norm = la.norm(full, ord=2)
            elif norm_ord == 'abs':
                cur_norm = la.norm(full, ord=1)
            else:
                cur_norm = la.norm(np.diag(full.T.dot(full)), norm_ord)
            appendable.append(cur_norm)
        #norms.append([la.norm(u.dot(s).dot(v), ord=norm_ord) for norm_ord in norm_ords])
        norms.append(appendable)
        print(["|| ({}) ||: {}, ".format(norm_ord, norm_value) for (norm_ord, norm_value) in zip(norm_ords, norms[-1])])
    u, s, v = factors[0]
    for i, factor in enumerate(factors[1:]):
        u1, s1, v1 = factor
        print("0 and {} factors are close? {}".format(i+1, np.allclose(u.dot(s).dot(v), u1.dot(s1).dot(v1))))