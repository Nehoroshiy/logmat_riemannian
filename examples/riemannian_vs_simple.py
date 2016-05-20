import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import theano

from scipy.sparse import coo_matrix, random as sparse_rnd
from scipy.sparse.linalg import norm as sparse_norm

import theano.tensor as T
from pymanopt import Problem
from pymanopt.manifolds import FixedRankEmbeeded, Simple, FixedRankEmbeeded2Factors
from pymanopt.solvers import TrustRegions, ConjugateGradient, SteepestDescent


def _bootstrap_problem(A, k, minstepsize=1e-9, man_type='fixed'):
    m, n = A.shape
    if man_type == 'fixed':
        manifold = FixedRankEmbeeded(m, n, k)
    if man_type == 'fixed2':
        manifold = FixedRankEmbeeded2Factors(m, n, k)
    elif man_type == 'simple':
        manifold = Simple(m, n, k)
    #solver = TrustRegions(maxiter=500, minstepsize=1e-6)
    solver = ConjugateGradient(maxiter=500, minstepsize=minstepsize)
    return manifold, solver

def low_rank_matrix_approximation_theano(A, k, man_type='fixed', minstepsize=1e-9):
    manifold, solver = _bootstrap_problem(A, k, minstepsize, man_type)

    if man_type == 'fixed2':
        theano_arg = [T.matrix(sym) for sym in ['U', 'V']]
        U, V = theano_arg
        cost = T.sum((U.dot(V.T) - A)**2)
    else:
        theano_arg = [T.matrix(sym) for sym in ['U', 'S', 'V']]
        U, S, V = theano_arg
        cost = T.sum((U.dot(S).dot(V) - A)**2)

    problem = Problem(man=manifold, theano_cost=cost, theano_arg=theano_arg)
    return solver.solve(problem)


if __name__ == "__main__":
    # Generate random problem data.
    m, n = 10, 10
    k = 2
    U = rnd.randn(m, k)
    V = rnd.randn(k, n)
    A = U.dot(V)

    import matplotlib.pyplot as plt
    import seaborn as sns

    #rep = 50
    #types = ['fixed', 'simple']
    #for tp in types:
    #    for i in range(rep)
    #    u, s, v, iter = low_rank_matrix_approximation_theano(A, k, tp, minstepsize=1e-16)
    #    print("for {} || USV' - A || = {}".format(tp, la.norm(u.dot(s.dot(v)) - A)))

    (u, v) = low_rank_matrix_approximation_theano(A, k, 'fixed2')
    #print(iter)
    print("|| USV' - A || = {}".format(la.norm(u.dot(v.T) - A)))