import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from scipy.sparse import coo_matrix, random as sparse_rnd
from scipy.sparse.linalg import norm as sparse_norm

import theano.tensor as T
from pymanopt import Problem
from pymanopt.manifolds import FixedRankEmbeeded
from pymanopt.manifolds.fixed_rank import ManifoldElement
from pymanopt.solvers import TrustRegions, ConjugateGradient


def _bootstrap_problem(A, k):
    m, n = A.shape
    manifold = FixedRankEmbeeded(m, n, k)
    #solver = TrustRegions(maxiter=500, minstepsize=1e-6)
    solver = ConjugateGradient(maxiter=500, minstepsize=1e-9)
    return manifold, solver


def dense_low_rank_approximation(A, k):
    manifold, solver = _bootstrap_problem(A, k)

    def cost(X):
        return 0.5 * la.norm(X.U.dot(X.S).dot(X.V) - A)**2

    def egrad(X):
        return X.U.dot(X.S).dot(X.V) - A

    def ehess(X, Z):
        U = np.hstack((X.U.dot(Z.M) + Z.Up, X.U))
        S = np.eye(2*X.S.shape[0])
        V = np.vstack((X.V, Z.Vp))
        return U.dot(S).dot(V)

    problem = Problem(man=manifold, cost=cost, egrad=egrad, ehess=ehess)
    return solver.solve(problem)


def low_rank_matrix_approximation(A, sigma_set, k):
    density = 1.0 * len(sigma_set[0]) / A.size
    a_data = np.array(A[sigma_set]).ravel()
    a_sparse = coo_matrix((a_data, sigma_set), A.shape).tocsr()
    manifold, solver = _bootstrap_problem(a_sparse, k)

    def cost(X):
        return 0.5 * sparse_norm(X[sigma_set] - a_sparse) ** 2

    def egrad(X):
        return X[sigma_set] - a_sparse

    def ehess(X, Z):
        U = np.hstack((X.U.dot(Z.M) + Z.Up, X.U))
        S = np.eye(2*X.S.shape[0])
        V = np.vstack((X.V, Z.Vp))
        return ManifoldElement(U, S, V)[sigma_set]

    problem = Problem(man=manifold, cost=cost, egrad=egrad, ehess=ehess)
    return solver.solve(problem)


def low_rank_matrix_approximation_theano(A, k):
    manifold, solver = _bootstrap_problem(A, k)

    #Y = T.matrix()
    #cost = T.sum((T.dot(Y, Y.T) - A) ** 2)
    #U = T.matrix()
    #S = T.matrix()
    #V = T.matrix()
    #X = T.dot(T.dot(U, S), V)
    Y = T.matrix()

    #cost = T.sum((T.dot(T.dot(U, S), V) - A) ** 2)
    cost = T.sum((Y - A)**2)

    problem = Problem(man=manifold, theano_cost=cost, theano_arg=Y)
    return solver.solve(problem)


def generate_sigma_set(shape, percent):
    return sparse_rnd(*shape, density=percent).nonzero()

if __name__ == "__main__":
    # Generate random problem data.
    m, n = 1000, 500
    k = 5
    U = rnd.randn(m, k)
    V = rnd.randn(k, n)
    A = U.dot(V)

    sigma_set = generate_sigma_set(A.shape, 0.05)
    # Solve the sparse problem with pymanopt.
    x_sparse = low_rank_matrix_approximation(A, sigma_set, k)
    print
    # Solve the dense problem with pymanopt.
    x_dense = dense_low_rank_approximation(A, k)
    print
    #x_theano = low_rank_matrix_approximation_theano(A, k)


    # Print information about the solution.
    print

    print("rank of A: %d" % la.matrix_rank(A))
    print("rank of x_sparse: %d" % la.matrix_rank(x_sparse.U.dot(x_sparse.S).dot(x_sparse.V)))
    print("rank of x_dense: %d" % la.matrix_rank(x_dense.U.dot(x_dense.S).dot(x_dense.V)))
    #print("rank of x_theano: %d" % la.matrix_rank(x_theano))
    full_matrix_from_sparse = x_sparse.U.dot(x_sparse.S).dot(x_sparse.V)
    full_matrix_from_dense = x_dense.U.dot(x_dense.S).dot(x_dense.V)
    print('[sparse] norm eps: {}'.format(la.norm(A - full_matrix_from_sparse) / la.norm(A)))
    print('[sparse] max diff: {}'.format(np.max(np.abs(A - full_matrix_from_sparse))))
    print('[dense]  norm eps: {}'.format(la.norm(A - full_matrix_from_dense) / la.norm(A)))
    print('[dense] max diff: {}'.format(np.max(np.abs(A - full_matrix_from_dense))))
    #print('[sparse] norm eps: {}'.format(la.norm(A - full_matrix_from_sparse) / la.norm(A)))
    #print('[sparse] max diff: {}'.format(np.max(np.abs(A - full_matrix_from_sparse))))
