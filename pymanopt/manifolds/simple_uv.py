import warnings

import math
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.stiefel import Stiefel

class SimpleUV(Manifold):
    """
    Manifold of m-by-n real matrices of fixed rank k. This follows the
    embedded geometry described in Bart Vandereycken's 2013 paper:
    "Low-rank matrix completion by Riemannian optimization".

    Paper link: http://arxiv.org/pdf/1209.3834.pdf

    A point X on the manifold is represented as a structure with three
    fields: U, S and V. The matrices U (mxk) and V (kxn) are orthonormal,
    while the matrix S (kxk) is any /diagonal/, full rank matrix.
    Following the SVD formalism, X = U*S*V. Note that the diagonal entries
    of S are not constrained to be nonnegative.

    Tangent vectors are represented as a structure with three fields: Up, M
    and Vp. The matrices Up (mxn) and Vp (kxn) obey Up*U = 0 and Vp*V = 0.
    The matrix M (kxk) is arbitrary. Such a structure corresponds to the
    following tangent vector in the ambient space of mxn matrices:
      Z = U*M*V + Up*V + U*Vp
    where (U, S, V) is the current point and (Up, M, Vp) is the tangent
    vector at that point.

    Vectors in the ambient space are best represented as mxn matrices. If
    these are low-rank, they may also be represented as structures with
    U, S, V fields, such that Z = U*S*V. Their are no resitrictions on what
    U, S and V are, as long as their product as indicated yields a real, mxn
    matrix.

    The chosen geometry yields a Riemannian submanifold of the embedding
    space R^(mxn) equipped with the usual trace (Frobenius) inner product.


    Please cite the Manopt paper as well as the research paper:
        @Article{vandereycken2013lowrank,
          Title   = {Low-rank matrix completion by {Riemannian} optimization},
          Author  = {Vandereycken, B.},
          Journal = {SIAM Journal on Optimization},
          Year    = {2013},
          Number  = {2},
          Pages   = {1214--1236},
          Volume  = {23},
          Doi     = {10.1137/110845768}
        }
    """
    def __init__(self, m, n, k):
        self._m = m
        self._n = n
        self._k = k
        self.stiefelm = Stiefel(self._m, self._k)
        self.stiefeln = Stiefel(self._n, self._k)
        self._name = ('Manifold of {:d}x{:d} matrices of rank {:d}'.format(m, n, k))

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return (self._m + self._n - self._k) * self._k

    @property
    def typicaldist(self):
        return self.dim

    def inner(self, X, G, H):
        Gu, Gv = G
        Hu, Hv = H
        return np.trace(Gu.dot(Gv.dot(Hv.T)).dot(Hu.T))

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        raise NotImplementedError("method is not imlemented")

    def tangent(self, X, Z):
        raise NotImplementedError("method is not imlemented")

    def proj(self, X, Z):
        raise NotImplementedError("method is not imlemented")

    def from_partial(self, X, dX):
        return dX

    def egrad2rgrad(self, X, Z):
        return NotImplementedError("method is not imlemented")

    def ehess2rhess(self, X, egrad, ehess, H):
        raise NotImplementedError("method is not imlemented")

    def tangent2ambient(self, X, Z):
        raise NotImplementedError("method is not imlemented")

    def retr(self, X, Z):
        XU, XV = X
        ZU, ZV = Z
        return (XU + ZU, XV + ZV)

    def exp(self, X, U):
        warnings.warn("Exponential map for fixed-rank matrix"
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(X, U)

    def rand(self):
        U = self.stiefelm.rand()
        V = self.stiefeln.rand().T
        return (U, V)

    def randvec(self, X):
        H = self.rand()
        P = self.proj(X, H)
        return self._normalize(P)

    def zerovec(self, X):
        return (np.zeros((self._m, self._k)),
                np.zeros((self._k, self._k)),
                np.zeros((self._k, self._n)))

    def vec(self, X, Z):
        Zamb = self.tangent2ambient(X, Z)
        U, S, V = Zamb
        Zamb_mat = U.dot(S).dot(V)
        Zvec = Zamb_mat.T.ravel()
        return Zvec

    def _normalize(self, P):
        Up = P.Up
        M = P.M / np.linalg.norm(P.M)
        Vp = P.Vp
        return (Up, M, Vp)

    def log(self, X, Y):
        raise NotImplementedError

    def transp(self, x1, x2, d):
        return self.proj(x2, self.tangent2ambient(x1, d))

    def cubic_root(self, x):
        if x > 0:
            return math.pow(x, float(1)/3)
        elif x < 0:
            return -math.pow(abs(x), float(1)/3)
        else:
            return 0

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        Up1, Vp1 = u1
        #a1s = self.cubic_root(a1)
        a1s = a1
        if u2 is None and a2 is None:
            Up = a1s * Up1
            Vp = a1s * Vp1
            return (Up, Vp)
        elif None not in [a1, u1, a2, u2]:
            Up2, Vp2 = u2
            #a2s = self.cubic_root(a2)
            a2s = a2
            Up = a1s * Up1 + a2s * Up2
            Vp = a1s * Vp1 + a2s * Vp2
            return (Up, Vp)
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')
