import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from scipy.sparse import coo_matrix
from scipy.linalg import solve_lyapunov as lyap, rq, solve_sylvester as sylv

from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.stiefel import Stiefel


def cov(m):
    return m.T.dot(m)


def mldivide(a, b):
    if a.shape[0] == a.shape[1]:
        return la.solve(a, b)
    else:
        return la.lstsq(a, b)


def mrdivide(a, b):
    return mldivide(a.T, b.T).T

def symm(a):
    return 0.5 * (a + a.T)


class FixedRankEmbeeded2Factors(Manifold):
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
        self._name = ('Manifold of {:d}x{:d} matrices of rank {:d}'.format(m, n, k))

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return (self._m + self._n - self._k) * self._k

    @property
    def typicaldist(self):
        return 10*self._k

    def inner(self, X, G, H):
        XL, XR = X
        LtL, RtR = cov(XL), cov(XR)
        GL, GR = G
        HL, HR = H
        return np.trace(mldivide(LtL, GL.T.dot(HL))) + np.trace(mldivide(RtR, GR.T.dot(HR)))

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        raise NotImplementedError

    def egrad2rgrad(self, X, Z):
        XL, XR = X
        ZL, ZR = Z
        LtL, RtR = cov(XL), cov(XR)
        rgradL = ZL.dot(LtL)
        rgradR = ZR.dot(RtR)
        return rgradL, rgradR

    def from_partial(self, X, dX):
        return self.egrad2rgrad(X, dX)

    def ehess2rhess(self, X, egrad, ehess, H):
        XL, XR = X
        HL, HR = H
        egradL, egradR = egrad
        ehessL, ehessR = ehess
        LtL, RtR = cov(XL), cov(XR)

        # Riemannian gradient computation
        rgradL, rgradR = self.egrad2rgrad(X, egrad);

        # Directional derivative of the Riemannian gradient.
        HessL = ehessL.dot(LtL) + 2*egradL.dot(symm(HL.T.dot(XL)))
        HessR = ehessR.dot(RtR) + 2*egradR.dot(symm(HR.T.dot(XR)))

        # We need a correction term for the non-constant metric.
        HessL -= (rgradL.dot(mldivide(LtL, symm(XL.T.dot(HL))))
                  - HL.dot(mldivide(LtL, symm(XL.T.dot(rgradL))))
                  + XL.dot(mldivide(LtL, symm(HL.T.dot(rgradL)))))
        HessR -= (rgradR.dot(mldivide(RtR, symm(XR.T.dot(HR))))
                  - HR.dot(mldivide(RtR, symm(XR.T.dot(rgradR))))
                  + XR.dot(mldivide(RtR, symm(HR.T.dot(rgradR)))))

        # Projection onto the horizontal space.
        return self.proj(X, (HessL, HessR))

    def proj(self, X, Z):
        XL, XR = X
        ZL, ZR = Z
        LtL, RtR = cov(XL), cov(XR)
        SS = LtL.dot(RtR)
        AS = LtL.dot(XR.T.dot(ZR)) - (ZL.T.dot(XL)).dot(RtR)
        Omega = sylv(SS, SS, AS)
        return ZL + XL.dot(Omega.T), ZR - XR.dot(Omega)

    def tangent(self, X, Z):
        return self.proj(X, Z)

    def tangent2ambient(self, X, Z):
        return Z

    def retr(self, X, Z):
        XL, XR = X
        ZL, ZR = Z

        YL, YR = XL + ZL, XR + ZR

        # Numerical conditioning step: A simpler version.
        # We need to ensure that L and R do not have very relative
        # skewed norms.

        scaling = np.sqrt(la.norm(XL, 'fro') / la.norm(XR, 'fro'))
        YL /= scaling
        YR *= scaling

        # Y = prepare(Y)
        return YL, YR

    def exp(self, X, U):
        warnings.warn("Exponential map for fixed-rank matrix"
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(X, U)

    def rand(self):
        U = rnd.randn(self._m, self._k)
        V = rnd.randn(self._n, self._k)
        # X = prepare(X)
        return U, V

    def randvec(self, X):
        ZL, ZR = rnd.randn(self._m, self._k), rnd.randn(self._n, self._k)
        Z = self.proj(X, (ZL, ZR))
        nrm = self.norm(X, Z)
        return ZL / nrm, ZR / nrm

    def zerovec(self, X):
        return (np.zeros((self._m, self._k)),
                np.zeros((self._k, self._n)))

    def vec(self, X, Z):
        raise NotImplementedError

    def log(self, X, Y):
        raise NotImplementedError

    def transp(self, x1, x2, d):
        return self.proj(x2, self.tangent2ambient(x1, d))

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        UL1, UR1 = u1
        if u2 is None and a2 is None:
            return UL1 * a1, UR1 * a1
        elif None not in [a1, u1, a2, u2]:
            UL2, UR2 = u2
            return UL1 * a1 + UL2 * a2, UR1 * a1 + UR2 * a2
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')
