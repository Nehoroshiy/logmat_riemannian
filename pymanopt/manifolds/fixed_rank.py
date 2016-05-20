import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from scipy.sparse import coo_matrix
from scipy.linalg import solve_lyapunov as lyap, rq

from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds.stiefel import Stiefel


class SymFixedRankYY(Manifold):
    """
    Manifold of n-by-n symmetric positive semidefinite matrices of rank k.

    A point X on the manifold is parameterized as YY^T where Y is a matrix of
    size nxk. As such, X is symmetric, positive semidefinite. We restrict to
    full-rank Y's, such that X has rank exactly k. The point X is numerically
    represented by Y (this is more efficient than working with X, which may
    be big). Tangent vectors are represented as matrices of the same size as
    Y, call them Ydot, so that Xdot = Y Ydot' + Ydot Y. The metric is the
    canonical Euclidean metric on Y.

    Since for any orthogonal Q of size k, it holds that (YQ)(YQ)' = YY',
    we "group" all matrices of the form YQ in an equivalence class. The set
    of equivalence classes is a Riemannian quotient manifold, implemented
    here.

    Notice that this manifold is not complete: if optimization leads Y to be
    rank-deficient, the geometry will break down. Hence, this geometry should
    only be used if it is expected that the points of interest will have rank
    exactly k. Reduce k if that is not the case.

    An alternative, complete, geometry for positive semidefinite matrices of
    rank k is described in Bonnabel and Sepulchre 2009, "Riemannian Metric
    and Geometric Mean for Positive Semidefinite Matrices of Fixed Rank",
    SIAM Journal on Matrix Analysis and Applications.


    The geometry implemented here is the simplest case of the 2010 paper:
    M. Journee, P.-A. Absil, F. Bach and R. Sepulchre,
    "Low-Rank Optimization on the Cone of Positive Semidefinite Matrices".
    Paper link: http://www.di.ens.fr/~fbach/journee2010_sdp.pdf
    """
    def __init__(self, n, k):
        self._n = n
        self._k = k

        self._name = ("YY' quotient manifold of {:d}x{:d} psd matrices of "
                      "rank {:d}".format(n, n, k))

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        n = self._n
        k = self._k
        return k * n - k * (k - 1) / 2

    @property
    def typicaldist(self):
        return 10 + self._k

    def inner(self, Y, U, V):
        # Euclidean metric on the total space.
        return float(np.tensordot(U, V))

    def norm(self, Y, U):
        return la.norm(U, "fro")

    def dist(self, U, V):
        raise NotImplementedError

    def proj(self, Y, H):
        # Projection onto the horizontal space
        YtY = Y.T.dot(Y)
        AS = Y.T.dot(H) - H.T.dot(Y)
        Omega = lyap(YtY, -AS)
        return H - Y.dot(Omega)
    
    tangent = proj

    def egrad2rgrad(self, Y, H):
        return H

    def ehess2rhess(self, Y, egrad, ehess, U):
        return self.proj(Y, ehess)

    def exp(self, Y, U):
        warnings.warn("Exponential map for symmetric, fixed-rank "
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(Y, U)
    
    def lincomb(self, X, a1, u1, a2=None, u2=None):
        if u2 is None and a2 is None:
            return a1 * u1
        elif None not in [a1, u1, a2, u2]:
            return a1 * u1 + a2 * u2
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')

    def retr(self, Y, U):
        return Y + U

    def log(self, Y, U):
        raise NotImplementedError

    def rand(self):
        return rnd.randn(self._n, self._k)

    def randvec(self, Y):
        H = self.rand()
        P = self.proj(Y, H)
        return self._normalize(P)

    def transp(self, Y, Z, U):
        return self.proj(Z, U)

    def _normalize(self, Y):
        return Y / self.norm(None, Y)


class SymFixedRankYYComplex(SymFixedRankYY):
    """
    Manifold of n x n complex Hermitian pos. semidefinite matrices of rank k.

    Manifold of n-by-n complex Hermitian positive semidefinite matrices of
    fixed rank k. This follows the quotient geometry described
    in Sarod Yatawatta's 2013 paper:
    "Radio interferometric calibration using a Riemannian manifold", ICASSP.

    Paper link: http://dx.doi.org/10.1109/ICASSP.2013.6638382.

    A point X on the manifold M is parameterized as YY^*, where
    Y is a complex matrix of size nxk. For any point Y on the manifold M,
    given any kxk complex unitary matrix U, we say Y*U  is equivalent to Y,
    i.e., YY^* does not change. Therefore, M is the set of equivalence
    classes and is a Riemannian quotient manifold C^{nk}/SU(k).
    The metric is the usual real-trace inner product, that is,
    it is the usual metric for the complex plane identified with R^2.

    Notice that this manifold is not complete: if optimization leads Y to be
    rank-deficient, the geometry will break down. Hence, this geometry should
    only be used if it is expected that the points of interest will have rank
    exactly k. Reduce k if that is not the case.
    """
    def __init__(self, *args, **kwargs):
        super(SymFixedRankYYComplex, self).__init__(*args, **kwargs)

        n = self._n
        k = self._k
        self._name = ("YY' quotient manifold of Hermitian {:d}x{:d} complex "
                      "matrices of rank {:d}".format(n, n, k))

    @property
    def dim(self):
        n = self._n
        k = self._k
        return 2 * k * n - k * k

    def inner(self, Y, U, V):
        return 2 * float(np.tensordot(U, V).real)

    def norm(self, Y, U):
        return np.sqrt(self.inner(Y, U, U))

    def dist(self, U, V):
        S, _, D = la.svd(V.T.conj().dot(U))
        E = U - V.dot(S).dot(D)  # numpy's svd returns D.H
        return self.inner(None, E, E) / 2

    def exp(self, Y, U):
        # We only overload this to adjust the warning.
        warnings.warn("Exponential map for symmetric, fixed-rank complex "
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(Y, U)

    def rand(self):
        rand_ = super(SymFixedRankYYComplex, self).rand
        return rand_() + 1j * rand_()

'''
class ManifoldElement():
    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V

    def dot(self, other):
        if isinstance(other, ManifoldElement):
            mid = self.S.dot(self.V.dot(other.U)).dot(other.S)
            U, S, V = la.svd(mid, full_matrices=False)
            return ManifoldElement(self.U.dot(U), np.diag(self.S), V.dot(self.V))
        else:
            raise ValueError('dot must be performed on ManifoldElements.')

    def __getitem__(self, item):
        if hasattr(item, '__len__') and len(item) == 2 and len(item[0]) == len(item[1]):
            rows = self.U[item[0], :].dot(self.S)
            cols = self.V[:, item[1]]
            data = (rows * cols.T).sum(1)
            assert(data.size == len(item[0]))
            shape = (self.U.shape[0], self.V.shape[1])
            return coo_matrix((data, tuple(item)), shape=shape).tocsr()
        else:
            raise ValueError('__getitem__ now supports only indices set')

    @property
    def T(self):
        return ManifoldElement(self.V.T, self.S.T, self.U.T)


class TangentVector():
    def __init__(self, Up, M, Vp):
        self.Up = Up
        self.M = M
        self.Vp = Vp

    def __neg__(self):
        return TangentVector(-self.Up, -self.M, -self.Vp)

    def __add__(self, other):
        if isinstance(other, TangentVector):
            return TangentVector(self.Up + other.Up, self.M + other.M, self.Vp + other.Vp)

    def __sub__(self, other):
        if isinstance(other, TangentVector):
            return TangentVector(self.Up - other.Up, self.M - other.M, self.Vp - other.Vp)

    def __mul__(self, other):
        if np.isscalar(other):
            return TangentVector(self.Up * other, self.M * other, self.Vp * other)
        else:
            raise ValueError('TangentVector supports only multiplying by scalar')

    def __rmul__(self, other):
        return self.__mul__(other)


class FixedRankEmbeeded(Manifold):
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
        return G.M.ravel().dot(H.M.ravel()) + \
               G.Up.ravel().dot(H.Up.ravel()) + \
               G.Vp.ravel().dot(H.Vp.ravel())

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        raise NotImplementedError

    def tangent(self, X, Z):
        Z.Up = Z.Up - X.U.dot(X.U.T.dot(Z.Up))
        Z.Vp = Z.Vp - (Z.Vp.dot(X.V.T)).dot(X.V)

    def apply_ambient(self, Z, W):
        if isinstance(Z, ManifoldElement):
            return Z.U.dot(Z.S.dot(Z.V.dot(W)))
        if isinstance(Z, TangentVector):
            return Z.Up.dot(Z.M.dot(Z.Vp.dot(W)))
        else:
            return Z.dot(W)

    def apply_ambient_transpose(self, Z, W):
        if isinstance(Z, ManifoldElement):
            return Z.V.T.dot(Z.S.T.dot(Z.U.T.dot(W)))
        if isinstance(Z, TangentVector):
            return Z.Vp.T.dot(Z.M.T.dot(Z.Up.T.dot(W)))
        else:
            return Z.T.dot(W)

    def proj(self, X, Z):
        ZV = self.apply_ambient(Z, X.V.T)
        UtZV = X.U.T.dot(ZV)
        ZtU = self.apply_ambient_transpose(Z, X.U).T

        Zproj = TangentVector(ZV - X.U.dot(UtZV), UtZV, ZtU - (UtZV.dot(X.V)))
        return Zproj

    def egrad2rgrad(self, X, Z):
        return self.proj(X, Z)

    def ehess2rhess(self, X, egrad, ehess, H):
        # Euclidean part
        rhess = self.proj(X, ehess)
        Sinv = np.diag(1.0 / np.diag(X.S))

        # Curvature part
        T = self.apply_ambient(egrad, H.Vp.T).dot(Sinv)
        rhess.Up += (T - X.U.dot(X.U.T.dot(T)))
        T = self.apply_ambient_transpose(egrad, H.Up).dot(Sinv)
        rhess.Vp += (T - X.V.T.dot(X.V.dot(T))).T
        return rhess

    def tangent2ambient(self, X, Z):
        U = np.hstack((X.U.dot(Z.M) + Z.Up, X.U))
        S = np.eye(2*self._k)
        V = np.vstack((X.V, Z.Vp))
        return ManifoldElement(U, S, V)

    def retr(self, X, Z, t=None):
        if t is None:
            t = 1.0
        Qu, Ru = la.qr(Z.Up)
        Rv, Qv = rq(Z.Vp, mode='economic')

        zero_block = np.zeros((Ru.shape[0], Rv.shape[1]))
        block_mat = np.array(np.bmat([[X.S + t * Z.M, t * Rv],
                                     [t * Ru, zero_block]]))

        Ut, St, Vt = la.svd(block_mat, full_matrices=False)

        U = np.hstack((X.U, Qu)).dot(Ut[:, :self._k])
        V = Vt[:self._k, :].dot(np.vstack((X.V, Qv)))
        # add some machinery eps to get a slightly perturbed element of a manifold
        # even if we have some zeros in S
        S = np.diag(St[:self._k]) + np.diag(np.spacing(1) * np.ones(self._k))
        return ManifoldElement(U, S, V)

    def exp(self, X, U, t=None):
        warnings.warn("Exponential map for fixed-rank matrix"
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(X, U, t)

    def rand(self):
        U = self.stiefelm.rand()
        V = self.stiefeln.rand().T
        s = np.sort(np.random.random(self._k))[::-1]
        S = np.diag(s / la.norm(s) + np.spacing(1) * np.ones(self._k))
        return ManifoldElement(U, S, V)

    def randvec(self, X):
        H = self.rand()
        P = self.proj(X, H)
        return self._normalize(P)

    def zerovec(self, X):
        return TangentVector(np.zeros((self._m, self._k)),
                             np.zeros((self._k, self._k)),
                             np.zeros((self._k, self._n)))

    def vec(self, X, Z):
        Zamb = self.tangent2ambient(X, Z)
        Zamb_mat = Zamb.U.dot(Zamb.S).dot(Zamb.V)
        Zvec = Zamb_mat.T.ravel()
        return Zvec

    def _normalize(self, P):
        Up = P.Up
        M = P.M / la.norm(P.M)
        Vp = P.Vp
        return TangentVector(Up, M, Vp)

    def log(self, X, Y):
        raise NotImplementedError

    def transp(self, x1, x2, d):
        return self.proj(x2, self.tangent2ambient(x1, d))

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        if u2 is None and a2 is None:
            Up = a1 * u1.Up
            Vp = a1 * u1.Vp
            M = a1 * u1.M
            return TangentVector(Up, M, Vp)
        elif None not in [a1, u1, a2, u2]:
            Up = a1 * u1.Up + a2 * u2.Up
            Vp = a1 * u1.Vp + a2 * u2.Vp
            M = a1 * u1.M + a2 * u2.M
            return TangentVector(Up, M, Vp)
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')
'''

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

srnd = RandomStreams(rnd.randint(0, 1000))

import warnings
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import copy

import theano
from theano import tensor


class FixedRankEmbeeded(Manifold):
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
        Gm, Gup, Gvp = G
        Hm, Hup, Hvp = H
        return Gm.ravel().dot(Hm.ravel()) + \
               Gup.ravel().dot(Hup.ravel()) + \
               Gvp.ravel().dot(Hvp.ravel())

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        raise NotImplementedError

    def tangent(self, X, Z):
        U, S, V = X
        Up, M, Vp = Z
        Up -= U.dot(U.T.dot(Up))
        Vp -= (Vp.dot(V.T)).dot(V)
        return Up, M, Vp

    def apply_ambient(self, Z, W, arg_type="mat"):
        if arg_type == "tan_vec":
            Up, M, Vp = Z
            return Up.dot(M.dot(Vp.dot(W)))
        elif arg_type == "mat":
            return Z.dot(W)
        else:
            raise TypeError("'type' must be 'mat', or 'tan_vec'")

    def apply_ambient_transpose(self, Z, W, arg_type="mat"):
        if arg_type == "tan_vec":
            Up, M, Vp = Z
            return Vp.T.dot(M.T.dot(Up.T.dot(W)))
        elif arg_type == "mat":
            return Z.T.dot(W)
        else:
            raise TypeError("'type' must be 'mat', or 'tan_vec'")

    def proj(self, X, Z):
        if isinstance(Z, np.ndarray):
            arg_type = "mat"
        elif isinstance(Z, list) or isinstance(Z, tuple):
            if all([isinstance(z_elem, np.ndarray) for z_elem in Z]):
                arg_type = "tan_vec"
            else:
                raise TypeError("Z must me a tuple of ndarrays or single ndarray")
        else:
            raise TypeError("Z must me a tuple of ndarrays or single ndarray")
        U, S, V = X
        ZV = self.apply_ambient(Z, V.T, arg_type=arg_type)
        UtZV = U.T.dot(ZV)
        ZtU = self.apply_ambient_transpose(Z, U, arg_type=arg_type).T

        Zproj = (ZV - U.dot(UtZV), UtZV, ZtU - (UtZV.dot(V)))
        return Zproj

    def from_partial(self, X, dX):
        U, S, V = X
        dU, dS, dV = dX

        ZV = dU.dot(np.diag(1.0 / np.diag(S)))
        UtZV = dS
        ZtU = np.diag(1.0 / np.diag(S)).dot(dV)

        Zproj = (ZV - U.dot(UtZV), UtZV, ZtU - (UtZV.dot(V)))
        return Zproj

    def egrad2rgrad(self, X, Z):
        return self.proj(X, Z)

    def ehess2rhess(self, X, egrad, ehess, H):
        # TODO same problem as tangent
        """
        # Euclidean part
        rhess = self.proj(X, ehess)
        Sinv = tensor.diag(1.0 / tensor.diag(X.S))

        # Curvature part
        T = self.apply_ambient(egrad, H.Vp.T).dot(Sinv)
        rhess.Up += (T - X.U.dot(X.U.T.dot(T)))
        T = self.apply_ambient_transpose(egrad, H.Up).dot(Sinv)
        rhess.Vp += (T - X.V.T.dot(X.V.dot(T))).T
        return rhess
        """
        raise NotImplementedError("method is not imlemented")

    def tangent2ambient(self, X, Z):
        XU, XS, XV = X
        ZUp, ZM, ZVp = Z
        U = np.hstack((XU.dot(ZM) + ZUp, XU))
        S = np.eye(2*self._k)
        V = np.vstack((XV, ZVp))
        return (U, S, V)

    def retr(self, X, Z):
        XU, XS, XV = X
        ZUp, ZM, ZVp = Z
        Qu, Ru = la.qr(ZUp)
        Rv, Qv = rq(ZVp, mode='economic')

        zero_block = np.zeros((Ru.shape[0], Rv.shape[1]))
        block_mat = np.array(np.bmat([[XS + ZM, Rv],
                                     [Ru, zero_block]]))

        Ut, St, Vt = la.svd(block_mat, full_matrices=False)

        U = np.hstack((XU, Qu)).dot(Ut[:, :self._k])
        V = Vt[:self._k, :].dot(np.vstack((XV, Qv)))
        # add some machinery eps to get a slightly perturbed element of a manifold
        # even if we have some zeros in S
        S = np.diag(St[:self._k]) + np.diag(np.spacing(1) * np.ones(self._k))
        return (U, S, V)

    def exp(self, X, U):
        warnings.warn("Exponential map for fixed-rank matrix"
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(X, U)

    def rand(self):
        U = self.stiefelm.rand()
        V = self.stiefeln.rand().T
        s = np.sort(np.random.random(self._k))[::-1]
        S = np.diag(s / la.norm(s) + np.spacing(1) * np.ones(self._k))
        return (U, S, V)

    def randvec(self, X):
        H = self.rand()
        P = self.proj(X, H)
        return self._normalize(P)

    def zerovec(self, X):
        return (tensor.zeros((self._m, self._k)),
                tensor.zeros((self._k, self._k)),
                tensor.zeros((self._k, self._n)))

    def vec(self, X, Z):
        Zamb = self.tangent2ambient(X, Z)
        U, S, V = Zamb
        Zamb_mat = U.dot(S).dot(V)
        Zvec = Zamb_mat.T.ravel()
        return Zvec

    def _normalize(self, P):
        Up = P.Up
        M = P.M / tensor.nlinalg.norm(P.M)
        Vp = P.Vp
        return (Up, M, Vp)

    def log(self, X, Y):
        raise NotImplementedError

    def transp(self, x1, x2, d):
        return self.proj(x2, self.tangent2ambient(x1, d))

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        Up1, M1, Vp1 = u1
        if u2 is None and a2 is None:
            Up = a1 * Up1
            Vp = a1 * Vp1
            M = a1 * M1
            return (Up, M, Vp)
        elif None not in [a1, u1, a2, u2]:
            Up2, M2, Vp2 = u2
            Up = a1 * Up1 + a2 * Up2
            Vp = a1 * Vp1 + a2 * Vp2
            M = a1 * M1 + a2 * M2
            return (Up, M, Vp)
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')
