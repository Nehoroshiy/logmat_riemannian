import time
import numpy as np
import scipy as sp
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import numpy.linalg as la
import numpy.random as rnd
import theano

from scipy.sparse import coo_matrix, random as sparse_rnd
from scipy.sparse.linalg import norm as sparse_norm

import theano.tensor as T
from theano.tensor import slinalg
from pymanopt import Problem
from pymanopt.manifolds import FixedRankEmbeeded2Factors, FixedRankEmbeeded
from pymanopt.solvers import TrustRegions, ConjugateGradient, SteepestDescent

import matplotlib.pyplot as plt

SCONST = 1.0

def sum_lowrank(lowrank_matrix):
    u, v = lowrank_matrix
    return u.sum(0).dot(v.sum(0))


class LineSearchAdaptive(object):
    def __init__(self):
        self._contraction_factor = 0.5
        self._suff_decr = 0.5
        self._max_steps = 10
        self._initial_stepsize = 1

        self._oldalpha = None

    def search(self, objective, man, x, d, f0, df0):
        norm_d = man.norm(x, d)

        if self._oldalpha is not None:
            alpha = self._oldalpha
        else:
            alpha = self._initial_stepsize / norm_d
        print("alpha: {}".format(alpha))
        newx = man.retr(x, man.lincomb(x, alpha, d))
        newf = objective(newx)
        cost_evaluations = 1
        print("func:{}".format(newf))
        print("params:{}".format(f0 + self._suff_decr * alpha * df0))
        while (newf > f0 + self._suff_decr * alpha * df0 and
               cost_evaluations <= self._max_steps):
            # Reduce the step size.
            alpha *= self._contraction_factor

            # Look closer down the line.
            newx = man.retr(x, man.lincomb(x, alpha, d))
            newf = objective(newx)

            cost_evaluations += 1

        if newf > f0:
            alpha = 0
            newx = x

        stepsize = alpha * norm_d

        # Store a suggestion for what the next initial step size trial should
        # be. On average we intend to do only one extra cost evaluation. Notice
        # how the suggestion is not about stepsize but about alpha. This is the
        # reason why this line search is not invariant under rescaling of the
        # search direction d.

        # If things go reasonably well, try to keep pace.
        if cost_evaluations == 2:
            self._oldalpha = alpha
        # If things went very well or we backtracked a lot (meaning the step
        # size is probably quite small), speed up.
        else:
            self._oldalpha = 2 * alpha

        return stepsize, newx



def accurate_log_exp1(x):
    cases = [
        x <= -37.0,
        np.logical_and(x > -37.0, x <= 18.0),
        np.logical_and(x > 18.0, x <= 33.3),
        x > 33.3
    ]
    safe_funcs = [
        lambda x: np.exp(x),
        lambda x: np.log1p(np.exp(x)),
        lambda x: x + np.exp(-(x)),
        lambda x: x
    ]
    return np.piecewise(x, cases, safe_funcs)


def load_matrix_sparse(filename='cut.tsv', num_users=None, num_items=None):
    t0 = time.time()
    matrix_data = pd.read_csv(filename, sep='\t', header=None, names=['i', 'j', 'counts'], dtype={'counts': np.float})
    counts = coo_matrix((matrix_data.counts, (matrix_data.i, matrix_data.j)),
                        shape=(matrix_data.i.max() + 1, matrix_data.j.max() + 1)).tocsr()
    num_users = counts.shape[0] if num_users is None else num_users
    num_items = counts.shape[1] if num_items is None else num_items
    counts = counts[:num_users, :num_items]
    print("nonzeros: {}, total: {}".format(np.prod(counts.shape) - counts.size, counts.sum()))
    alpha = (np.prod(counts.shape) - counts.size) / counts.sum()
    #alpha = sp.sparse.linalg.norm(counts) * 100
    print('alpha %.5f' % alpha)
    counts *= alpha
    t1 = time.time()
    print('Finished loading matrix in %f seconds' % (t1 - t0))
    print('Maximum element is {}'.format(counts.max()))
    del matrix_data
    return counts


class LogisticMF():

    def __init__(self, counts, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=30):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.iterations = iterations
        self.reg_param = reg_param
        self.gamma = gamma

    def train_model_adagrad(self, x0=None):
        self.loss_history = []
        self.ones = np.ones((self.num_users, self.num_items))
        if x0 is None:
            self.user_vectors = np.random.normal(size=(self.num_users,
                                                       self.num_factors))
            self.item_vectors = np.random.normal(size=(self.num_items,
                                                       self.num_factors))
        else:
            self.user_vectors, self.item_vectors = x0
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        user_vec_deriv_sum = np.zeros((self.num_users, self.num_factors))
        item_vec_deriv_sum = np.zeros((self.num_items, self.num_factors))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))
        for i in range(self.iterations):
            t0 = time.time()
            # Fix items and solve for users
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            user_vec_deriv, user_bias_deriv = self.deriv(True)
            user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)
            vec_step_size = 1e-10#self.gamma / np.sqrt(user_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv

            # Fix users and solve for items
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            item_vec_deriv, item_bias_deriv = self.deriv(False)
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            vec_step_size = 1e-10#self.gamma / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv
            t1 = time.time()
            self.loss_history.append(self.log_likelihood())
            print('log_likelihood: {}'.format(self.loss_history[-1]))

            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)


    def train_model(self, x0=None):
        self.loss_history = []
        self.ones = np.ones((self.num_users, self.num_items))
        if x0 is None:
            self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
            self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))
        else:
            self.user_vectors, self.item_vectors = x0
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        user_vec_deriv_sum = np.zeros((self.num_users, self.num_factors))
        item_vec_deriv_sum = np.zeros((self.num_items, self.num_factors))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))
        for i in range(self.iterations):
            t0 = time.time()
            # Fix items and solve for users
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            user_vec_deriv, user_bias_deriv = self.deriv(True)
            user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv

            # Fix users and solve for items
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            item_vec_deriv, item_bias_deriv = self.deriv(False)
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv
            t1 = time.time()
            self.loss_history.append(self.log_likelihood())
            print('log_likelihood: {}'.format(self.loss_history[-1]))

            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)


    def deriv(self, user):
        if user:
            vec_deriv = np.dot(self.counts, self.item_vectors)
            bias_deriv = np.expand_dims(np.sum(self.counts, axis=1), 1)

        else:
            vec_deriv = np.dot(self.counts.T, self.user_vectors)
            bias_deriv = np.expand_dims(np.sum(self.counts, axis=0), 1)
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        A = np.exp(A)
        A /= (A + self.ones)
        A = (self.counts + self.ones) * A

        if user:
            vec_deriv -= np.dot(A, self.item_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.user_vectors
        else:
            vec_deriv -= np.dot(A.T, self.user_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.item_vectors
        return (vec_deriv, bias_deriv)

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.counts
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        A = (self.counts + self.ones) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.item_vectors))
        return loglik

    def print_vectors(self):
        user_vecs_file = open('logmf-user-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_users):
            vec = ' '.join(map(str, self.user_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            user_vecs_file.write(line)
        user_vecs_file.close()
        item_vecs_file = open('logmf-item-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_items):
            vec = ' '.join(map(str, self.item_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            item_vecs_file.write(line)
        item_vecs_file.close()


class FixedRiemannianLogisticMF():

    def __init__(self, counts, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=30):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.manifold = FixedRankEmbeeded(counts.shape[0], counts.shape[1], num_factors)
        self.num_factors = num_factors + 1
        self.iterations = iterations
        self.reg_param = reg_param
        self.gamma = gamma

    def train_model(self, x0=None):
        self.loss_history = []
        self.ones = np.ones((self.num_users, self.num_items))
        if x0 is None:
            self.user_vectors, self.middle, self.item_vectors = self.manifold.rand()
            self.item_vectors = self.item_vectors.T
        else:
            self.user_vectors, self.middle, self.item_vectors = x0
            self.item_vectors = self.item_vectors.T
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        x_deriv_sum = (np.zeros((self.num_users, self.num_factors)),
                       np.zeros((self.num_items, self.num_factors)))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))

        searcher = LineSearchAdaptive()
        for i in range(self.iterations):
            """
            x = (self.user_vectors, self.middle, self.item_vectors.T)
            t0 = time.time()

            x_deriv, user_bias_deriv, item_bias_deriv = self.all_der()
            x_grad = self.manifold.from_partial(x, x_deriv)
            gradnorm = self.manifold.norm(x, x_grad)

            desc_dir = self.manifold.lincomb(x, -1.0, x_grad)

            # Perform line-search
            step_size, x = searcher.search(self.log_likelihood_obj, self.manifold, x, desc_dir,
                                                 self.log_likelihood(), gradnorm**2)
            self.user_vectors, self.middle, self.item_vectors = x
            self.item_vectors = self.item_vectors.T
            """
            x = (self.user_vectors, self.middle, self.item_vectors.T)
            t0 = time.time()
            # Solve for users and items
            x_deriv, user_bias_deriv, item_bias_deriv = self.all_der()
            #x_egrad_square = self.manifold.egrad2rgrad(x, np.square(x_deriv))
            x_egrad = self.manifold.from_partial(x, x_deriv)
            #x_egrad_square = np.square(x_egrad[0].dot(x_egrad[1]))

            #x_deriv_sum = self.manifold.lincomb(x, 1.0, x_deriv_sum, 1.0, x_egrad_square)
            x_step_size = 1e-10#self.gamma / np.sqrt(self.manifold.norm(x, x_deriv_sum))
            #user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)
            #user_vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
            user_bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)

            #item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            #item_vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
            item_bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)

            #user_proj_deriv, item_proj_deriv = self.manifold.egrad2rgrad(X, (user_vec_deriv, item_vec_deriv))

            self.user_vectors, self.middle, self.item_vectors = self.manifold.retr(x, self.manifold.lincomb(x, x_step_size, x_egrad))
            self.item_vectors = self.item_vectors.T
            self.user_biases += user_bias_step_size * user_bias_deriv
            self.item_biases += item_bias_step_size * item_bias_deriv
            t1 = time.time()
            self.loss_history.append(self.log_likelihood())
            print('log_likelihood: {}'.format(self.loss_history[-1]))

            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)

    def all_der(self):
        du = np.dot(self.counts, self.item_vectors.dot(self.middle.T))
        dbu = np.expand_dims(np.sum(self.counts, axis=1), 1)

        dv = np.dot(self.counts.T, self.user_vectors.dot(self.middle))
        dbv = np.expand_dims(np.sum(self.counts, axis=0), 1)

        ds = np.dot(self.item_vectors.T, self.counts.T.dot(self.user_vectors))

        A = np.dot(self.user_vectors.dot(self.middle), self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        #A = 1. / (1. + np.exp(-A))
        A = np.exp(A)
        A /= (A + 1)
        A = (self.counts + 1) * A

        du -= np.dot(A, self.item_vectors.dot(self.middle.T))
        dbu -= np.expand_dims(np.sum(A, axis=1), 1)

        dv -= np.dot(A.T, self.user_vectors.dot(self.middle))
        dbv -= np.expand_dims(np.sum(A, axis=0), 1)

        ds -= np.dot(self.item_vectors.T, A.T.dot(self.user_vectors))
        ds -= self.middle
        return (du, ds.T, dv.T), dbu, dbv

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.counts
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        A = (self.counts + self.ones) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(np.diag(self.middle)))
        return loglik

    def log_likelihood_obj(self, x):
        user_vectors, middle, item_vectors = x
        item_vectors = item_vectors.T
        loglik = 0
        A = np.dot(user_vectors, item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.counts
        loglik += np.sum(B)

        A = np.exp(A)
        A += self.ones

        A = np.log(A)
        A = (self.counts + self.ones) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(np.diag(middle)))
        return loglik

    def print_vectors(self):
        user_vecs_file = open('logmf-user-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_users):
            vec = ' '.join(map(str, self.user_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            user_vecs_file.write(line)
        user_vecs_file.close()
        item_vecs_file = open('logmf-item-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_items):
            vec = ' '.join(map(str, self.item_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            item_vecs_file.write(line)
        item_vecs_file.close()


class RiemannianLogisticMF():
    def __init__(self, counts, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=30):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.manifold = FixedRankEmbeeded2Factors(counts.shape[0], counts.shape[1], num_factors)
        self.num_factors = num_factors
        self.iterations = iterations
        self.reg_param = reg_param
        self.gamma = gamma

    def train_model(self, x0=None):
        self.loss_history = []
        self.ones = np.ones((self.num_users, self.num_items))
        if x0 is None:
            self.user_vectors, self.item_vectors = self.manifold.rand()
        else:
            self.user_vectors, self.item_vectors = x0
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        x_deriv_sum = (np.zeros((self.num_users, self.num_factors)),
                       np.zeros((self.num_items, self.num_factors)))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))
        x_ = (self.user_vectors, self.item_vectors)

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        momentum = (np.zeros((self.num_users, self.num_factors)),
            np.zeros((self.num_items, self.num_factors)))

        velocity = (np.zeros((self.num_users, self.num_factors)),
            np.zeros((self.num_items, self.num_factors)))

        for i in range(self.iterations):
            x = (self.user_vectors, self.item_vectors)

            x_deriv_sum = self.manifold.transp(x_, x, x_deriv_sum)

            t0 = time.time()
            # Solve for users and items
            x_deriv, (user_bias_deriv, item_bias_deriv) = self.deriv()
            x_egrad = self.manifold.egrad2rgrad(x, x_deriv)

            x_deriv_squared = tuple(dx**2 for dx in x_deriv)

            x_egrad_squared = self.manifold.egrad2rgrad(x, x_deriv_squared)

            x_deriv_sum = self.manifold.lincomb(x, 1.0, x_deriv_sum, 1.0, x_egrad_squared)

            ambient_deriv_sum = self.manifold.tangent2ambient(x, x_deriv_sum)
            average_deriv_sum = 1.0 * sum_lowrank(ambient_deriv_sum) / (np.prod(self.counts.shape))

            x_step_size = self.gamma / (np.sqrt(average_deriv_sum))

            user_bias_deriv_sum += np.square(user_bias_deriv)
            user_bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)

            item_bias_deriv_sum += np.square(item_bias_deriv)
            item_bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)

            modified_egrad = self.manifold.lincomb(x, x_step_size, x_egrad)

            self.user_vectors, self.item_vectors = self.manifold.retr(x, modified_egrad)
            self.user_biases += user_bias_step_size * user_bias_deriv
            self.item_biases += item_bias_step_size * item_bias_deriv
            t1 = time.time()
            self.loss_history.append(self.log_likelihood())
            print('log_likelihood: {}'.format(self.loss_history[-1]))

            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)
            """
            x = (self.user_vectors, self.item_vectors)
            t0 = time.time()
            # Solve for users and items
            x_deriv, (user_bias_deriv, item_bias_deriv) = self.deriv()
            x_egrad_square = self.manifold.egrad2rgrad(x, (np.square(x_deriv[0]), np.square(x_deriv[1])))
            x_egrad = self.manifold.egrad2rgrad(x, x_deriv)


            x_deriv_sum = self.manifold.lincomb(x, 1.0, x_deriv_sum, 1.0, x_egrad_square)
            x_step_size = 1e-10#self.gamma / (np.sqrt(self.manifold.norm(x, x_deriv_sum)) + 1e-8)
            #user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)
            #user_vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
            user_bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)

            #item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            #item_vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
            item_bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)

            #user_proj_deriv, item_proj_deriv = self.manifold.egrad2rgrad(X, (user_vec_deriv, item_vec_deriv))

            self.user_vectors, self.item_vectors = self.manifold.retr(x, self.manifold.lincomb(x, x_step_size, x_egrad))
            self.user_biases += user_bias_step_size * user_bias_deriv
            self.item_biases += item_bias_step_size * item_bias_deriv
            t1 = time.time()
            self.loss_history.append(self.log_likelihood())
            print('log_likelihood: {}'.format(self.loss_history[-1]))

            print 'iteration %i finished in %f seconds' % (i + 1, t1 - t0)
            """


    def deriv(self):
        user_vec_deriv = np.dot(self.counts, self.item_vectors)
        user_bias_deriv = np.expand_dims(np.sum(self.counts, axis=1), 1)
        item_vec_deriv = np.dot(self.counts.T, self.user_vectors)
        item_bias_deriv = np.expand_dims(np.sum(self.counts, axis=0), 1)
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        A = self.ones / (self.ones + np.exp(-A))
        #A = np.exp(A)
        #A /= (A + self.ones)
        A = (self.counts + self.ones) * A

        user_vec_deriv -= np.dot(A, self.item_vectors)
        user_bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)

        item_vec_deriv -= np.dot(A.T, self.user_vectors)
        item_bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
        # L2 regularization
        user_vec_deriv -= self.reg_param * self.user_vectors
        item_vec_deriv -= self.reg_param * self.item_vectors
        return (user_vec_deriv, item_vec_deriv), (user_bias_deriv, item_bias_deriv)




    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.counts
        loglik += np.sum(B)

        #A = np.exp(A)
        #A += self.ones

        #A = np.log(A)
        A = accurate_log_exp1(A)
        A = (self.counts + self.ones) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.item_vectors))
        return loglik

    def print_vectors(self):
        user_vecs_file = open('logmf-user-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_users):
            vec = ' '.join(map(str, self.user_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            user_vecs_file.write(line)
        user_vecs_file.close()
        item_vecs_file = open('logmf-item-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_items):
            vec = ' '.join(map(str, self.item_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            item_vecs_file.write(line)
        item_vecs_file.close()


class WildLogisticMF():
    def __init__(self, counts, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=30, minstepsize=1e-9):
        self.counts = counts
        self.num_users = self.counts.shape[0]
        self.num_items = self.counts.shape[1]
        self.num_factors = num_factors
        self.iterations = iterations
        self.minstepsize = minstepsize
        self.reg_param = reg_param
        self.gamma = gamma
        self._bootstrap_problem()

    def _bootstrap_problem(self):
        self.manifold = FixedRankEmbeeded2Factors(self.num_users, self.num_items, self.num_factors + 1)
        self.solver = ConjugateGradient(maxiter=self.iterations, minstepsize=self.minstepsize)

    def train_model(self, x0=None):
        self.L = T.matrix('L')
        self.R = T.matrix('R')
        problem = Problem(man=self.manifold,
                          theano_cost=self.log_likelihood(),
                          theano_arg=[self.L, self.R])

        if x0 is None:
            user_vectors = np.random.normal(size=(self.num_users,
                                                       self.num_factors))
            item_vectors = np.random.normal(size=(self.num_items,
                                                       self.num_factors))
            user_biases = np.random.normal(size=(self.num_users, 1)) / SCONST
            item_biases = np.random.normal(size=(self.num_items, 1)) / SCONST
            x0 = (np.hstack((user_vectors, user_biases)),
                  np.hstack((item_vectors, item_biases)))
        else:
            x0 = x0
        (left, right), self.loss_history = self.solver.solve(problem, x=x0)

        self.user_vectors = left[:, :-1]
        self.item_vectors = right[:, :-1]
        self.user_biases = left[:, -1]
        self.item_biases = right[:, -1]
        print('U norm: {}'.format(la.norm(self.user_vectors)))
        print('V norm: {}'.format(la.norm(self.item_vectors)))



    def evaluate_lowrank(self, U, V, item, fast=False):
        if hasattr(item, '__len__') and len(item) == 2 and len(item[0]) == len(item[1]):
            if fast:
                rows = U[item[0], :]
                cols = V[item[1], :]
                data = (rows * cols).sum(1)
                return data
            else:
                idx_argsort = item[0].argsort()
                item = (item[0][idx_argsort], item[1][idx_argsort])

                vals, idxs, counts = [theano.shared(it) for it in\
                                      np.unique(item[0], return_index=True, return_counts=True)]
                output = T.zeros(int(np.max(counts.get_value())))
                it1 = theano.shared(item[1])

                def process_partial_dot(row_idx, out, U, V, item):
                    partial_dot = T.dot(U[vals[row_idx], :], V[item[idxs[row_idx]: idxs[row_idx] + counts[row_idx]], :].T)
                    return T.set_subtensor(out[:counts[row_idx]], partial_dot)
                parts, updates = theano.scan(fn=process_partial_dot,
                                             outputs_info=output,
                                             sequences=T.arange(vals.size),
                                             non_sequences=[U, V, it1])
                mask = np.ones((vals.get_value().size, int(np.max(counts.get_value()))))
                for i, count in enumerate(counts.get_value()):
                    mask[i, count:] = 0
                return parts[theano.shared(mask).nonzero()].ravel()
        else:
            raise ValueError('__getitem__ now supports only indices set')

    def ll_func(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.counts
        loglik += np.sum(B)

        #A = np.exp(A)
        #A += self.ones

        #A = np.log(A)
        A = accurate_log_exp1(A)
        A = (self.counts + 1) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.item_vectors))
        return loglik

    def log_likelihood(self):
        Users = self.L[:, :-1]
        Items = self.R[:, :-1]
        UserBiases = self.L[:, -1].reshape((-1, 1))
        ItemBiases = self.R[:, -1].reshape((-1, 1))
        #UserOuter = self.L[:, -2]
        #ItemOuter = self.R[:, -1]

        ## A = T.dot(Users, Items.T)
        ## A += UserBiases
        ## A += ItemBiases.T
        ## B = A * self.counts
        ## loglik = T.sum(B)
        A = T.dot(self.L[:, :-1], (self.R[:, :-1]).T)
        A = T.inc_subtensor(A[:, :], UserBiases)
        A = T.inc_subtensor(A[:, :], ItemBiases.T)
        B = A * self.counts
        loglik = T.sum(B)

        # A implicitly stored as self.L @ self.R.T
        # loglik = T.sum(A * self.counts) => sum over nonzeros only
        #print('nnz size: {}'.format(self.counts.nonzero()[0].size))
        #loglik = T.dot(self.evaluate_lowrank(self.L, self.R, self.counts.nonzero(), fast=False),
        #          np.array(self.counts[self.counts.nonzero()]).ravel())

        A = T.exp(A)
        A += 1
        A = T.log(A)
        # There we use Taylor series ln(exp(x) + 1) = ln(2) + x/2 + x^2/8 + O(x^4) at x=0
        # ln(2)
        #const_term = (T.ones((self.num_users, 1)) * np.log(2), T.ones((self.num_items, 1)))
        # x/2
        #first_order_term = (0.5 * self.L, 0.5 * self.R)
        # x^2/8
        #second_order_term = hadamard((self.L, self.R), (self.L, self.R), self.num_factors)
        #second_order_term = tuple(factor / 8.0 for factor in second_order_term)

        #grouped_factors = list(zip(const_term, first_order_term, second_order_term))
        #A = (T.concatenate(grouped_factors[0], axis=1), T.concatenate(grouped_factors[1], axis=1))

        A = (self.counts + 1) * A
        loglik -= T.sum(A)
        #loglik -= sum_lowrank(A)
        #loglik -= T.dot(self.evaluate_lowrank(A[0], A[1], self.counts.nonzero(), fast=False),
        #          np.array(self.counts[self.counts.nonzero()]).ravel())


        # L2 regularization
        loglik -= 0.5 * self.reg_param * T.sum(T.square(self.L[:, :-1]))
        loglik -= 0.5 * self.reg_param * T.sum(T.square(self.R[:, :-1]))

        # we need strictly maintain UserOuter and ItemOuter be ones, just to ensure they properly
        # outer products with biases
        #C = 10
        #loglik -= C * T.sum(T.abs_(self.L[:, -2] - SCONST))
        #loglik -= C * T.sum(T.abs_(self.R[:, -1] - SCONST))

        # Return negation of LogLikelihood cause we will minimize cost
        return -loglik

    def print_vectors(self):
        user_vecs_file = open('logmf-user-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_users):
            vec = ' '.join(map(str, self.user_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            user_vecs_file.write(line)
        user_vecs_file.close()
        item_vecs_file = open('logmf-item-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_items):
            vec = ' '.join(map(str, self.item_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            item_vecs_file.write(line)
        item_vecs_file.close()

class UsvRiemannianLogisticMF():
    def __init__(self, counts, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=30, minstepsize=1e-9):
        self.counts = counts
        self.num_users = self.counts.shape[0]
        self.num_items = self.counts.shape[1]
        self.num_factors = num_factors + 1
        self.iterations = iterations
        self.minstepsize = minstepsize
        self.reg_param = reg_param
        self.gamma = gamma
        self._bootstrap_problem()

    def _bootstrap_problem(self):
        self.manifold = FixedRankEmbeeded(self.num_users, self.num_items, self.num_factors)
        self.solver = ConjugateGradient(maxiter=self.iterations, minstepsize=self.minstepsize)

    def train_model(self, x0=None):
        self.U = T.matrix('U')
        self.S = T.matrix('S')
        self.V = T.matrix('V')
        problem = Problem(man=self.manifold,
                          theano_cost=self.log_likelihood(),
                          theano_arg=[self.U, self.S, self.V.T])

        if x0 is None:
            user_vectors = la.qr(np.random.normal(size=(self.num_users,
                                                       self.num_factors)))[0]
            item_vectors = la.qr(np.random.normal(size=(self.num_items,
                                                       self.num_factors)))[0]
            s = rnd.random(self.num_factors)
            s = np.sort(s)[::-1]
            middle = np.diag(s)
            x0 = (np.hstack((user_vectors, user_biases)),
                  np.hstack((item_vectors, item_biases)))
        else:
            x0 = x0
        (left, right), self.loss_history = self.solver.solve(problem, x=x0)

        self.user_vectors = left[:, :-1]
        self.item_vectors = right[:, :-1]
        self.user_biases = left[:, -1]
        self.item_biases = right[:, -1]
        print('U norm: {}'.format(la.norm(self.user_vectors)))
        print('V norm: {}'.format(la.norm(self.item_vectors)))
        #print("how much user outer? {}".format(np.average(np.isclose(left[:, -2], 1))))
        #print("how much item outer? {}".format(np.average(np.isclose(right[:, -1], 1))))
        #print('user delta: {} in norm, {} in max abs'.format(la.norm(left[:, -2] - 1), np.max(np.abs(left[:, -2] - 1))))
        #print('item delta: {} in norm, {} in max abs'.format(la.norm(right[:, -1] - 1), np.max(np.abs(right[:, -1] - 1))))
        #C = 1000
        #smm = C * np.sum(np.square(left[:, -2] - SCONST))
        #smm += C * np.sum(np.square(right[:, -1] - SCONST))
        #print('smm: {}'.format(smm))
        #print('ll func: {}'.format(self.ll_func()))



    def evaluate_lowrank(self, U, V, item, fast=False):
        if hasattr(item, '__len__') and len(item) == 2 and len(item[0]) == len(item[1]):
            if fast:
                rows = U[item[0], :]
                cols = V[item[1], :]
                data = (rows * cols).sum(1)
                return data
            else:
                idx_argsort = item[0].argsort()
                item = (item[0][idx_argsort], item[1][idx_argsort])

                vals, idxs, counts = [theano.shared(it) for it in\
                                      np.unique(item[0], return_index=True, return_counts=True)]
                output = T.zeros(int(np.max(counts.get_value())))
                it1 = theano.shared(item[1])

                def process_partial_dot(row_idx, out, U, V, item):
                    partial_dot = T.dot(U[vals[row_idx], :], V[item[idxs[row_idx]: idxs[row_idx] + counts[row_idx]], :].T)
                    return T.set_subtensor(out[:counts[row_idx]], partial_dot)
                parts, updates = theano.scan(fn=process_partial_dot,
                                             outputs_info=output,
                                             sequences=T.arange(vals.size),
                                             non_sequences=[U, V, it1])
                mask = np.ones((vals.get_value().size, int(np.max(counts.get_value()))))
                for i, count in enumerate(counts.get_value()):
                    mask[i, count:] = 0
                return parts[theano.shared(mask).nonzero()].ravel()
        else:
            raise ValueError('__getitem__ now supports only indices set')

    def ll_func(self):
        loglik = 0
        A = np.dot(self.user_vectors, self.item_vectors.T)
        A += self.user_biases
        A += self.item_biases.T
        B = A * self.counts
        loglik += np.sum(B)

        #A = np.exp(A)
        #A += self.ones

        #A = np.log(A)
        A = accurate_log_exp1(A)
        A = (self.counts + 1) * A
        loglik -= np.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * np.sum(np.square(self.item_vectors))
        return loglik

    def log_likelihood(self):
        Users = self.L[:, :-1]
        Items = self.R[:, :-1]
        UserBiases = self.L[:, -1].reshape((-1, 1))
        ItemBiases = self.R[:, -1].reshape((-1, 1))
        #UserOuter = self.L[:, -2]
        #ItemOuter = self.R[:, -1]

        ## A = T.dot(Users, Items.T)
        ## A += UserBiases
        ## A += ItemBiases.T
        ## B = A * self.counts
        ## loglik = T.sum(B)
        A = T.dot(self.L[:, :-1], (self.R[:, :-1]).T)
        A = T.inc_subtensor(A[:, :], UserBiases)
        A = T.inc_subtensor(A[:, :], ItemBiases.T)
        B = A * self.counts
        loglik = T.sum(B)

        # A implicitly stored as self.L @ self.R.T
        # loglik = T.sum(A * self.counts) => sum over nonzeros only
        #print('nnz size: {}'.format(self.counts.nonzero()[0].size))
        #loglik = T.dot(self.evaluate_lowrank(self.L, self.R, self.counts.nonzero(), fast=False),
        #          np.array(self.counts[self.counts.nonzero()]).ravel())

        A = T.exp(A)
        A += 1
        A = T.log(A)
        # There we use Taylor series ln(exp(x) + 1) = ln(2) + x/2 + x^2/8 + O(x^4) at x=0
        # ln(2)
        #const_term = (T.ones((self.num_users, 1)) * np.log(2), T.ones((self.num_items, 1)))
        # x/2
        #first_order_term = (0.5 * self.L, 0.5 * self.R)
        # x^2/8
        #second_order_term = hadamard((self.L, self.R), (self.L, self.R), self.num_factors)
        #second_order_term = tuple(factor / 8.0 for factor in second_order_term)

        #grouped_factors = list(zip(const_term, first_order_term, second_order_term))
        #A = (T.concatenate(grouped_factors[0], axis=1), T.concatenate(grouped_factors[1], axis=1))

        A = (self.counts + 1) * A
        loglik -= T.sum(A)
        #loglik -= sum_lowrank(A)
        #loglik -= T.dot(self.evaluate_lowrank(A[0], A[1], self.counts.nonzero(), fast=False),
        #          np.array(self.counts[self.counts.nonzero()]).ravel())


        # L2 regularization
        loglik -= 0.5 * self.reg_param * T.sum(T.square(self.L[:, :-1]))
        loglik -= 0.5 * self.reg_param * T.sum(T.square(self.R[:, :-1]))

        # we need strictly maintain UserOuter and ItemOuter be ones, just to ensure they properly
        # outer products with biases
        #C = 10
        #loglik -= C * T.sum(T.abs_(self.L[:, -2] - SCONST))
        #loglik -= C * T.sum(T.abs_(self.R[:, -1] - SCONST))

        # Return negation of LogLikelihood cause we will minimize cost
        return -loglik

    def print_vectors(self):
        user_vecs_file = open('logmf-user-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_users):
            vec = ' '.join(map(str, self.user_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            user_vecs_file.write(line)
        user_vecs_file.close()
        item_vecs_file = open('logmf-item-vecs-%i' % self.num_factors, 'w')
        for i in range(self.num_items):
            vec = ' '.join(map(str, self.item_vectors[i]))
            line = '%i\t%s\n' % (i, vec)
            item_vecs_file.write(line)
        item_vecs_file.close()


if __name__ == "__main__":
    import os
    folder_path = "lastfm-dataset-360K"
    mat_path = os.path.join(folder_path, 'cut.tsv')

    M, N = 1000, 1000

    mat = load_matrix_sparse(mat_path, M, N)
    mat = np.array(mat[:M, :N].todense())
    print("{} users, {} items".format(*mat.shape))
    print("number of nonzero entries: {}".format(mat.size))

    num_factors = 20

    n_iters = 200
    wmf = WildLogisticMF(mat, num_factors, reg_param=0.6, gamma=1.0, iterations=n_iters)
    lmf = LogisticMF(mat, num_factors, reg_param=0.6, gamma=1.0, iterations=n_iters)
    rlmf = RiemannianLogisticMF(mat, num_factors, reg_param=0.6, gamma=1.0, iterations=n_iters)
    frlmf = FixedRiemannianLogisticMF(mat, num_factors, reg_param=0.6*1e-2, gamma=1.0, iterations=n_iters)

    left, right = np.random.randn(M, num_factors), np.random.randn(N, num_factors)
    x0 = left.dot(right.T)
    u, s, v = la.svd(x0, full_matrices=False)
    u = u[:, :num_factors]
    s = np.diag(s[:num_factors])
    v = v[:num_factors, :]
    user_ones = np.ones((M, 1)) * SCONST
    item_ones = np.ones((N, 1)) * SCONST
    user_b = rnd.randn(M, 1) / SCONST
    item_b = rnd.randn(N, 1) / SCONST

    print("train 2 factor Wild Logistic MF")
    wmf.train_model(x0=(np.hstack((left, user_b)),
                        np.hstack((right, item_b))))
    print("end of training.")

    print("train 2 factor Riemannian Logistic MF:")
    rlmf.train_model(x0=(left, right))
    print("end of training.")



    #print("train 3 factor Riemannian Logistic MF:")
    #frlmf.train_model(x0=(u, s, v))
    #print("end of training.")

    print("train Logistic MF:")
    lmf.train_model(x0=(left, right))
    print("end of training.")


    begin = 50
    llmf, llrmf, llfrmf = lmf.loss_history[begin:], rlmf.loss_history[begin:], wmf.loss_history[begin:]
    plt.plot(np.arange(len(llmf)), llmf, 'r')
    #plt.plot(np.arange(len(llrmf)), llrmf, 'g')
    plt.plot(np.arange(len(llfrmf)), -np.array(llfrmf), 'b')
    plt.legend(['Alternating', 'CG'], loc=2)
    plt.show()