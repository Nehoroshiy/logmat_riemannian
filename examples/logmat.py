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

    def train_model(self):
        self.loss_history = []
        self.ones = np.ones((self.num_users, self.num_items))
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))
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
        self.num_factors = num_factors
        self.iterations = iterations
        self.reg_param = reg_param
        self.gamma = gamma

    def train_model(self):
        self.loss_history = []
        self.ones = np.ones((self.num_users, self.num_items))
        self.user_vectors, self.middle, self.item_vectors = self.manifold.rand()
        self.item_vectors = self.item_vectors.T
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        x_deriv_sum = (np.zeros((self.num_users, self.num_factors)),
                       np.zeros((self.num_items, self.num_factors)))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))
        for i in range(self.iterations):
            x = (self.user_vectors, self.middle, self.item_vectors.T)
            t0 = time.time()
            # Solve for users and items
            x_deriv, user_bias_deriv, item_bias_deriv = self.all_der()
            #x_egrad_square = self.manifold.egrad2rgrad(x, np.square(x_deriv))
            x_egrad = self.manifold.egrad2rgrad(x, x_deriv)
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

    def train_model(self):
        self.loss_history = []
        self.ones = np.ones((self.num_users, self.num_items))
        self.user_vectors, self.item_vectors = self.manifold.rand()
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.item_biases = np.random.normal(size=(self.num_items, 1))

        x_deriv_sum = (np.zeros((self.num_users, self.num_factors)),
                       np.zeros((self.num_items, self.num_factors)))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        item_bias_deriv_sum = np.zeros((self.num_items, 1))
        for i in range(self.iterations):
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

    n_iters = 50
    lmf = LogisticMF(mat, num_factors, reg_param=0.6, gamma=1.0, iterations=n_iters)
    rlmf = RiemannianLogisticMF(mat, num_factors, reg_param=0.6, gamma=1.0, iterations=n_iters)
    frlmf = FixedRiemannianLogisticMF(mat, num_factors, reg_param=0.6, gamma=1.0, iterations=n_iters)

    print("train 2 factor Riemannian Logistic MF:")
    rlmf.train_model()
    print("end of training.")

    print("train 3 factor Riemannian Logistic MF:")
    frlmf.train_model()
    print("end of training.")

    print("train Logistic MF:")
    lmf.train_model()
    print("end of training.")


    llmf, llrmf, llfrmf = lmf.loss_history, rlmf.loss_history, frlmf.loss_history
    plt.plot(np.arange(len(llmf)), llmf, 'r')
    plt.plot(np.arange(len(llrmf)), llrmf, 'g')
    plt.plot(np.arange(len(llfrmf)), llfrmf, 'b')
    plt.legend(['Alternating', 'GH^T', 'USV^T'], loc=2)
    plt.show()