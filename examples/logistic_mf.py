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
from pymanopt.manifolds import FixedRankEmbeeded2Factors
from pymanopt.solvers import TrustRegions, ConjugateGradient, SteepestDescent


def load_matrix_sparse(filename='cut.tsv', num_users=None, num_items=None):
    t0 = time.time()
    matrix_data = pd.read_csv(filename, sep='\t', header=None, names=['i', 'j', 'counts'], dtype={'counts': np.float})
    counts = coo_matrix((matrix_data.counts, (matrix_data.i, matrix_data.j)),
                        shape=(matrix_data.i.max() + 1, matrix_data.j.max() + 1)).tocsr()
    num_users = counts.shape[0] if num_users is None else num_users
    num_items = counts.shape[1] if num_items is None else num_items
    counts = counts[:num_users, :num_items]
    alpha = sp.sparse.linalg.norm(counts) * 100
    print('alpha %.5f' % alpha)
    counts /= alpha
    t1 = time.time()
    print('Finished loading matrix in %f seconds' % (t1 - t0))
    print('Maximum element is {}'.format(counts.max()))
    del matrix_data
    return counts


def hadamard(left, right, r):
    u1, v1 = left
    u2, v2 = right
    ind1, ind2 = np.repeat(np.arange(r), r), np.tile(np.arange(r), r)
    u = u1[:, ind1] * u2[:, ind2]
    v = v1[:, ind1] * v2[:, ind2]
    return u, v


def sum_lowrank(lowrank_matrix):
    u, v = lowrank_matrix
    return u.sum(0).dot(v.sum(0))


class LogisticMF():
    def __init__(self, counts, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=30, minstepsize=1e-9):
        self.counts = counts
        N = 20000
        self.counts = counts[:N, :N]
        self.num_users = self.counts.shape[0]
        self.num_items = self.counts.shape[1]
        self.num_factors = num_factors + 2
        self.iterations = iterations
        self.minstepsize = minstepsize
        self.reg_param = reg_param
        self.gamma = gamma
        self._bootstrap_problem()

    def _bootstrap_problem(self):
        self.manifold = FixedRankEmbeeded2Factors(self.num_users, self.num_items, self.num_factors)
        self.solver = ConjugateGradient(maxiter=self.iterations, minstepsize=self.minstepsize)

    def train_model(self):
        self.L = T.matrix('L')
        self.R = T.matrix('R')
        problem = Problem(man=self.manifold,
                          theano_cost=self.log_likelihood(),
                          theano_arg=[self.L, self.R])
        left, right = self.solver.solve(problem)
        self.user_vectors = left[:, :-2]
        self.item_vectors = right[:, :-2]
        self.user_biases = left[:, -1]
        self.item_biases = right[:, -2]
        print('U norm: {}'.format(la.norm(self.user_vectors)))
        print('V norm: {}'.format(la.norm(self.item_vectors)))
        print("how much user outer? {}".format(np.average(np.isclose(left[:, -2], 1))))
        print("how much item outer? {}".format(np.average(np.isclose(right[:, -1], 1))))
        print('user delta: {} in norm, {} in max abs'.format(la.norm(left[:, -2] - 1), np.max(np.abs(left[:, -2] - 1))))
        print('item delta: {} in norm, {} in max abs'.format(la.norm(right[:, -1] - 1), np.max(np.abs(right[:, -1] - 1))))

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

    def log_likelihood(self):
        Users = self.L[:, :-2]
        Items = self.R[:, :-2]
        UserBiases = self.L[:, -1]
        ItemBiases = self.R[:, -2]
        UserOuter = self.L[:, -2]
        ItemOuter = self.R[:, -1]

        ## A = T.dot(Users, Items.T)
        ## A += UserBiases
        ## A += ItemBiases.T
        ## B = A * self.counts
        ## loglik = T.sum(B)

        # A implicitly stored as self.L @ self.R.T
        # loglik = T.sum(A * self.counts) => sum over nonzeros only
        print('nnz size: {}'.format(self.counts.nonzero()[0].size))
        loglik = T.dot(self.evaluate_lowrank(self.L, self.R, self.counts.nonzero(), fast=False),
                  np.array(self.counts[self.counts.nonzero()]).ravel())

        ## A = T.exp(A)
        ## A += 1
        ## A = T.log(A)
        # There we use Taylor series ln(exp(x) + 1) = ln(2) + x/2 + x^2/8 + O(x^4) at x=0
        # ln(2)
        const_term = (T.ones((self.num_users, 1)) * np.log(2), T.ones((self.num_items, 1)))
        # x/2
        first_order_term = (0.5 * self.L, 0.5 * self.R)
        # x^2/8
        second_order_term = hadamard((self.L, self.R), (self.L, self.R), self.num_factors)
        second_order_term = tuple(factor / 8.0 for factor in second_order_term)

        grouped_factors = list(zip(const_term, first_order_term, second_order_term))
        A = (T.concatenate(grouped_factors[0], axis=1), T.concatenate(grouped_factors[1], axis=1))

        ## A = (self.counts + 1) * A
        ## loglik -= T.sum(A)
        loglik -= sum_lowrank(A)
        loglik -= T.dot(self.evaluate_lowrank(A[0], A[1], self.counts.nonzero(), fast=False),
                  np.array(self.counts[self.counts.nonzero()]).ravel())


        # L2 regularization
        loglik -= 0.5 * self.reg_param * T.sum(T.square(Users))
        loglik -= 0.5 * self.reg_param * T.sum(T.square(Items))

        # we need strictly maintain UserOuter and ItemOuter be ones, just to ensure they properly
        # outer products with biases
        loglik -= self.num_users * T.sum(T.square(UserOuter - 1))
        loglik -= self.num_items * T.sum(T.square(ItemOuter - 1))

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


def reformat_users_items(filename, new_filename=None):
    TOTAL_LINES = 17559530
    DISCRETIZATION=10
    from collections import defaultdict
    if new_filename is None:
        new_filename = 'CUT-' + filename
    t0 = time.time()
    users = defaultdict(lambda: len(users))
    items = defaultdict(lambda: len(items))
    with open(new_filename, 'w') as destination:
        for i, line in enumerate(open(filename, 'r')):

            if (i + 1) % (TOTAL_LINES / DISCRETIZATION) == 0:
                print(i + 1, TOTAL_LINES / DISCRETIZATION)
                print('pass {}% of all lines'.format((i + 1) * 100 / TOTAL_LINES))
            user_sha, artist_sha, artist_title, count = line.split('\t')
            destination.write('\t'.join([str(users[user_sha]), str(items[artist_sha]), count]) + '\n')
    t1 = time.time()
    print('time spent {} s'.format(t1 - t0))
    print('overall users: {}'.format(len(users)))
    print('overall items: {}'.format(len(items)))


def reformat_big_matrix():
    folder_path = "lastfm-dataset-360K"
    import os
    filename = os.path.join(folder_path, 'usersha1-artmbid-artname-plays.tsv')
    reformat_users_items(filename, os.path.join(folder_path, 'cut.tsv'))

if __name__ == "__main__":
    import os
    folder_path = "lastfm-dataset-360K"
    mat_path = os.path.join(folder_path, 'cut.tsv')

    mat = load_matrix_sparse(mat_path)
    print("{} users, {} items".format(*mat.shape))
    print("number of nonzero entries: {}".format(mat.size))

    logistic_mf = LogisticMF(mat, num_factors=3, reg_param=1.0, gamma=1.0, iterations=50, minstepsize=1e-9)
    logistic_mf.train_model()