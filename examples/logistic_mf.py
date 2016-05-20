import time
import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import theano

from scipy.sparse import coo_matrix, random as sparse_rnd
from scipy.sparse.linalg import norm as sparse_norm

import theano.tensor as T
from pymanopt import Problem
from pymanopt.manifolds import FixedRankEmbeeded2Factors
from pymanopt.solvers import TrustRegions, ConjugateGradient, SteepestDescent


def load_matrix(filename, num_users, num_items):
    t0 = time.time()
    counts = np.zeros((num_users, num_items))
    total = 0.0
    num_zeros = num_users * num_items
    for i, line in enumerate(open(filename, 'r')):
        user, item, count = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = float(count)
        counts[user][item] = count
        total += count
        num_zeros -= 1
    alpha = num_zeros / total
    print 'alpha %.2f' % alpha
    counts *= alpha
    t1 = time.time()
    print 'Finished loading matrix in %f seconds' % (t1 - t0)
    return counts


def stupid_load(filename, lines=5000):
    with open("stupid-" + filename, 'w') as f:
        for i, line in enumerate(open(filename, 'r')):
            if i >= lines:
                break
            f.write(line + '\n')
    return None

def load_users_items(filename, num_users, num_items):
    t0 = time.time()
    counts = []
    for i, line in enumerate(open(filename, 'r')):
        counts.append(line.split('\t'))

        if user_sha is not user_map:
            if len(user_map) >= num_users:
                break
            user_map[user_sha] = user_count
            user_count += 1
            user_sha.append()
        user_info = counts.get(user_sha, {})
        user_info.append()
        counts[user_sha]


    counts = np.zeros((num_users, num_items))
    total = 0.0
    num_zeros = num_users * num_items
    for i, line in enumerate(open(filename, 'r')):
        user, item, count = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = float(count)
        counts[user][item] = count
        total += count
        num_zeros -= 1
    alpha = num_zeros / total
    print 'alpha %.2f' % alpha
    counts *= alpha
    t1 = time.time()
    print 'Finished loading matrix in %f seconds' % (t1 - t0)
    return counts


class LogisticMF:
    def __init__(self, counts, num_factors, reg_param=0.6, gamma=1.0,
                 iterations=30, minstepsize=1e-9):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
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
        print("how much user outer? {}".format(np.average(np.isclose(left[:, -2], 1))))
        print("how much item outer? {}".format(np.average(np.isclose(right[:, -1], 1))))

    def log_likelihood(self):
        Users = self.L[:, :-2]
        Items = self.R[:, :-2]
        UserBiases = self.L[:, -1]
        ItemBiases = self.R[:, -2]
        UserOuter = self.L[:, -2]
        ItemOuter = self.R[:, -1]

        A = T.dot(Users, Items.T)
        A += UserBiases
        A += ItemBiases.T
        B = A * self.counts
        loglik = T.sum(B)

        A = T.exp(A)
        A += 1

        A = T.log(A)
        A = (self.counts + 1) * A
        loglik -= T.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * T.sum(T.square(Users))
        loglik -= 0.5 * self.reg_param * T.sum(T.square(Items))

        # we need strictly maintain UserOuter and ItemOuter be ones, just to ensure they properly
        # outer products with biases
        loglik -= 100 * T.sum(T.abs(UserOuter - 1))
        loglik -= 100 * T.sum(T.abs(ItemOuter - 1))

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
    folder_path = "lastfm-dataset-360K"
