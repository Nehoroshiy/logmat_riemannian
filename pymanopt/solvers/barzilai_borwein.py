"""
Module containing steepest descent (gradient descent) algorithm based on
steepestdescent.m from the manopt MATLAB package.
"""
import time

from pymanopt.tools import theano_functions as tf
from pymanopt.solvers import linesearch
from pymanopt.solvers.solver import Solver
import numpy as np


class BarzilaiBorwein(Solver):
    def __init__(self, *args, **kwargs):
        super(BarzilaiBorwein, self).__init__(*args, **kwargs)

        # How to tune this parameters?
        self.gamma = 1e-4
        self.contraction_factor = 0.8
        self.eps = 1e-10
        self.alpha_max = 1e1
        self.m = 10

        self._searcher = linesearch.LineSearchAdaptive()

    # Function to solve optimisation problem using steepest descent.
    def solve(self, problem, x=None):
        """
        Perform optimization using gradient descent with linesearch. Both obj
        and arg must be theano TensorVariable objects. This method first
        computes the gradient (derivative) of obj w.r.t. arg, and then
        optimizes by moving in the direction of steepest descent (which is the
        opposite direction to the gradient).
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .man attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        man = problem.man
        # Compile the objective function and compute and compile its
        # gradient.
        if self._verbosity >= 1:
            print "Computing gradient and compiling..."
        problem.prepare(need_grad=True)

        objective = problem.cost
        gradient = problem.grad

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        if self._verbosity >= 1:
            print "Optimizing..."
        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        if self._verbosity >= 2:
            print " iter\t\t   cost val\t    grad. norm"

        while True:
            x_seq = [x]
            f_seq = [objective(x)]
            grad_seq = [gradient(x)]

            alphas = [(self.alpha_max - self.eps) * np.random.random() + self.eps]

            grad_norm = man.inner(x_seq[-1], grad_seq[-1], grad_seq[-1])

            if self._verbosity >= 2:
                print "%5d\t%+.16e\t%.8e" % (0, f_seq[-1], grad_norm)

            desc_dir = man.lincomb(x_seq[-1], -1.0, grad_seq[-1])
            step_size, x = self._searcher.search(objective, man, x, desc_dir, f_seq[-1], -grad_norm**2)
            x_seq.append(x)
            f_seq.append(objective(x))
            grad_seq.append(gradient(x))
            alphas.append(step_size / man.norm(x_seq[-1], grad_seq[-1]))

            if self._verbosity >= 2:
                print "%5d\t%+.16e\t%.8e" % (1, f_seq[-1], man.inner(x_seq[-1], grad_seq[-1], grad_seq[-1]))

            while True:
                alpha = alphas[-1]
                nuk = man.lincomb(x_seq[-1], -alpha, grad_seq[-1])
                x_shift = man.retr(x, nuk)
                f_shift = objective(x_shift)
                max_steps = 30
                #while f_shift > max([f_i - self.gamma * alpha * man.inner(x_i, g_i, g_i) \
                #                   for (x_i, f_i, g_i) in zip(x_seq, f_seq, grad_seq)]) and max_steps > 0:
                while f_shift > max(f_seq) - self.gamma * alpha * man.inner(x_seq[-1], grad_seq[-1], grad_seq[-1]) and max_steps > 0:
                    nuk = man.lincomb(x_seq[-1], -alpha, grad_seq[-1])
                    x_shift = man.retr(x, nuk)
                    f_shift = objective(x_shift)
                    alpha *= self.contraction_factor
                    max_steps -= 1

                if f_shift > f_seq[-1]:
                    x = x_seq[-1]
                    break
                    alpha = 0
                    x_shift = x_seq[-1]

                x_seq.append(x_shift)
                f_seq.append(f_shift)
                grad_seq.append(gradient(x_shift))

                nuk = man.lincomb(x_seq[-2], -alpha, grad_seq[-2])
                sk = man.lincomb(x_seq[-1], -alpha, man.transp(x_seq[-2], x_seq[-1], grad_seq[-2]))
                yk = man.lincomb(x_seq[-1], 1.0, grad_seq[-1], 1.0 / alpha if alpha != 0 else 0, man.transp(x_seq[-2], x_seq[-1], nuk))

                skyk = man.inner(x_seq[-1], sk, yk)
                sksk = man.inner(x_seq[-1], sk, sk)
                if skyk > 0:
                    alphas.append(min(self.alpha_max, max(self.eps, sksk / skyk)))
                else:
                    alphas.append(self.alpha_max)

                step_size = alphas[-1]

                iter = iter + 1

                if len(x_seq) > self.m:
                    x_seq = x_seq[-self.m:]
                    f_seq = f_seq[-self.m:]
                    grad_seq = grad_seq[-self.m:]

                if self._verbosity >= 2:
                    print "%5d\t%+.16e\t%.8e" % (iter + 1, f_seq[-1], sksk)

                # Check stopping conditions
                #if step_size < self._minstepsize:
                #    if self._verbosity >= 1:
                #        print ("Terminated - min stepsize reached after %d "
                #               "iterations, %.2f seconds." % (
                #                   iter, (time.time() - time0)))
                #    return x, iter

                #if sksk < self._mingradnorm:
                #    if self._verbosity >= 1:
                #        print ("Terminated - min grad norm reached after %d "
                #               "iterations, %.2f seconds." % (
                #                   iter, (time.time() - time0)))
                #    return x, iter

                if iter >= self._maxiter:
                    if self._verbosity >= 1:
                        print ("Terminated - max iterations reached after "
                               "%.2f seconds." % (time.time() - time0))
                    return x, iter

                if time.time() >= time0 + self._maxtime:
                    if self._verbosity >= 1:
                        print ("Terminated - max time reached after %d "
                               "iterations." % iter)
                    return x, iter
