import sys

sys.path.append('../_src/')

# import logging

# logger = logging.getLogger('rr')
# logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(process)d:%(name)s:%(levelname)s:\n\t%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

import unittest

import numpy as np
np.set_printoptions(linewidth=np.inf)

from rr import solveNp


class rr_fgh(unittest.TestCase):

  def check_solution(self, Fj, kvji, yi, tol):
    Aji = 1. - kvji
    ti = 1. - Fj.dot(Aji)
    gj = Aji.dot(yi / ti)
    solved = np.linalg.norm(gj) < tol
    nfwindow = np.all(ti > 0.)
    return solved & nfwindow

  def test_case_1(self):
    yi = np.array(
      [0.204322076984, 0.070970999150, 0.267194323384, 0.296291964579, 0.067046080882, 0.062489248292, 0.031685306730]
    )
    kvji = np.array([
      [1.23466988745, 0.89727701141, 2.29525708098, 1.58954899888, 0.23349348597, 0.02038108640, 1.40715641002],
      [1.52713341421, 0.02456487977, 1.46348240453, 1.16090546194, 0.24166289908, 0.14815282572, 14.3128010831],
    ])
    Fj0 = np.array([0.3333, 0.3333])
    tol = np.float64(1e-6)
    Niter = 5
    tol_ls = np.float64(1e-5)
    Niter_ls = 10
    Fj = solveNp(kvji, yi, Fj0, tol, Niter, tol_ls, Niter_ls)
    self.assertTrue(self.check_solution(Fj, kvji, yi, tol))
    pass


if __name__ == '__main__':
  unittest.main()
