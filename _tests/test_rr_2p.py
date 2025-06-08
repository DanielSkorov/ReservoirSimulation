import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('rr')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
  '%(process)d:%(name)s:%(levelname)s:\n\t%(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

import unittest

import numpy as np
np.set_printoptions(linewidth=np.inf)

from rr import (
  solve2p_FGH,
)


class rr_fgh(unittest.TestCase):

  @staticmethod
  def fD(a, yi, di):
    denom = 1. / (di * (a + 1.) + a)
    return yi[0] + a * yi[1:-1].dot(denom) - yi[-1] * a

  def check_solution(self, F, kvi, yi, tol):
    idx = kvi.argsort()[::-1]
    ysi = yi[idx]
    kvsi = kvi[idx]
    ci = 1. / (1. - kvsi)
    di = (ci[0] - ci[1:-1]) / (ci[-1] - ci[0])
    a = (F - ci[0]) / (ci[-1] - F)
    D = self.fD(a, ysi, di)
    solved = np.abs(D) < tol
    nfwindow = 1. / (1. - kvi.max()) < F < 1. / (1. - kvi.min())
    return solved & nfwindow

  def test_01(self):
    yi = np.array([0.770, 0.200, 0.010, 0.010, 0.005, 0.005])
    kvi = np.array([1.00003, 1.00002, 1.00001, 0.99999, 0.99998, 0.99997])
    tol = np.float64(1e-10)
    maxiter = 2
    F = solve2p_FGH(kvi, yi, tol, maxiter)
    self.assertTrue(self.check_solution(F, kvi, yi, tol))
    pass

  def test_02(self):
    yi = np.array([0.44, 0.55, 3.88E-03, 2.99E-03, 2.36E-03, 1.95E-03])
    kvi = np.array([161.59, 6.90, 0.15, 1.28E-03, 5.86E-06, 2.32E-08])
    tol = np.float64(1e-10)
    maxiter = 3
    F = solve2p_FGH(kvi, yi, tol, maxiter)
    self.assertTrue(self.check_solution(F, kvi, yi, tol))
    pass

  def test_03(self):
    eps = 1e-9
    kvi = 1. + np.array([2.*eps, 1.5*eps, eps, -eps, -1.5*eps, -2.*eps])
    yi = np.full_like(kvi, 1. / 6.)
    tol = np.float64(1e-10)
    maxiter = 1
    F = solve2p_FGH(kvi, yi, tol, maxiter)
    self.assertTrue(self.check_solution(F, kvi, yi, tol))
    pass

  def test_04(self):
    yi = np.array([
      0.8097,
      0.0566,
      0.0306,
      0.0457,
      0.0330,
      0.0244,
    ])
    kvi = np.array([
      1.000065,
      0.999922,
      0.999828,
      0.999650,
      0.999490,
      0.999282,
    ])
    tol = np.float64(1e-10)
    maxiter = 5
    F = solve2p_FGH(kvi, yi, tol, maxiter)
    self.assertTrue(self.check_solution(F, kvi, yi, tol))
    pass

  def test_05(self):
    yi = np.array([
      0.1789202106,
      0.0041006011,
      0.7815241261,
      0.0164691242,
      0.0189859122,
      0.0000000257,
    ])
    kvi = np.array([
      445.995819899,
      441.311360487,
      411.625356748,
      339.586063803,
      29.7661058122,
      0.00596602417,
    ])
    tol = np.float64(1e-8)
    maxiter = 1
    F = solve2p_FGH(kvi, yi, tol, maxiter)
    self.assertTrue(self.check_solution(F, kvi, yi, tol))
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
