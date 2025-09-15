import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('rr')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import unittest

import numpy as np

np.set_printoptions(linewidth=np.inf)

from rr import (
  solve2p_FGH,
  solve2p_GH,
)

from matplotlib import (
  pyplot as plt,
)


class rr_fgh(unittest.TestCase):

  def test_01(self):
    yi = np.array([0.770, 0.200, 0.010, 0.010, 0.005, 0.005])
    kvi = np.array([1.00003, 1.00002, 1.00001, 0.99999, 0.99998, 0.99997])
    f = solve2p_FGH(kvi, yi, maxiter=2)
    pass

  def test_02(self):
    yi = np.array([0.770, 0.200, 0.010, 0.010, 0.005, 0.005])
    kvi = np.array([1.00003, 1.00002, 1.00001, 0.99999, 0.99998, 0.99997])
    f = solve2p_GH(kvi, yi, maxiter=3)
    pass

  def test_03(self):
    yi = np.array([0.44, 0.55, 3.88E-03, 2.99E-03, 2.36E-03, 1.95E-03])
    kvi = np.array([161.59, 6.90, 0.15, 1.28E-03, 5.86E-06, 2.32E-08])
    f = solve2p_FGH(kvi, yi, maxiter=4)
    pass

  def test_04(self):
    yi = np.array([0.44, 0.55, 3.88E-03, 2.99E-03, 2.36E-03, 1.95E-03])
    kvi = np.array([161.59, 6.90, 0.15, 1.28E-03, 5.86E-06, 2.32E-08])
    f = solve2p_GH(kvi, yi, maxiter=6)
    pass

  def test_05(self):
    eps = 1e-9
    kvi = 1. + np.array([2.*eps, 1.5*eps, eps, -eps, -1.5*eps, -2.*eps])
    yi = np.full_like(kvi, 1. / 6.)
    f = solve2p_FGH(kvi, yi, maxiter=1)
    pass

  def test_06(self):
    eps = 1e-9
    kvi = 1. + np.array([2.*eps, 1.5*eps, eps, -eps, -1.5*eps, -2.*eps])
    yi = np.full_like(kvi, 1. / 6.)
    f = solve2p_GH(kvi, yi, maxiter=1)
    pass

  def test_07(self):
    yi = np.array([0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244])
    kvi = np.array([1.000065, 0.999922, 0.999828, 0.999650, 0.999490,
                    0.999282])
    f = solve2p_FGH(kvi, yi, maxiter=6)
    pass

  def test_08(self):
    yi = np.array([0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244])
    kvi = np.array([1.000065, 0.999922, 0.999828, 0.999650, 0.999490,
                    0.999282])
    f = solve2p_GH(kvi, yi, maxiter=6)
    pass

  def test_09(self):
    yi = np.array([0.1789202106, 0.0041006011, 0.7815241261, 0.0164691242,
                   0.0189859122, 0.0000000257])
    kvi = np.array([445.995819899, 441.311360487, 411.625356748,
                    339.586063803, 29.7661058122, 0.00596602417])
    f = solve2p_FGH(kvi, yi, tol=1e-8, maxiter=1)
    pass

  def test_10(self):
    yi = np.array([0.1789202106, 0.0041006011, 0.7815241261, 0.0164691242,
                   0.0189859122, 0.0000000257])
    kvi = np.array([445.995819899, 441.311360487, 411.625356748,
                    339.586063803, 29.7661058122, 0.00596602417])
    f = solve2p_GH(kvi, yi, tol=1e-8, maxiter=3)
    pass

  def test_11(self):
    yi = np.array([
      8.49690204e-5, 2.20315563e-3, 9.31360141e-1, 4.77686855e-2,
      1.33133987e-2, 1.58602664e-3, 2.42095466e-3, 4.68831934e-4,
      3.34433521e-4, 2.82342929e-4, 1.73279904e-4, 3.62280482e-6,
      1.52954183e-7, 4.80259710e-9, 7.42408981e-18,
    ])
    kvi = np.array([
      3.15031058e-1, 2.68764606e-2, 1.03729080e-1, 8.05158226e-1,
      3.70517991e+0, 1.07260023e+1, 1.63929253e+1, 4.99360406e+1,
      7.06745013e+1, 2.30165523e+2, 1.40995102e+3, 4.58288939e+4,
      8.80176404e+5, 2.09818157e+7, 1.40679959e+14,
    ])
    f = solve2p_FGH(kvi, yi, maxiter=1)
    pass

  def test_12(self):
    yi = np.array([
      8.49690204e-5, 2.20315563e-3, 9.31360141e-1, 4.77686855e-2,
      1.33133987e-2, 1.58602664e-3, 2.42095466e-3, 4.68831934e-4,
      3.34433521e-4, 2.82342929e-4, 1.73279904e-4, 3.62280482e-6,
      1.52954183e-7, 4.80259710e-9, 7.42408981e-18,
    ])
    kvi = np.array([
      3.15031058e-1, 2.68764606e-2, 1.03729080e-1, 8.05158226e-1,
      3.70517991e+0, 1.07260023e+1, 1.63929253e+1, 4.99360406e+1,
      7.06745013e+1, 2.30165523e+2, 1.40995102e+3, 4.58288939e+4,
      8.80176404e+5, 2.09818157e+7, 1.40679959e+14,
    ])
    f = solve2p_GH(kvi, yi, maxiter=5)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
