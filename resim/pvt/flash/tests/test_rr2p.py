import sys
import logging
import unittest

from numpy import (
  array as np_array,
  full_like as np_full_like,
)

from resim.pvt.flash import (
  rr2p_fgh,
  rr2p_gh,
)


logger = logging.getLogger('rr')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class rr2p(unittest.TestCase):

  def test_01(self):
    yi = np_array([0.770, 0.200, 0.010, 0.010, 0.005, 0.005])
    kvi = np_array([1.00003, 1.00002, 1.00001, 0.99999, 0.99998, 0.99997])
    f = rr2p_fgh(kvi, yi, maxiter=2)
    pass

  def test_02(self):
    yi = np_array([0.770, 0.200, 0.010, 0.010, 0.005, 0.005])
    kvi = np_array([1.00003, 1.00002, 1.00001, 0.99999, 0.99998, 0.99997])
    f = rr2p_gh(kvi, yi, maxiter=3)
    pass

  def test_03(self):
    yi = np_array([0.44, 0.55, 3.88E-03, 2.99E-03, 2.36E-03, 1.95E-03])
    kvi = np_array([161.59, 6.90, 0.15, 1.28E-03, 5.86E-06, 2.32E-08])
    f = rr2p_fgh(kvi, yi, maxiter=3)
    pass

  def test_04(self):
    yi = np_array([0.44, 0.55, 3.88E-03, 2.99E-03, 2.36E-03, 1.95E-03])
    kvi = np_array([161.59, 6.90, 0.15, 1.28E-03, 5.86E-06, 2.32E-08])
    f = rr2p_gh(kvi, yi, maxiter=5)
    pass

  def test_05(self):
    eps = 1e-9
    kvi = 1. + np_array([2.*eps, 1.5*eps, eps, -eps, -1.5*eps, -2.*eps])
    yi = np_full_like(kvi, 1. / 6.)
    f = rr2p_fgh(kvi, yi, maxiter=1)
    pass

  def test_06(self):
    eps = 1e-9
    kvi = 1. + np_array([2.*eps, 1.5*eps, eps, -eps, -1.5*eps, -2.*eps])
    yi = np_full_like(kvi, 1. / 6.)
    f = rr2p_gh(kvi, yi, maxiter=1)
    pass

  def test_07(self):
    yi = np_array([0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244])
    kvi = np_array([1.000065, 0.999922, 0.999828, 0.999650, 0.999490,
                    0.999282])
    f = rr2p_fgh(kvi, yi, maxiter=5)
    pass

  def test_08(self):
    yi = np_array([0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244])
    kvi = np_array([1.000065, 0.999922, 0.999828, 0.999650, 0.999490,
                    0.999282])
    f = rr2p_gh(kvi, yi, maxiter=6)
    pass

  def test_09(self):
    yi = np_array([0.1789202106, 0.0041006011, 0.7815241261, 0.0164691242,
                   0.0189859122, 0.0000000257])
    kvi = np_array([445.995819899, 441.311360487, 411.625356748,
                    339.586063803, 29.7661058122, 0.00596602417])
    f = rr2p_fgh(kvi, yi, maxiter=2)
    pass

  def test_10(self):
    yi = np_array([0.1789202106, 0.0041006011, 0.7815241261, 0.0164691242,
                   0.0189859122, 0.0000000257])
    kvi = np_array([445.995819899, 441.311360487, 411.625356748,
                    339.586063803, 29.7661058122, 0.00596602417])
    f = rr2p_gh(kvi, yi, maxiter=2)
    pass

  def test_11(self):
    yi = np_array([
      8.49690204e-5, 2.20315563e-3, 9.31360141e-1, 4.77686855e-2,
      1.33133987e-2, 1.58602664e-3, 2.42095466e-3, 4.68831934e-4,
      3.34433521e-4, 2.82342929e-4, 1.73279904e-4, 3.62280482e-6,
      1.52954183e-7, 4.80259710e-9, 7.42408981e-18,
    ])
    kvi = np_array([
      3.15031058e-1, 2.68764606e-2, 1.03729080e-1, 8.05158226e-1,
      3.70517991e+0, 1.07260023e+1, 1.63929253e+1, 4.99360406e+1,
      7.06745013e+1, 2.30165523e+2, 1.40995102e+3, 4.58288939e+4,
      8.80176404e+5, 2.09818157e+7, 1.40679959e+14,
    ])
    f = rr2p_fgh(kvi, yi, maxiter=1)
    pass

  def test_12(self):
    yi = np_array([
      8.49690204e-5, 2.20315563e-3, 9.31360141e-1, 4.77686855e-2,
      1.33133987e-2, 1.58602664e-3, 2.42095466e-3, 4.68831934e-4,
      3.34433521e-4, 2.82342929e-4, 1.73279904e-4, 3.62280482e-6,
      1.52954183e-7, 4.80259710e-9, 7.42408981e-18,
    ])
    kvi = np_array([
      3.15031058e-1, 2.68764606e-2, 1.03729080e-1, 8.05158226e-1,
      3.70517991e+0, 1.07260023e+1, 1.63929253e+1, 4.99360406e+1,
      7.06745013e+1, 2.30165523e+2, 1.40995102e+3, 4.58288939e+4,
      8.80176404e+5, 2.09818157e+7, 1.40679959e+14,
    ])
    f = rr2p_gh(kvi, yi, maxiter=5)
    pass

  def test_13(self):
    yi = np_array([
      8.49690204e-05, 0.002203155630, 0.931360141000, 0.047768685500,
      0.013313398700, 0.001586026640, 0.002420954660, 0.000468831934,
      0.000334433521, 0.000282342929, 0.000173279904, 3.62280482e-06,
      1.52954183e-07, 4.80259710e-09, 7.42408981e-18,
    ])
    kvi = np_array([
      2.7362474716097953000, 32.235845970268360000, 8.12903723679372300000,
      1.0359648822934502000, 0.2251226497700584000, 0.07827289602106047000,
      0.0510297806179701500, 0.0168604765150673200, 0.01189992601669396400,
      0.0036762433413586515, 0.0006072450254410366, 1.9441746797141283e-05,
      1.0546172268246862e-6, 4.634474990499742e-08, 7.8675406618576900e-15,
    ])
    f = rr2p_fgh(kvi, yi)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
