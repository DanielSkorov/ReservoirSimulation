import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('flash')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

import unittest

import numpy as np
np.set_printoptions(linewidth=np.inf)

from eos import (
  pr78,
)

from flash import (
  flash2pPT,
)


class flash(unittest.TestCase):

  def test_01(self):
    P = np.float64(6e6)
    T = np.float64(10. + 273.15)
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='ss', tol=tol, maxiter=7,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_02(self):
    P = np.float64(6e6)
    T = np.float64(10. + 273.15)
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='qnss', tol=tol, maxiter=6,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_03(self):
    P = np.float64(6e6)
    T = np.float64(10. + 273.15)
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='newton', tol=tol, maxiter=3,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_04(self):
    P = np.float64(6e6)
    T = np.float64(10. + 273.15)
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='ss-newton', tol=tol, maxiter=4,
                      stabkwargs=dict(method='ss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_05(self):
    P = np.float64(17e6)
    T = np.float64(68. + 273.15)
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([4.599e6, 4.872e6, 4.248e6, 3.796e6, 2.398e6])
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02])
    wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414])
    mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120])
    vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661])
    dij = np.array([
      0.002689,
      0.008537, 0.001662,
      0.014748, 0.004914, 0.000866,
      0.039265, 0.021924, 0.011676, 0.006228,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='ss', tol=tol, maxiter=73,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_06(self):
    P = np.float64(17e6)
    T = np.float64(68. + 273.15)
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([4.599e6, 4.872e6, 4.248e6, 3.796e6, 2.398e6])
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02])
    wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414])
    mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120])
    vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661])
    dij = np.array([
      0.002689,
      0.008537, 0.001662,
      0.014748, 0.004914, 0.000866,
      0.039265, 0.021924, 0.011676, 0.006228,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='qnss', tol=tol, maxiter=11,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_07(self):
    P = np.float64(17e6)
    T = np.float64(68. + 273.15)
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([4.599e6, 4.872e6, 4.248e6, 3.796e6, 2.398e6])
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02])
    wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414])
    mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120])
    vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661])
    dij = np.array([
      0.002689,
      0.008537, 0.001662,
      0.014748, 0.004914, 0.000866,
      0.039265, 0.021924, 0.011676, 0.006228,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='newton', tol=tol, maxiter=4,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_08(self):
    P = np.float64(17e6)
    T = np.float64(68. + 273.15)
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([4.599e6, 4.872e6, 4.248e6, 3.796e6, 2.398e6])
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.02])
    wi = np.array([0.012, 0.100, 0.152, 0.200, 0.414])
    mwi = np.array([0.016043, 0.03007, 0.044097, 0.058123, 0.120])
    vsi = np.array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661])
    dij = np.array([
      0.002689,
      0.008537, 0.001662,
      0.014748, 0.004914, 0.000866,
      0.039265, 0.021924, 0.011676, 0.006228,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='qnss-newton', tol=tol, maxiter=4,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_09(self):
    P = np.float64(1e6)
    T = np.float64(160.)
    yi = np.array([0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140])
    Pci = np.array([4.599, 4.872, 4.248, 3.796, 3.370, 3.025, 3.400]) * 1e6
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 469.70, 507.60, 126.20])
    wi = np.array([0.0115, 0.0995, 0.1523, 0.2002, 0.2515, 0.3013, 0.0377])
    mwi = np.array([0.0160, 0.0301, 0.0441, 0.0581, 0.0722, 0.0860, 0.0280])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.00269,
      0.00854, 0.00166,
      0.01475, 0.00491, 0.00087,
      0.02064, 0.00858, 0.00271, 0.00051,
      0.02535, 0.01175, 0.00462, 0.00149, 0.0002,
      0.02500, 0.06000, 0.09000, 0.09500, 0.11000, 0.11000,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='ss', tol=tol, maxiter=11,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_10(self):
    P = np.float64(1e6)
    T = np.float64(160.)
    yi = np.array([0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140])
    Pci = np.array([4.599, 4.872, 4.248, 3.796, 3.370, 3.025, 3.400]) * 1e6
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 469.70, 507.60, 126.20])
    wi = np.array([0.0115, 0.0995, 0.1523, 0.2002, 0.2515, 0.3013, 0.0377])
    mwi = np.array([0.0160, 0.0301, 0.0441, 0.0581, 0.0722, 0.0860, 0.0280])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.00269,
      0.00854, 0.00166,
      0.01475, 0.00491, 0.00087,
      0.02064, 0.00858, 0.00271, 0.00051,
      0.02535, 0.01175, 0.00462, 0.00149, 0.0002,
      0.02500, 0.06000, 0.09000, 0.09500, 0.11000, 0.11000,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='qnss', tol=tol, maxiter=9,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_11(self):
    P = np.float64(1e6)
    T = np.float64(160.)
    yi = np.array([0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140])
    Pci = np.array([4.599, 4.872, 4.248, 3.796, 3.370, 3.025, 3.400]) * 1e6
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 469.70, 507.60, 126.20])
    wi = np.array([0.0115, 0.0995, 0.1523, 0.2002, 0.2515, 0.3013, 0.0377])
    mwi = np.array([0.0160, 0.0301, 0.0441, 0.0581, 0.0722, 0.0860, 0.0280])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.00269,
      0.00854, 0.00166,
      0.01475, 0.00491, 0.00087,
      0.02064, 0.00858, 0.00271, 0.00051,
      0.02535, 0.01175, 0.00462, 0.00149, 0.0002,
      0.02500, 0.06000, 0.09000, 0.09500, 0.11000, 0.11000,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='newton', tol=tol, maxiter=5,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_12(self):
    P = np.float64(1e6)
    T = np.float64(160.)
    yi = np.array([0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140])
    Pci = np.array([4.599, 4.872, 4.248, 3.796, 3.370, 3.025, 3.400]) * 1e6
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 469.70, 507.60, 126.20])
    wi = np.array([0.0115, 0.0995, 0.1523, 0.2002, 0.2515, 0.3013, 0.0377])
    mwi = np.array([0.0160, 0.0301, 0.0441, 0.0581, 0.0722, 0.0860, 0.0280])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.00269,
      0.00854, 0.00166,
      0.01475, 0.00491, 0.00087,
      0.02064, 0.00858, 0.00271, 0.00051,
      0.02535, 0.01175, 0.00462, 0.00149, 0.0002,
      0.02500, 0.06000, 0.09000, 0.09500, 0.11000, 0.11000,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='ss-newton', tol=tol, maxiter=7,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_13(self):
    P = np.float64(8.7625e6)
    T = np.float64(316.48)
    yi = np.array([
      0.590516, 0.028933, 0.072729, 0.081162, 0.131012, 0.064671, 0.030979,
    ])
    Pci = np.array([73.36, 45.99, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np.array([304.20, 166.67, 338.81, 466.12, 611.11, 777.78, 972.22])
    wi = np.array([0.225, 0.008, 0.126, 0.244, 0.639, 1.000, 1.281])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    mwi = np.array([0.044, 0.016, 0.037, 0.072, 0.161, 0.312, 0.495])
    dij = np.array([
      0.05,
      0.05, 0.00853,
      0.05, 0.02064, 0.00271,
      0.09, 0.05428, 0.02079, 0.00863,
      0.09, 0.09301, 0.04829, 0.02887, 0.00615,
      0.09, 0.12546, 0.07402, 0.05002, 0.01794, 0.00316,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='ss', tol=tol, maxiter=211,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_14(self):
    P = np.float64(8.7625e6)
    T = np.float64(316.48)
    yi = np.array([
      0.590516, 0.028933, 0.072729, 0.081162, 0.131012, 0.064671, 0.030979,
    ])
    Pci = np.array([73.36, 45.99, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np.array([304.20, 166.67, 338.81, 466.12, 611.11, 777.78, 972.22])
    wi = np.array([0.225, 0.008, 0.126, 0.244, 0.639, 1.000, 1.281])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    mwi = np.array([0.044, 0.016, 0.037, 0.072, 0.161, 0.312, 0.495])
    dij = np.array([
      0.05,
      0.05, 0.00853,
      0.05, 0.02064, 0.00271,
      0.09, 0.05428, 0.02079, 0.00863,
      0.09, 0.09301, 0.04829, 0.02887, 0.00615,
      0.09, 0.12546, 0.07402, 0.05002, 0.01794, 0.00316,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='qnss', tol=tol, maxiter=36,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_15(self):
    P = np.float64(8.7625e6)
    T = np.float64(316.48)
    yi = np.array([
      0.590516, 0.028933, 0.072729, 0.081162, 0.131012, 0.064671, 0.030979,
    ])
    Pci = np.array([73.36, 45.99, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np.array([304.20, 166.67, 338.81, 466.12, 611.11, 777.78, 972.22])
    wi = np.array([0.225, 0.008, 0.126, 0.244, 0.639, 1.000, 1.281])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    mwi = np.array([0.044, 0.016, 0.037, 0.072, 0.161, 0.312, 0.495])
    dij = np.array([
      0.05,
      0.05, 0.00853,
      0.05, 0.02064, 0.00271,
      0.09, 0.05428, 0.02079, 0.00863,
      0.09, 0.09301, 0.04829, 0.02887, 0.00615,
      0.09, 0.12546, 0.07402, 0.05002, 0.01794, 0.00316,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='newton', tol=tol, maxiter=12,
                      forcenewton=True, stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_16(self):
    P = np.float64(8.7625e6)
    T = np.float64(316.48)
    yi = np.array([
      0.590516, 0.028933, 0.072729, 0.081162, 0.131012, 0.064671, 0.030979,
    ])
    Pci = np.array([73.36, 45.99, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np.array([304.20, 166.67, 338.81, 466.12, 611.11, 777.78, 972.22])
    wi = np.array([0.225, 0.008, 0.126, 0.244, 0.639, 1.000, 1.281])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    mwi = np.array([0.044, 0.016, 0.037, 0.072, 0.161, 0.312, 0.495])
    dij = np.array([
      0.05,
      0.05, 0.00853,
      0.05, 0.02064, 0.00271,
      0.09, 0.05428, 0.02079, 0.00863,
      0.09, 0.09301, 0.04829, 0.02887, 0.00615,
      0.09, 0.12546, 0.07402, 0.05002, 0.01794, 0.00316,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='ss-newton', tol=tol, maxiter=37,
                      maxiter_ss=30, forcenewton=True,
                      stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass

  def test_17(self):
    P = np.float64(8.7625e6)
    T = np.float64(316.48)
    yi = np.array([
      0.590516, 0.028933, 0.072729, 0.081162, 0.131012, 0.064671, 0.030979,
    ])
    Pci = np.array([73.36, 45.99, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np.array([304.20, 166.67, 338.81, 466.12, 611.11, 777.78, 972.22])
    wi = np.array([0.225, 0.008, 0.126, 0.244, 0.639, 1.000, 1.281])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    mwi = np.array([0.044, 0.016, 0.037, 0.072, 0.161, 0.312, 0.495])
    dij = np.array([
      0.05,
      0.05, 0.00853,
      0.05, 0.02064, 0.00271,
      0.09, 0.05428, 0.02079, 0.00863,
      0.09, 0.09301, 0.04829, 0.02887, 0.00615,
      0.09, 0.12546, 0.07402, 0.05002, 0.01794, 0.00316,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-5
    flash = flash2pPT(pr, method='qnss-newton', tol=tol, maxiter=24,
                      maxiter_qnss=20, stabkwargs=dict(method='qnss'))
    res = flash.run(P, T, yi)
    self.assertTrue(res.gnorm < tol)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
