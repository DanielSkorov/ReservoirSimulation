import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('stab')
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

from stability import (
  stabilityPT,
)


class stab(unittest.TestCase):

  def test_01(self):
    P = 2e6
    T = 40. + 273.15
    yi = np.array([.15, .85])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='ss', tol=tol, maxiter=1)
    res = stab.run(P, T, yi)
    self.assertTrue((res.stable) & (res.gnorm < tol))
    pass

  def test_02(self):
    P = 6e6
    T = 10. + 273.15
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='ss', tol=tol, maxiter=7)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_03(self):
    P = 101325.
    T = 20. + 273.15
    yi = np.array([.1, .6, .3])
    Pci = np.array([4.600155e6, 3.2890095e6, 22.04832e6])
    Tci = np.array([190.6, 507.5, 647.3])
    wi = np.array([.008, .27504, .344])
    mwi = np.array([0.016043, 0.086, 0.018015])
    vsi = np.array([0., 0., 0.])
    dij = np.array([.0253, 0.4907, 0.48])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='ss', level=1, tol=tol, maxiter=4)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_04(self):
    P = 17e6
    T = 68. + 273.15
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
    tol = 1e-6
    stab = stabilityPT(pr, method='ss', tol=tol, maxiter=71)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_05(self):
    P = 2e6
    T = 40. + 273.15
    yi = np.array([.15, .85])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=3)
    res = stab.run(P, T, yi)
    self.assertTrue((res.stable) & (res.gnorm < tol))
    pass

  def test_06(self):
    P = 6e6
    T = 10. + 273.15
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=6)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_07(self):
    P = 101325.
    T = 20. + 273.15
    yi = np.array([.1, .6, .3])
    Pci = np.array([4.600155e6, 3.2890095e6, 22.04832e6])
    Tci = np.array([190.6, 507.5, 647.3])
    wi = np.array([.008, .27504, .344])
    mwi = np.array([0.016043, 0.086, 0.018015])
    vsi = np.array([0., 0., 0.])
    dij = np.array([.0253, 0.4907, 0.48])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', level=1, tol=tol, maxiter=5)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_08(self):
    P = 17e6
    T = 68. + 273.15
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
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=20)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_09(self):
    P = 2e6
    T = 40. + 273.15
    yi = np.array([.15, .85])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='newton', tol=tol, maxiter=3)
    res = stab.run(P, T, yi)
    self.assertTrue((res.stable) & (res.gnorm < tol))
    pass

  def test_10(self):
    P = 6e6
    T = 10. + 273.15
    yi = np.array([.9, .1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([.225, .008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='newton', tol=tol, maxiter=4)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_11(self):
    P = 17e6
    T = 68. + 273.15
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
    tol = 1e-6
    stab = stabilityPT(pr, method='newton', tol=tol, maxiter=8)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_12(self):
    P = 17e6
    T = 68. + 273.15
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
    tol = 1e-6
    stab = stabilityPT(pr, method='ss-newton', tol=tol, maxiter=15)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_13(self):
    P = 17e6
    T = 68. + 273.15
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
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss-newton', tol=tol, maxiter=11)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_14(self):
    P = 1e6
    T = 160.
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
    tol = 1e-6
    stab = stabilityPT(pr, method='ss', tol=tol, maxiter=3)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_15(self):
    P = 1e6
    T = 160.
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
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=6)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_16(self):
    P = 1e6
    T = 160.
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
    tol = 1e-6
    stab = stabilityPT(pr, method='newton', tol=tol, maxiter=4)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_17(self):
    P = 8.7625e6
    T = 316.48
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
    tol = 1e-6
    stab = stabilityPT(pr, method='ss', tol=tol, maxiter=33)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_18(self):
    P = 8.7625e6
    T = 316.48
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
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=18)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_19(self):
    P = 8.7625e6
    T = 316.48
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
    tol = 1e-6
    stab = stabilityPT(pr, method='newton', tol=tol, maxiter=6)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_20(self):
    P = 8.7625e6
    T = 316.48
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
    tol = 1e-6
    stab = stabilityPT(pr, method='ss-newton', tol=tol, maxiter=13)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_21(self):
    P = 8.7625e6
    T = 316.48
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
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss-newton', tol=tol, maxiter=9)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_22(self):
    P = 58948692.3
    T = 373.15
    yi = np.array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np.array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np.array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np.array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np.array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=55)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_23(self):
    P = 16.5e6
    T = 68. + 273.15
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
    vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    dij = np.array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=22)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_24(self):
    P = 16e6
    T = 62.68 + 273.15
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
    vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    dij = np.array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-6
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=26)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_25(self):
    P = 10781100.0
    T = 233.15
    yi = np.array([0.26, 0.04, 0.66, 0.03, 0.01])
    Pci = np.array([89.37, 73.76, 46.00, 48.84, 42.46]) * 1e5
    Tci = np.array([373.2, 304.2, 190.6, 305.4, 369.8])
    mwi = np.array([34.08, 44.01, 16.043, 30.07, 44.097]) / 1e3
    wi = np.array([0.117, 0.225, 0.008, 0.098, 0.152])
    vsi = np.array([0., 0., 0., 0., 0.])
    dij = np.array([
      0.135,
      0.070, 0.105,
      0.085, 0.130, 0.005,
      0.080, 0.125, 0.010, 0.005,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-8
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=27)
    res = stab.run(P, T, yi)
    self.assertTrue((res.stable) & (res.gnorm < tol))
    pass

  def test_26(self):
    P = 64843561.5
    T = 373.15
    yi = np.array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np.array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np.array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np.array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np.array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-10
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=49)
    res = stab.run(P, T, yi)
    self.assertTrue((res.stable) & (res.gnorm < tol))
    pass

  def test_27(self):
    P = 23.3e6
    T = 373.15
    yi = np.array([0.55, 0.1323, 0.0459, 0.0376, 0.0149, 0.0542, 0.0711,
                   0.0370, 0.0238, 0.0124, 0.0208])
    Pci = np.array([73.82, 45.40, 48.20, 41.90, 37.50, 28.82, 23.74, 18.59,
                    14.80, 11.95, 8.52]) * 1e5
    Tci = np.array([304.21, 190.60, 305.40, 369.80, 425.20, 516.67, 590.00,
                    668.61, 745.78, 812.67, 914.89])
    mwi = np.array([44.0, 16.0, 30.1, 44.1, 58.1, 89.9, 125.7, 174.4, 240.3,
                    336.1, 536.7]) / 1e3
    wi = np.array([0.225, 0.008, 0.098, 0.152, 0.193, 0.265, 0.364, 0.499,
                   0.661, 0.877, 1.279])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.105,
      0.115, 0.000,
      0.115, 0.000, 0.002,
      0.115, 0.000, 0.005, 0.001,
      0.115, 0.045, 0.016, 0.008, 0.003,
      0.115, 0.055, 0.027, 0.016, 0.009, 0.001,
      0.115, 0.055, 0.042, 0.027, 0.019, 0.006, 0.002,
      0.115, 0.060, 0.057, 0.040, 0.030, 0.013, 0.006, 0.001,
      0.115, 0.080, 0.070, 0.051, 0.040, 0.020, 0.011, 0.004, 0.001,
      0.115, 0.280, 0.089, 0.068, 0.055, 0.032, 0.020, 0.010, 0.004, 0.001,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-10
    stab = stabilityPT(pr, method='ss-newton', tol=tol, maxiter=33)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_28(self):
    P = 23.3e6
    T = 373.15
    yi = np.array([0.55, 0.1323, 0.0459, 0.0376, 0.0149, 0.0542, 0.0711,
                   0.0370, 0.0238, 0.0124, 0.0208])
    Pci = np.array([73.82, 45.40, 48.20, 41.90, 37.50, 28.82, 23.74, 18.59,
                    14.80, 11.95, 8.52]) * 1e5
    Tci = np.array([304.21, 190.60, 305.40, 369.80, 425.20, 516.67, 590.00,
                    668.61, 745.78, 812.67, 914.89])
    mwi = np.array([44.0, 16.0, 30.1, 44.1, 58.1, 89.9, 125.7, 174.4, 240.3,
                    336.1, 536.7]) / 1e3
    wi = np.array([0.225, 0.008, 0.098, 0.152, 0.193, 0.265, 0.364, 0.499,
                   0.661, 0.877, 1.279])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.105,
      0.115, 0.000,
      0.115, 0.000, 0.002,
      0.115, 0.000, 0.005, 0.001,
      0.115, 0.045, 0.016, 0.008, 0.003,
      0.115, 0.055, 0.027, 0.016, 0.009, 0.001,
      0.115, 0.055, 0.042, 0.027, 0.019, 0.006, 0.002,
      0.115, 0.060, 0.057, 0.040, 0.030, 0.013, 0.006, 0.001,
      0.115, 0.080, 0.070, 0.051, 0.040, 0.020, 0.011, 0.004, 0.001,
      0.115, 0.280, 0.089, 0.068, 0.055, 0.032, 0.020, 0.010, 0.004, 0.001,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-10
    stab = stabilityPT(pr, method='qnss-newton', tol=tol, maxiter=33)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_29(self):
    P = 345227.1214393105
    T = 68. + 273.15
    yi = np.array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np.array([45.99, 48.72, 42.48, 37.96, 23.975]) * 1e5
    Tci = np.array([190.56, 305.32, 369.83, 425.12, 551.022])
    mwi = np.array([16.043, 30.07, 44.097, 58.123, 120.0]) / 1e3
    wi = np.array([0.012, 0.1, 0.152, 0.2, 0.414])
    vsi = np.array([-0.1017, -0.0766, -0.0499, -0.0219, 0.0909])
    dij = np.array([
      0.0027,
      0.0085, 0.0017,
      0.0147, 0.0049, 0.0009,
      0.0393, 0.0219, 0.0117, 0.0062,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-10
    stab = stabilityPT(pr, method='qnss-newton', tol=tol, maxiter=4)
    res = stab.run(P, T, yi)
    self.assertTrue((res.stable) & (res.gnorm < tol))
    pass

  def test_30(self):
    P = 5e6
    T = 0. + 273.15
    yi = np.array([0.0001, 0.3499, 0.0300, 0.0400, 0.0600, 0.0400, 0.0300,
                   0.0500, 0.0500, 0.3000, 0.0500])
    Pci = np.array([73.84, 46.04, 48.84, 42.57, 37.46, 32.77, 29.72, 27.37,
                    25.10, 22.06, 15.86]) * 1e5
    Tci = np.array([304.04, 190.59, 305.21, 369.71, 419.04, 458.98, 507.54,
                    540.32, 568.93, 615.15, 694.82])
    mwi = np.array([44.01, 16.04, 30.07, 44.10, 58.12, 72.15, 86.18, 100.20,
                    114.23, 142.29, 198.39]) / 1e3
    wi = np.array([0.225, 0.010, 0.099, 0.152, 0.187, 0.252, 0.296, 0.351,
                   0.394, 0.491, 0.755])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.100,
      0.130, 0.000,
      0.135, 0.000, 0.000,
      0.130, 0.000, 0.000, 0.000,
      0.125, 0.000, 0.000, 0.000, 0.000,
      0.120, 0.020, 0.030, 0.030, 0.030, 0.000,
      0.120, 0.030, 0.030, 0.030, 0.030, 0.000, 0.000,
      0.120, 0.035, 0.030, 0.030, 0.030, 0.000, 0.000, 0.000,
      0.120, 0.040, 0.030, 0.030, 0.030, 0.000, 0.000, 0.000, 0.000,
      0.120, 0.060, 0.030, 0.030, 0.030, 0.000, 0.000, 0.000, 0.000, 0.000,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-10
    stab = stabilityPT(pr, method='qnss-newton', tol=tol, maxiter=7)
    res = stab.run(P, T, yi)
    self.assertTrue((not res.stable) & (res.gnorm < tol))
    pass

  def test_31(self):
    P = 3034482.7586206896
    T = 225.7362068965517
    yi = np.array([0.9, 0.1])
    Pci = np.array([7.37646e6, 4.600155e6])
    Tci = np.array([304.2, 190.6])
    wi = np.array([0.225, 0.008])
    mwi = np.array([0.04401, 0.016043])
    vsi = np.array([0., 0.])
    dij = np.array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    tol = 1e-8
    stab = stabilityPT(pr, method='qnss', tol=tol, maxiter=6)
    res = stab.run(P, T, yi)
    self.assertTrue((res.stable) & (res.gnorm < tol))
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
