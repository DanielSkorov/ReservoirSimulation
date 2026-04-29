import sys
import logging
import unittest

from functools import (
  partial,
)

from numpy import (
  array as np_array,
)

from resim.pvt.eos import (
  pr78,
)

from resim.pvt.stab import (
  _stabPT_qnssnewt,
  _stabPT_ssbfgs,
  _stabPT_ssnewt,
  stabtest,
)


logger = logging.getLogger('stab')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class stabtestPT(unittest.TestCase):

  def test_01(self):
    logger.info('\nTest #01.\nComponents: C1 CO2')
    P = 2e6
    T = 40. + 273.15
    yi = np_array([0.15, 0.85])
    Pci = np_array([7.37646e6, 4.600155e6])
    Tci = np_array([304.2, 190.6])
    wi = np_array([0.225, 0.008])
    mwi = np_array([0.04401, 0.016043])
    dij = np_array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=1)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_02(self):
    logger.info('\nTest #02.\nComponents: C1 CO2')
    P = 2e6
    T = 40. + 273.15
    yi = np_array([0.15, 0.85])
    Pci = np_array([7.37646e6, 4.600155e6])
    Tci = np_array([304.2, 190.6])
    wi = np_array([0.225, 0.008])
    mwi = np_array([0.04401, 0.016043])
    dij = np_array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=2)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_03(self):
    logger.info('\nTest #03.\nComponents: C1 CO2')
    P = 2e6
    T = 40. + 273.15
    yi = np_array([0.15, 0.85])
    Pci = np_array([7.37646e6, 4.600155e6])
    Tci = np_array([304.2, 190.6])
    wi = np_array([0.225, 0.008])
    mwi = np_array([0.04401, 0.016043])
    dij = np_array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=1)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_04(self):
    logger.info('\nTest #04.\nComponents: C1 CO2')
    P = 6e6
    T = 10. + 273.15
    yi = np_array([0.9, 0.1])
    Pci = np_array([7.37646e6, 4.600155e6])
    Tci = np_array([304.2, 190.6])
    wi = np_array([0.225, 0.008])
    mwi = np_array([0.04401, 0.016043])
    dij = np_array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=11)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_05(self):
    logger.info('\nTest #05.\nComponents: C1 CO2')
    P = 6e6
    T = 10. + 273.15
    yi = np_array([0.9, 0.1])
    Pci = np_array([7.37646e6, 4.600155e6])
    Tci = np_array([304.2, 190.6])
    wi = np_array([0.225, 0.008])
    mwi = np_array([0.04401, 0.016043])
    dij = np_array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=6)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_06(self):
    logger.info('\nTest #06.\nComponents: C1 CO2')
    P = 6e6
    T = 10. + 273.15
    yi = np_array([0.9, 0.1])
    Pci = np_array([7.37646e6, 4.600155e6])
    Tci = np_array([304.2, 190.6])
    wi = np_array([0.225, 0.008])
    mwi = np_array([0.04401, 0.016043])
    dij = np_array([0.025])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=11)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_07(self):
    logger.info('\nTest #07.\nComponents: C1 C6 H2O')
    P = 101325.
    T = 20. + 273.15
    yi = np_array([0.1, 0.6, 0.3])
    Pci = np_array([4.600155e6, 3.2890095e6, 22.04832e6])
    Tci = np_array([190.6, 507.5, 647.3])
    wi = np_array([0.008, 0.27504, 0.344])
    mwi = np_array([0.016043, 0.086, 0.018015])
    dij = np_array([0.0253, 0.4907, 0.48])
    pr = pr78(Pci, Tci, wi, mwi, dij, kvlevel=1)
    solver = partial(_stabPT_ssnewt, maxiter=5)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_08(self):
    logger.info('\nTest #08.\nComponents: C1 C6 H2O')
    P = 101325.
    T = 20. + 273.15
    yi = np_array([0.1, 0.6, 0.3])
    Pci = np_array([4.600155e6, 3.2890095e6, 22.04832e6])
    Tci = np_array([190.6, 507.5, 647.3])
    wi = np_array([0.008, 0.27504, 0.344])
    mwi = np_array([0.016043, 0.086, 0.018015])
    dij = np_array([0.0253, 0.4907, 0.48])
    pr = pr78(Pci, Tci, wi, mwi, dij, kvlevel=1)
    solver = partial(_stabPT_qnssnewt, maxiter=4)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_09(self):
    logger.info('\nTest #09.\nComponents: C1 C6 H2O')
    P = 101325.
    T = 20. + 273.15
    yi = np_array([0.1, 0.6, 0.3])
    Pci = np_array([4.600155e6, 3.2890095e6, 22.04832e6])
    Tci = np_array([190.6, 507.5, 647.3])
    wi = np_array([0.008, 0.27504, 0.344])
    mwi = np_array([0.016043, 0.086, 0.018015])
    dij = np_array([0.0253, 0.4907, 0.48])
    pr = pr78(Pci, Tci, wi, mwi, dij, kvlevel=1)
    solver = partial(_stabPT_ssbfgs, maxiter=5)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_10(self):
    logger.info('\nTest #10.\nComponents: C1 C2 C3 NC4 C5+')
    P = 17e6
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([4.599e6, 4.872e6, 4.248e6, 3.796e6, 2.398e6])
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.02])
    wi = np_array([0.012, 0.100, 0.152, 0.200, 0.414])
    mwi = np_array([0.016043, 0.03007, 0.044097, 0.058123, 0.120])
    s0i = np_array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661])
    dij = np_array([
      0.002689,
      0.008537, 0.001662,
      0.014748, 0.004914, 0.000866,
      0.039265, 0.021924, 0.011676, 0.006228,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i)
    solver = partial(_stabPT_ssnewt, maxiter=16)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_11(self):
    logger.info('\nTest #11.\nComponents: C1 C2 C3 NC4 C5+')
    P = 17e6
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([4.599e6, 4.872e6, 4.248e6, 3.796e6, 2.398e6])
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.02])
    wi = np_array([0.012, 0.100, 0.152, 0.200, 0.414])
    mwi = np_array([0.016043, 0.03007, 0.044097, 0.058123, 0.120])
    s0i = np_array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661])
    dij = np_array([
      0.002689,
      0.008537, 0.001662,
      0.014748, 0.004914, 0.000866,
      0.039265, 0.021924, 0.011676, 0.006228,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i)
    solver = partial(_stabPT_qnssnewt, maxiter=10)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_12(self):
    logger.info('\nTest #12.\nComponents: C1 C2 C3 NC4 C5+')
    P = 17e6
    T = 68. + 273.15
    yi = np_array([0.7167, 0.0895, 0.0917, 0.0448, 0.0573])
    Pci = np_array([4.599e6, 4.872e6, 4.248e6, 3.796e6, 2.398e6])
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 551.02])
    wi = np_array([0.012, 0.100, 0.152, 0.200, 0.414])
    mwi = np_array([0.016043, 0.03007, 0.044097, 0.058123, 0.120])
    s0i = np_array([-0.1595, -0.1134, -0.0863, -0.0675, 0.05661])
    dij = np_array([
      0.002689,
      0.008537, 0.001662,
      0.014748, 0.004914, 0.000866,
      0.039265, 0.021924, 0.011676, 0.006228,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij, s0i)
    solver = partial(_stabPT_ssbfgs, maxiter=18)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_13(self):
    logger.info('\nTest #13.\nComponents: C1 C2 C3 NC4 FC6 N2')
    P = 1e6
    T = 160.
    yi = np_array([0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140])
    Pci = np_array([4.599, 4.872, 4.248, 3.796, 3.370, 3.025, 3.400]) * 1e6
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 469.70, 507.60, 126.20])
    wi = np_array([0.0115, 0.0995, 0.1523, 0.2002, 0.2515, 0.3013, 0.0377])
    mwi = np_array([0.0160, 0.0301, 0.0441, 0.0581, 0.0722, 0.0860, 0.0280])
    dij = np_array([
      0.00269,
      0.00854, 0.00166,
      0.01475, 0.00491, 0.00087,
      0.02064, 0.00858, 0.00271, 0.00051,
      0.02535, 0.01175, 0.00462, 0.00149, 0.0002,
      0.02500, 0.06000, 0.09000, 0.09500, 0.11000, 0.11000,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=5)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_14(self):
    logger.info('\nTest #14.\nComponents: C1 C2 C3 NC4 FC6 N2')
    P = 1e6
    T = 160.
    yi = np_array([0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140])
    Pci = np_array([4.599, 4.872, 4.248, 3.796, 3.370, 3.025, 3.400]) * 1e6
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 469.70, 507.60, 126.20])
    wi = np_array([0.0115, 0.0995, 0.1523, 0.2002, 0.2515, 0.3013, 0.0377])
    mwi = np_array([0.0160, 0.0301, 0.0441, 0.0581, 0.0722, 0.0860, 0.0280])
    dij = np_array([
      0.00269,
      0.00854, 0.00166,
      0.01475, 0.00491, 0.00087,
      0.02064, 0.00858, 0.00271, 0.00051,
      0.02535, 0.01175, 0.00462, 0.00149, 0.0002,
      0.02500, 0.06000, 0.09000, 0.09500, 0.11000, 0.11000,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=4)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_15(self):
    logger.info('\nTest #15.\nComponents: C1 C2 C3 NC4 FC6 N2')
    P = 1e6
    T = 160.
    yi = np_array([0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140])
    Pci = np_array([4.599, 4.872, 4.248, 3.796, 3.370, 3.025, 3.400]) * 1e6
    Tci = np_array([190.56, 305.32, 369.83, 425.12, 469.70, 507.60, 126.20])
    wi = np_array([0.0115, 0.0995, 0.1523, 0.2002, 0.2515, 0.3013, 0.0377])
    mwi = np_array([0.0160, 0.0301, 0.0441, 0.0581, 0.0722, 0.0860, 0.0280])
    dij = np_array([
      0.00269,
      0.00854, 0.00166,
      0.01475, 0.00491, 0.00087,
      0.02064, 0.00858, 0.00271, 0.00051,
      0.02535, 0.01175, 0.00462, 0.00149, 0.0002,
      0.02500, 0.06000, 0.09000, 0.09500, 0.11000, 0.11000,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=5)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_16(self):
    logger.info('\nTest #16.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C16 C17-C29 C30+')
    P = 8.7625e6
    T = 316.48
    yi = np_array([
      0.590516, 0.028933, 0.072729, 0.081162, 0.131012, 0.064671, 0.030979,
    ])
    Pci = np_array([73.36, 45.99, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np_array([304.20, 166.67, 338.81, 466.12, 611.11, 777.78, 972.22])
    wi = np_array([0.225, 0.008, 0.126, 0.244, 0.639, 1.000, 1.281])
    mwi = np_array([0.044, 0.016, 0.037, 0.072, 0.161, 0.312, 0.495])
    dij = np_array([
      0.05,
      0.05, 0.00853,
      0.05, 0.02064, 0.00271,
      0.09, 0.05428, 0.02079, 0.00863,
      0.09, 0.09301, 0.04829, 0.02887, 0.00615,
      0.09, 0.12546, 0.07402, 0.05002, 0.01794, 0.00316,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=15)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_17(self):
    logger.info('\nTest #17.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C16 C17-C29 C30+')
    P = 8.7625e6
    T = 316.48
    yi = np_array([
      0.590516, 0.028933, 0.072729, 0.081162, 0.131012, 0.064671, 0.030979,
    ])
    Pci = np_array([73.36, 45.99, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np_array([304.20, 166.67, 338.81, 466.12, 611.11, 777.78, 972.22])
    wi = np_array([0.225, 0.008, 0.126, 0.244, 0.639, 1.000, 1.281])
    mwi = np_array([0.044, 0.016, 0.037, 0.072, 0.161, 0.312, 0.495])
    dij = np_array([
      0.05,
      0.05, 0.00853,
      0.05, 0.02064, 0.00271,
      0.09, 0.05428, 0.02079, 0.00863,
      0.09, 0.09301, 0.04829, 0.02887, 0.00615,
      0.09, 0.12546, 0.07402, 0.05002, 0.01794, 0.00316,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=10)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_18(self):
    logger.info('\nTest #18.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C16 C17-C29 C30+')
    P = 8.7625e6
    T = 316.48
    yi = np_array([
      0.590516, 0.028933, 0.072729, 0.081162, 0.131012, 0.064671, 0.030979,
    ])
    Pci = np_array([73.36, 45.99, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np_array([304.20, 166.67, 338.81, 466.12, 611.11, 777.78, 972.22])
    wi = np_array([0.225, 0.008, 0.126, 0.244, 0.639, 1.000, 1.281])
    mwi = np_array([0.044, 0.016, 0.037, 0.072, 0.161, 0.312, 0.495])
    dij = np_array([
      0.05,
      0.05, 0.00853,
      0.05, 0.02064, 0.00271,
      0.09, 0.05428, 0.02079, 0.00863,
      0.09, 0.09301, 0.04829, 0.02887, 0.00615,
      0.09, 0.12546, 0.07402, 0.05002, 0.01794, 0.00316,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=16)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_19(self):
    logger.info('\nTest #19.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C14 C15-C24 C25+')
    P = 58948692.3
    T = 373.15
    yi = np_array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np_array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np_array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np_array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np_array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    dij = np_array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=44)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_20(self):
    logger.info('\nTest #20.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C14 C15-C24 C25+')
    P = 58948692.3
    T = 373.15
    yi = np_array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np_array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np_array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np_array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np_array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    dij = np_array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=15)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_21(self):
    logger.info('\nTest #21.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C14 C15-C24 C25+')
    P = 58948692.3
    T = 373.15
    yi = np_array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np_array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np_array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np_array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np_array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    dij = np_array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=53)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_22(self):
    logger.info('\nTest #22.\nComponents: H2S CO2 C1 C2 C3')
    P = 10781100.0
    T = 233.15
    yi = np_array([0.26, 0.04, 0.66, 0.03, 0.01])
    Pci = np_array([89.37, 73.76, 46.00, 48.84, 42.46]) * 1e5
    Tci = np_array([373.2, 304.2, 190.6, 305.4, 369.8])
    mwi = np_array([34.08, 44.01, 16.043, 30.07, 44.097]) / 1e3
    wi = np_array([0.117, 0.225, 0.008, 0.098, 0.152])
    dij = np_array([
      0.135,
      0.070, 0.105,
      0.085, 0.130, 0.005,
      0.080, 0.125, 0.010, 0.005,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=25)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_23(self):
    logger.info('\nTest #23.\nComponents: H2S CO2 C1 C2 C3')
    P = 10781100.0
    T = 233.15
    yi = np_array([0.26, 0.04, 0.66, 0.03, 0.01])
    Pci = np_array([89.37, 73.76, 46.00, 48.84, 42.46]) * 1e5
    Tci = np_array([373.2, 304.2, 190.6, 305.4, 369.8])
    mwi = np_array([34.08, 44.01, 16.043, 30.07, 44.097]) / 1e3
    wi = np_array([0.117, 0.225, 0.008, 0.098, 0.152])
    dij = np_array([
      0.135,
      0.070, 0.105,
      0.085, 0.130, 0.005,
      0.080, 0.125, 0.010, 0.005,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=8)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_24(self):
    logger.info('\nTest #24.\nComponents: H2S CO2 C1 C2 C3')
    P = 10781100.0
    T = 233.15
    yi = np_array([0.26, 0.04, 0.66, 0.03, 0.01])
    Pci = np_array([89.37, 73.76, 46.00, 48.84, 42.46]) * 1e5
    Tci = np_array([373.2, 304.2, 190.6, 305.4, 369.8])
    mwi = np_array([34.08, 44.01, 16.043, 30.07, 44.097]) / 1e3
    wi = np_array([0.117, 0.225, 0.008, 0.098, 0.152])
    dij = np_array([
      0.135,
      0.070, 0.105,
      0.085, 0.130, 0.005,
      0.080, 0.125, 0.010, 0.005,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=25)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_25(self):
    logger.info('\nTest #25.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C14 C15-C24 C25+')
    P = 64843561.5
    T = 373.15
    yi = np_array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np_array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np_array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np_array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np_array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    dij = np_array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=23)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_26(self):
    logger.info('\nTest #26.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C14 C15-C24 C25+')
    P = 64843561.5
    T = 373.15
    yi = np_array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np_array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np_array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np_array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np_array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    dij = np_array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=8)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_27(self):
    logger.info('\nTest #27.'
                '\nComponents: CO2 C1 C2-C3 C4-C6 C7-C14 C15-C24 C25+')
    P = 64843561.5
    T = 373.15
    yi = np_array([0.6673, 0.0958, 0.0354, 0.0445, 0.0859, 0.0447, 0.0264])
    Pci = np_array([73.76, 46.00, 45.05, 33.50, 24.24, 18.03, 17.26]) * 1e5
    Tci = np_array([304.20, 190.60, 343.64, 466.41, 603.07, 733.79, 923.20])
    mwi = np_array([44.01, 16.04, 38.40, 72.82, 135.82, 257.75, 479.95]) / 1e3
    wi = np_array([0.225, 0.008, 0.130, 0.244, 0.600, 0.903, 1.229])
    dij = np_array([
      0.12,
      0.12, 0.0051,
      0.12, 0.0207, 0.0053,
      0.12, 0.0405, 0.0174, 0.0035,
      0.12, 0.0611, 0.0321, 0.0117, 0.0024,
      0.12, 0.0693, 0.0384, 0.0156, 0.0044, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=23)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_28(self):
    logger.info('\nTest #28.\nComponents: '
                'CO2 C1 C2 C3 nC4 C5-C7 C8-C10 C11-C14 C15-C20 C21-C28 C29+')
    P = 23.3e6
    T = 373.15
    yi = np_array([0.55, 0.1323, 0.0459, 0.0376, 0.0149, 0.0542, 0.0711,
                   0.0370, 0.0238, 0.0124, 0.0208])
    Pci = np_array([73.82, 45.40, 48.20, 41.90, 37.50, 28.82, 23.74, 18.59,
                    14.80, 11.95, 8.52]) * 1e5
    Tci = np_array([304.21, 190.60, 305.40, 369.80, 425.20, 516.67, 590.00,
                    668.61, 745.78, 812.67, 914.89])
    mwi = np_array([44.0, 16.0, 30.1, 44.1, 58.1, 89.9, 125.7, 174.4, 240.3,
                    336.1, 536.7]) / 1e3
    wi = np_array([0.225, 0.008, 0.098, 0.152, 0.193, 0.265, 0.364, 0.499,
                   0.661, 0.877, 1.279])
    dij = np_array([
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
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=36)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_29(self):
    logger.info('\nTest #29.\nComponents: '
                'CO2 C1 C2 C3 nC4 C5-C7 C8-C10 C11-C14 C15-C20 C21-C28 C29+')
    P = 23.3e6
    T = 373.15
    yi = np_array([0.55, 0.1323, 0.0459, 0.0376, 0.0149, 0.0542, 0.0711,
                   0.0370, 0.0238, 0.0124, 0.0208])
    Pci = np_array([73.82, 45.40, 48.20, 41.90, 37.50, 28.82, 23.74, 18.59,
                    14.80, 11.95, 8.52]) * 1e5
    Tci = np_array([304.21, 190.60, 305.40, 369.80, 425.20, 516.67, 590.00,
                    668.61, 745.78, 812.67, 914.89])
    mwi = np_array([44.0, 16.0, 30.1, 44.1, 58.1, 89.9, 125.7, 174.4, 240.3,
                    336.1, 536.7]) / 1e3
    wi = np_array([0.225, 0.008, 0.098, 0.152, 0.193, 0.265, 0.364, 0.499,
                   0.661, 0.877, 1.279])
    dij = np_array([
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
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=17)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_30(self):
    logger.info('\nTest #30.\nComponents: '
                'CO2 C1 C2 C3 nC4 C5-C7 C8-C10 C11-C14 C15-C20 C21-C28 C29+')
    P = 23.3e6
    T = 373.15
    yi = np_array([0.55, 0.1323, 0.0459, 0.0376, 0.0149, 0.0542, 0.0711,
                   0.0370, 0.0238, 0.0124, 0.0208])
    Pci = np_array([73.82, 45.40, 48.20, 41.90, 37.50, 28.82, 23.74, 18.59,
                    14.80, 11.95, 8.52]) * 1e5
    Tci = np_array([304.21, 190.60, 305.40, 369.80, 425.20, 516.67, 590.00,
                    668.61, 745.78, 812.67, 914.89])
    mwi = np_array([44.0, 16.0, 30.1, 44.1, 58.1, 89.9, 125.7, 174.4, 240.3,
                    336.1, 536.7]) / 1e3
    wi = np_array([0.225, 0.008, 0.098, 0.152, 0.193, 0.265, 0.364, 0.499,
                   0.661, 0.877, 1.279])
    dij = np_array([
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
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=47)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_31(self):
    logger.info('\nTest #31.\nComponents: CO2 C1 C2 C3 C4 C5 C6 C7 C8 C10')
    P = 5e6
    T = 0. + 273.15
    yi = np_array([0.0001, 0.3499, 0.0300, 0.0400, 0.0600, 0.0400, 0.0300,
                   0.0500, 0.0500, 0.3000, 0.0500])
    Pci = np_array([73.84, 46.04, 48.84, 42.57, 37.46, 32.77, 29.72, 27.37,
                    25.10, 22.06, 15.86]) * 1e5
    Tci = np_array([304.04, 190.59, 305.21, 369.71, 419.04, 458.98, 507.54,
                    540.32, 568.93, 615.15, 694.82])
    mwi = np_array([44.01, 16.04, 30.07, 44.10, 58.12, 72.15, 86.18, 100.20,
                    114.23, 142.29, 198.39]) / 1e3
    wi = np_array([0.225, 0.010, 0.099, 0.152, 0.187, 0.252, 0.296, 0.351,
                   0.394, 0.491, 0.755])
    dij = np_array([
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
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=7)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_32(self):
    logger.info('\nTest #32.\nComponents: CO2 C1 C2 C3 C4 C5 C6 C7 C8 C10')
    P = 5e6
    T = 0. + 273.15
    yi = np_array([0.0001, 0.3499, 0.0300, 0.0400, 0.0600, 0.0400, 0.0300,
                   0.0500, 0.0500, 0.3000, 0.0500])
    Pci = np_array([73.84, 46.04, 48.84, 42.57, 37.46, 32.77, 29.72, 27.37,
                    25.10, 22.06, 15.86]) * 1e5
    Tci = np_array([304.04, 190.59, 305.21, 369.71, 419.04, 458.98, 507.54,
                    540.32, 568.93, 615.15, 694.82])
    mwi = np_array([44.01, 16.04, 30.07, 44.10, 58.12, 72.15, 86.18, 100.20,
                    114.23, 142.29, 198.39]) / 1e3
    wi = np_array([0.225, 0.010, 0.099, 0.152, 0.187, 0.252, 0.296, 0.351,
                   0.394, 0.491, 0.755])
    dij = np_array([
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
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=6)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_33(self):
    logger.info('\nTest #33.\nComponents: CO2 C1 C2 C3 C4 C5 C6 C7 C8 C10')
    P = 5e6
    T = 0. + 273.15
    yi = np_array([0.0001, 0.3499, 0.0300, 0.0400, 0.0600, 0.0400, 0.0300,
                   0.0500, 0.0500, 0.3000, 0.0500])
    Pci = np_array([73.84, 46.04, 48.84, 42.57, 37.46, 32.77, 29.72, 27.37,
                    25.10, 22.06, 15.86]) * 1e5
    Tci = np_array([304.04, 190.59, 305.21, 369.71, 419.04, 458.98, 507.54,
                    540.32, 568.93, 615.15, 694.82])
    mwi = np_array([44.01, 16.04, 30.07, 44.10, 58.12, 72.15, 86.18, 100.20,
                    114.23, 142.29, 198.39]) / 1e3
    wi = np_array([0.225, 0.010, 0.099, 0.152, 0.187, 0.252, 0.296, 0.351,
                   0.394, 0.491, 0.755])
    dij = np_array([
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
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=7)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_34(self):
    logger.info('\nTest #34.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 18e6
    T = 104.4 + 273.15
    yi = np_array([0.0091, 0.0016, 0.3647, 0.0967, 0.0695, 0.0144, 0.0393,
                   0.0144, 0.0141, 0.0433, 0.1320, 0.0757, 0.0510, 0.0315,
                   0.0427])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=10)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_35(self):
    logger.info('\nTest #35.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 18e6
    T = 104.4 + 273.15
    yi = np_array([0.0091, 0.0016, 0.3647, 0.0967, 0.0695, 0.0144, 0.0393,
                   0.0144, 0.0141, 0.0433, 0.1320, 0.0757, 0.0510, 0.0315,
                   0.0427])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=10)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_36(self):
    logger.info('\nTest #36.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 18e6
    T = 104.4 + 273.15
    yi = np_array([0.0091, 0.0016, 0.3647, 0.0967, 0.0695, 0.0144, 0.0393,
                   0.0144, 0.0141, 0.0433, 0.1320, 0.0757, 0.0510, 0.0315,
                   0.0427])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=12)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_37(self):
    logger.info('\nTest #37.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 101325.
    T = 20. + 273.15
    yi = np_array([0.0122802, 0.0045439, 0.7351892, 0.1175384, 0.0579690,
                   0.0091837, 0.0224902, 0.0062381, 0.0056632, 0.0125755,
                   0.0145719, 0.0011667, 0.0005780, 1.0668e-5, 1.3320e-6])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=3)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_38(self):
    logger.info('\nTest #38.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 101325.
    T = 20. + 273.15
    yi = np_array([0.0122802, 0.0045439, 0.7351892, 0.1175384, 0.0579690,
                   0.0091837, 0.0224902, 0.0062381, 0.0056632, 0.0125755,
                   0.0145719, 0.0011667, 0.0005780, 1.0668e-5, 1.3320e-6])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=3)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_39(self):
    logger.info('\nTest #39.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 101325.
    T = 20. + 273.15
    yi = np_array([0.0122802, 0.0045439, 0.7351892, 0.1175384, 0.0579690,
                   0.0091837, 0.0224902, 0.0062381, 0.0056632, 0.0125755,
                   0.0145719, 0.0011667, 0.0005780, 1.0668e-5, 1.3320e-6])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=3)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_40(self):
    logger.info('\nTest #40.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 13803e3
    T = 444.444 + 273.15
    yi = np_array([0.0091, 0.0016, 0.3647, 0.0967, 0.0695, 0.0144, 0.0393,
                   0.0144, 0.0141, 0.0433, 0.1320, 0.0757, 0.0510, 0.0315,
                   0.0427])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=27)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_41(self):
    logger.info('\nTest #41.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 13803e3
    T = 444.444 + 273.15
    yi = np_array([0.0091, 0.0016, 0.3647, 0.0967, 0.0695, 0.0144, 0.0393,
                   0.0144, 0.0141, 0.0433, 0.1320, 0.0757, 0.0510, 0.0315,
                   0.0427])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=13)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_42(self):
    logger.info('\nTest #42.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 13803e3
    T = 444.444 + 273.15
    yi = np_array([0.0091, 0.0016, 0.3647, 0.0967, 0.0695, 0.0144, 0.0393,
                   0.0144, 0.0141, 0.0433, 0.1320, 0.0757, 0.0510, 0.0315,
                   0.0427])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=30)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertFalse(state.kvji is None)
    pass

  def test_43(self):
    logger.info('\nTest #43.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 13803e3
    T = 444.444 + 273.15
    yi = np_array([
      0.009755196948358247, 0.0017485841684906926, 0.393700996975837500,
      0.102730821615165270, 0.0728694643842911900, 0.014932664187754175,
      0.040631323857286364, 0.0147261980149974370, 0.014396370505843558,
      0.043662804092823554, 0.1282837227803721000, 0.067997996620939850,
      0.045150142736511245, 0.0232206237239377700, 0.026193089387391122,
    ])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssnewt, maxiter=22)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_44(self):
    logger.info('\nTest #44.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 13803e3
    T = 444.444 + 273.15
    yi = np_array([
      0.009755196948358247, 0.0017485841684906926, 0.393700996975837500,
      0.102730821615165270, 0.0728694643842911900, 0.014932664187754175,
      0.040631323857286364, 0.0147261980149974370, 0.014396370505843558,
      0.043662804092823554, 0.1282837227803721000, 0.067997996620939850,
      0.045150142736511245, 0.0232206237239377700, 0.026193089387391122,
    ])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_qnssnewt, maxiter=13)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass

  def test_45(self):
    logger.info('\nTest #45.\nComponents: '
                'CO2 N2 C1 C2 C3 iC4 nC4 iC5 nC5 FC6 FC8 FC13 FC17 FC27 FC42')
    P = 13803e3
    T = 444.444 + 273.15
    yi = np_array([
      0.009755196948358247, 0.0017485841684906926, 0.393700996975837500,
      0.102730821615165270, 0.0728694643842911900, 0.014932664187754175,
      0.040631323857286364, 0.0147261980149974370, 0.014396370505843558,
      0.043662804092823554, 0.1282837227803721000, 0.067997996620939850,
      0.045150142736511245, 0.0232206237239377700, 0.026193089387391122,
    ])
    Pci = np_array([72.8, 33.5, 45.4, 48.2, 41.9, 36.0, 37.5, 33.4, 33.3,
                    32.46, 26.94, 18.25, 17.15, 10.118, 7.14]) * 101325.
    Tci = np_array([304.2, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.6, 507.5, 598.5, 718.6, 734.5, 872.53, 957.8])
    wi = np_array([0.225, 0.040, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.275, 0.391, 0.651, 0.684, 1.082, 1.330])
    mwi = np_array([44.010, 28.013, 16.043, 30.070, 44.097, 58.124, 58.124,
                    72.151, 72.151, 86., 121., 206., 222., 394., 539.]) / 1e3
    dij = np_array([
      -0.020,
       0.105, 0.025,
       0.130, 0.010, 0.003,
       0.125, 0.090, 0.009, 0.002,
       0.120, 0.095, 0.016, 0.005, 0.001,
       0.115, 0.095, 0.015, 0.005, 0.001, 0.000,
       0.115, 0.100, 0.021, 0.009, 0.003, 0.000, 0.001,
       0.115, 0.110, 0.021, 0.009, 0.003, 0.000, 0.001, 0.000,
       0.115, 0.110, 0.025, 0.012, 0.005, 0.001, 0.001, 0.000, 0.000,
       0.115, 0.110, 0.039, 0.022, 0.012, 0.006, 0.006, 0.003, 0.003, 0.002,
       0.115, 0.110, 0.067, 0.044, 0.029, 0.019, 0.020, 0.014, 0.015, 0.011,
       0.004,
       0.115, 0.110, 0.072, 0.048, 0.032, 0.022, 0.023, 0.017, 0.017, 0.013,
       0.005, 0.000,
       0.115, 0.110, 0.107, 0.079, 0.059, 0.045, 0.047, 0.038, 0.038, 0.032,
       0.020, 0.006, 0.004,
       0.115, 0.110, 0.133, 0.102, 0.080, 0.064, 0.066, 0.055, 0.055, 0.048,
       0.033, 0.014, 0.012, 0.002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, dij)
    solver = partial(_stabPT_ssbfgs, maxiter=25)
    state = stabtest.runPT(pr, P, T, yi, solver=solver)
    self.assertTrue(state.kvji is None)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
