import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('bound')
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

from boundary import (
  env2pPT,
)

from matplotlib import (
  pyplot as plt,
)

plotting = False


class env2p(unittest.TestCase):

  @staticmethod
  def plot(res, xmin, xmax, ymin, ymax):
    fig, ax = plt.subplots(1, 1, figsize=(6., 4.), tight_layout=True)
    ax.plot(res.Tk-273.15, res.Pk/1e6, lw=2., c='teal', zorder=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Temperature, C')
    ax.set_ylabel('Pressure, MPa')
    ax.grid(zorder=1)
    plt.show()
    pass

  def test_01(self):
    print('=== test_01 ===')
    P0 = 3.5e6
    T0 = 193.15
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
    env = env2pPT(pr, Tmin=193.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=118)
    if plotting:
      self.plot(res, -100., 140., 0., 20.)
    self.assertTrue(res.succeed)
    pass

  def test_02(self):
    print('=== test_02 ===')
    P0 = 8.1e6
    T0 = 233.15
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
    env = env2pPT(pr, Tmin=193.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=104, maxstep=0.1)
    if plotting:
      self.plot(res, -80., 10., 0., 30.)
    self.assertTrue(res.succeed)
    pass

  def test_03(self):
    print('=== test_03 ===')
    P0 = 7.3e6
    T0 = 200.
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
    env = env2pPT(pr, Tmin=193.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=33, sidx0=5)
    if plotting:
      self.plot(res, -80., 10., 0., 30.)
    self.assertTrue(res.succeed)
    pass

  def test_04(self):
    print('=== test_04 ===')
    P0 = 20e6
    T0 = 263.15
    yi = np.array([0.6077, 0.0277, 0.0697, 0.0778, 0.1255, 0.0620, 0.0296])
    Pci = np.array([73.77, 46.00, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np.array([304.2, 190.6, 338.8, 466.1, 611.1, 777.8, 972.2])
    mwi = np.array([44.01, 16.04, 36.01, 70.52, 147.18, 301.48, 562.81]) / 1e3
    wi = np.array([0.2250, 0.0080, 0.1260, 0.2439, 0.6386, 1.0002, 1.2812])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.05,
      0.05, 0.0052,
      0.05, 0.0214, 0.0057,
      0.09, 0.0494, 0.0234, 0.0062,
      0.09, 0.0717, 0.0401, 0.0162, 0.0024,
      0.09, 0.0783, 0.0452, 0.0196, 0.0038, 0.0002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    env = env2pPT(pr, Tmin=258.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=140)
    if plotting:
      self.plot(res, -20., 500., 0., 40.)
    self.assertTrue(res.succeed)
    pass

  def test_05(self):
    print('=== test_05 ===')
    P0 = 36e6
    T0 = 348.15
    yi = np.array([0.7548, 0.0173, 0.0436, 0.0486, 0.0785, 0.0387, 0.0185])
    Pci = np.array([73.77, 46.00, 45.53, 33.68, 20.95, 15.88, 15.84]) * 1e5
    Tci = np.array([304.2, 190.6, 338.8, 466.1, 611.1, 777.8, 972.2])
    mwi = np.array([44.01, 16.04, 36.01, 70.52, 147.18, 301.48, 562.81]) / 1e3
    wi = np.array([0.2250, 0.0080, 0.1260, 0.2439, 0.6386, 1.0002, 1.2812])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.05,
      0.05, 0.0052,
      0.05, 0.0214, 0.0057,
      0.09, 0.0494, 0.0234, 0.0062,
      0.09, 0.0717, 0.0401, 0.0162, 0.0024,
      0.09, 0.0783, 0.0452, 0.0196, 0.0038, 0.0002,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    env = env2pPT(pr, Tmin=293.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., sidx0=6, maxpoints=104)
    if plotting:
      self.plot(res, -20., 500., 0., 100.)
    self.assertTrue(res.succeed)
    pass

  def test_06(self):
    print('=== test_06 ===')
    P0 = 25e6
    T0 = 373.15
    yi = np.array([0.7051, 0.0526, 0.0673, 0.0502, 0.0727, 0.0365, 0.0156])
    Pci = np.array([73.76, 46.00, 44.69, 34.18, 21.87, 16.04, 15.21]) * 1e5
    Tci = np.array([304.20, 174.44, 347.26, 459.74, 595.14, 729.98, 910.18])
    mwi = np.array([44.01, 16.04, 37.91, 68.67, 135.09, 261.10, 479.70]) / 1e3
    wi = np.array([0.2250, 0.0080, 0.1331, 0.2358, 0.5977, 0.9118, 1.2444])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.085,
      0.085, 0.0055,
      0.085, 0.0195, 0.0044,
      0.104, 0.0451, 0.0199, 0.0057,
      0.104, 0.0677, 0.0363, 0.0158, 0.0026,
      0.104, 0.0760, 0.0427, 0.0203, 0.0046, 0.0003,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    env = env2pPT(pr, Tmin=295.6,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., sidx0=6, maxpoints=104)
    if plotting:
      self.plot(res, 0., 500., 0., 100.)
    self.assertTrue(res.succeed)
    pass

  def test_07(self):
    print('=== test_07 ===')
    P0 = 25e6
    T0 = 373.15
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
    env = env2pPT(pr, Tmin=328.15,
                  psatkwargs=dict(method='newton-b', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=101)
    if plotting:
      self.plot(res, 0., 500., 0., 100.)
    self.assertTrue(res.succeed)
    pass

  def test_08(self):
    print('=== test_08 ===')
    P0 = 23.3e6
    T0 = 373.15
    yi = np.array([0.5500, 0.1323, 0.0459, 0.0376, 0.0149, 0.0542, 0.0711,
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
    env = env2pPT(pr, Tmin=303.15,
                  psatkwargs=dict(method='newton-b', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=129)
    if plotting:
      self.plot(res, 0., 500., 0., 100.)
    self.assertTrue(res.succeed)
    pass

  def test_09(self):
    print('=== test_09 ===')
    P0 = 30e6
    T0 = 373.15
    yi = np.array([0.7000, 0.0882, 0.0306, 0.0251, 0.0099, 0.0361, 0.0474,
                   0.0247, 0.0158, 0.0083, 0.0139])
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
    env = env2pPT(pr, Tmin=323.15,
                  psatkwargs=dict(method='newton-b', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., sidx0=10, maxpoints=110)
    if plotting:
      self.plot(res, 0., 500., 0., 100.)
    self.assertTrue(res.succeed)
    pass

  def test_10(self):
    print('=== test_10 ===')
    P0 = 5e6
    T0 = 273.15
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
    env = env2pPT(pr, Tmin=123.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=322)
    if plotting:
      self.plot(res, -150., 350., 0., 16.)
    self.assertTrue(res.succeed)
    pass

  def test_11(self):
    print('=== test_11 ===')
    P0 = 5e6
    T0 = 273.15
    yi = np.array([0.200, 0.280, 0.024, 0.032, 0.048, 0.032, 0.024, 0.040,
                   0.040, 0.240, 0.040])
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
    env = env2pPT(pr, Tmin=123.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=321)
    if plotting:
      self.plot(res, -150., 350., 0., 16.)
    self.assertTrue(res.succeed)
    pass

  def test_12(self):
    print('=== test_12 ===')
    P0 = 10e6
    T0 = 313.15
    yi = np.array([0.800, 0.070, 0.006, 0.008, 0.012, 0.008, 0.006, 0.010,
                   0.010, 0.060, 0.010])
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
    env = env2pPT(pr, Tmin=123.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., sidx0=11, maxpoints=118)
    if plotting:
      self.plot(res, 0., 240., 0., 100.)
    self.assertTrue(res.succeed)
    pass

  def test_13(self):
    print('=== test_13 ===')
    P0 = 25e6
    T0 = 273.15
    yi = np.array([0.0046, 0.0336, 0.6236, 0.0890, 0.0531, 0.0092, 0.0208,
                   0.0073, 0.0085, 0.0105, 0.0185, 0.0175, 0.0140, 0.0464,
                   0.0291, 0.0143])
    Pci = np.array([33.94, 73.76, 46.00, 48.84, 42.46, 36.48, 38.00, 33.84,
                    33.74, 29.69, 31.95, 29.44, 26.08, 20.82, 15.80,
                    13.06]) * 1e5
    Tci = np.array([126.2, 304.3, 190.6, 305.5, 369.9, 408.2, 425.3, 460.5,
                    469.7, 507.5, 532.9, 553.1, 575.8, 635.0, 732.1, 908.1])
    mwi = np.array([28.0, 44.0, 16.0, 30.1, 44.1, 58.1, 58.1, 72.2, 72.2,
                    86.2, 95.0, 106.0, 121.0, 165.1, 265.1, 488.8]) / 1e3
    wi = np.array([0.040, 0.225, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227,
                   0.251, 0.296, 0.465, 0.497, 0.540, 0.669, 0.919, 1.246])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0.])
    dij = np.array([
      0.032,
      0.028, 0.120,
      0.041, 0.120, 0.,
      0.076, 0.120, 0., 0.,
      0.094, 0.120, 0., 0., 0.,
      0.070, 0.120, 0., 0., 0., 0.,
      0.087, 0.120, 0., 0., 0., 0., 0.,
      0.088, 0.120, 0., 0., 0., 0., 0., 0.,
      0.080, 0.120, 0., 0., 0., 0., 0., 0., 0.,
      0.080, 0.100, 0., 0., 0., 0., 0., 0., 0., 0.,
      0.080, 0.100, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0.080, 0.100, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0.080, 0.100, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0.080, 0.100, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0.080, 0.100, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    env = env2pPT(pr, Tmin=193.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=131)
    if plotting:
      self.plot(res, -100., 450., 0., 45.)
    self.assertTrue(res.succeed)
    pass

  def test_14(self):
    print('=== test_14 ===')
    P0 = 10e6
    T0 = 253.15
    yi = np.array([0.0987, 0.4023, 0.4990])
    Pci = np.array([72.8, 88.2, 45.4]) * 101325.
    Tci = np.array([304.2, 373.2, 190.6])
    mwi = np.array([44.010, 34.080, 16.043]) / 1e3
    wi = np.array([0.225, 0.100, 0.008])
    vsi = np.array([0., 0., 0.])
    dij = np.array([
      0.0974,
      0.1100, 0.0690,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    env = env2pPT(pr, Tmin=213.15, miniter=1,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton'),),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=255, maxstep=0.025)
    if plotting:
      self.plot(res, -70., 30., 0., 15.)
    self.assertTrue(res.succeed)
    pass

  def test_15(self):
    print('=== test_15 ===')
    P0 = 12e6
    T0 = 273.15
    yi = np.array([0.08798, 0.37452, 0.08064, 0.07376, 0.02630, 0.01496,
                   0.01744, 0.14328, 0.07280, 0.04840, 0.03576, 0.02416])
    Pci = np.array([73.74, 45.92, 48.75, 42.38, 37.93, 33.68, 29.64, 28.83,
                    19.32, 16.59, 15.27, 14.67]) * 1e5
    Tci = np.array([304.1, 190.6, 305.4, 369.8, 425.2, 469.6, 507.4, 616.2,
                    698.9, 770.4, 853.1, 1001.2])
    mwi = np.array([44.01, 16.04, 30.07, 44.10, 58.12, 72.15, 86.17, 118.30,
                    172.00, 236.00, 338.80, 451.00]) / 1e3
    wi = np.array([0.228, 0.008, 0.098, 0.152, 0.193, 0.251, 0.296, 0.454,
                   0.787, 1.048, 1.276, 1.299])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    dij = np.array([
      0.12,
      0.15, 0.,
      0.15, 0., 0.,
      0.15, 0., 0., 0.,
      0.15, 0., 0., 0., 0.,
      0.15, 0., 0., 0., 0., 0,
      0.15, 0., 0., 0., 0., 0, 0.,
      0.15, 0., 0., 0., 0., 0, 0., 0.,
      0.15, 0., 0., 0., 0., 0, 0., 0., 0.,
      0.15, 0., 0., 0., 0., 0, 0., 0., 0., 0.,
      0.15, 0., 0., 0., 0., 0, 0., 0., 0., 0., 0.,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    env = env2pPT(pr, Tmin=73.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=652, maxstep=0.5)
    if plotting:
      self.plot(res, -200., 550., 0., 35.)
    self.assertTrue(res.succeed)
    pass

  def test_16(self):
    print('=== test_16 ===')
    P0 = 20e6
    T0 = 273.15
    yi = np.array([0.00331450, 0.00945166, 0.55073030, 0.05606356,
                   0.03620950, 0.00759774, 0.01825016, 0.00788670,
                   0.01129328, 0.01700266, 0.03945694, 0.07437480,
                   0.07904020, 0.05889600, 0.03043200])
    Pci = np.array([73.815, 33.991, 46.043, 48.801, 42.492, 36.480, 37.969,
                    33.812, 33.688, 30.123, 31.776, 26.190, 19.637, 14.519,
                    11.066]) * 1e5
    Tci = np.array([304.2, 126.3, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4,
                    469.7, 507.4, 563.2, 638.3, 736.5, 837.0, 936.9])
    mwi = np.array([44.01, 28.01, 16.04, 30.07, 44.10, 58.12, 58.12, 72.15,
                    72.15, 86.18, 98.55, 135.84, 206.65, 319.83,
                    500.00]) / 1e3
    wi = np.array([0.2310, 0.0450, 0.0115, 0.0908, 0.1454, 0.1756, 0.1928,
                   0.2273, 0.2510, 0.2957, 0.2753, 0.3761, 0.5552, 0.8021,
                   1.108])
    vsi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0.])
    dij = np.array([
      0.000,
      0.105, 0.025,
      0.130, 0.010, 0.000,
      0.125, 0.090, 0.000, 0.,
      0.120, 0.095, 0.000, 0., 0.,
      0.115, 0.095, 0.000, 0., 0., 0.,
      0.115, 0.100, 0.000, 0., 0., 0., 0.,
      0.115, 0.110, 0.000, 0., 0., 0., 0., 0.,
      0.115, 0.110, 0.000, 0., 0., 0., 0., 0., 0.,
      0.115, 0.110, 0.020, 0., 0., 0., 0., 0., 0., 0.,
      0.115, 0.110, 0.028, 0., 0., 0., 0., 0., 0., 0., 0.,
      0.115, 0.110, 0.040, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      0.115, 0.110, 0.052, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0,
      0.115, 0.110, 0.064, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    ])
    pr = pr78(Pci, Tci, wi, mwi, vsi, dij)
    env = env2pPT(pr, Tmin=123.15,
                  psatkwargs=dict(method='newton', tol=1e-8, tol_tpd=1e-8,
                                  stabkwargs=dict(method='qnss-newton')),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=162)
    if plotting:
      self.plot(res, -150., 500., 0., 100.)
    self.assertTrue(res.succeed)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
