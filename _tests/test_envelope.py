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

plotting = True


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
                  psatkwargs=dict(method='newton',
                                  stabkwargs=dict(method='qnss-newton'),
                                  tol=1e-8, tol_tpd=1e-8),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=120, maxstep=0.25)
    self.assertTrue(res.succeed)
    if plotting:
      self.plot(res, -100., 140., 0., 20.)
    pass

  def test_02(self):
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
                  psatkwargs=dict(method='newton',
                                  stabkwargs=dict(method='qnss-newton'),
                                  tol=1e-8, tol_tpd=1e-8),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=385)
    self.assertTrue(res.succeed)
    if plotting:
      self.plot(res, -80., 10., 0., 30.)
    pass

  def test_03(self):
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
                  psatkwargs=dict(method='newton',
                                  stabkwargs=dict(method='qnss-newton'),
                                  tol=1e-8, tol_tpd=1e-8),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=147)
    self.assertTrue(res.succeed)
    if plotting:
      self.plot(res, -80., 10., 0., 30.)
    pass

  def test_04(self):
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
                  psatkwargs=dict(method='newton',
                                  stabkwargs=dict(method='qnss-newton'),
                                  tol=1e-8, tol_tpd=1e-8),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., sidx0=6, maxpoints=149, maxstep=0.25)
    self.assertTrue(res.succeed)
    if plotting:
      self.plot(res, -20., 500., 0., 40.)
    pass

  def test_05(self):
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
                  psatkwargs=dict(method='newton',
                                  stabkwargs=dict(method='qnss-newton'),
                                  tol=1e-8, tol_tpd=1e-8),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., sidx0=6, maxpoints=107, maxstep=0.25)
    self.assertTrue(res.succeed)
    if plotting:
      self.plot(res, -20., 500., 0., 100.)
    pass

  def test_06(self):
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
                  psatkwargs=dict(method='newton',
                                  stabkwargs=dict(method='qnss-newton'),
                                  tol=1e-8, tol_tpd=1e-8),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxpoints=105, maxstep=0.25)
    self.assertTrue(res.succeed)
    if plotting:
      self.plot(res, 0., 500., 0., 100.)
    pass

  def test_07(self):
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
                  psatkwargs=dict(method='newton-b',
                                  stabkwargs=dict(method='qnss-newton'),
                                  tol=1e-8, tol_tpd=1e-8),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., sidx0=6, maxpoints=102, maxstep=0.25)
    self.assertTrue(res.succeed)
    if plotting:
      self.plot(res, 0., 500., 0., 100.)
    pass

  def test_08(self):
    P0 = 23.3e6
    T0 = 373.15
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
    env = env2pPT(pr, Tmin=303.15,
                  psatkwargs=dict(method='newton-b',
                                  stabkwargs=dict(method='qnss-newton'),
                                  tol=1e-8, tol_tpd=1e-8),
                  flashkwargs=dict(method='qnss-newton', runstab=False,
                                   useprev=True, tol=1e-8))
    res = env.run(P0, T0, yi, 0., maxstep=0.25, maxpoints=135)
    self.assertTrue(res.succeed)
    if plotting:
      self.plot(res, 0., 500., 0., 100.)
    pass



if __name__ == '__main__':
  unittest.main(verbosity=0)
