import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('bound')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
  '%(process)d:%(name)s:%(levelname)s: %(message)s'
)
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
    env = env2pPT(pr, stabkwargs=dict(method='qnss'))
    res = env.run(P0, T0, yi, 0., maxpoints=202)
    self.assertTrue(res.success)
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
    env = env2pPT(pr, Tmin=193.15, stabkwargs=dict(method='qnss'))
    res = env.run(P0, T0, yi, 0., improve_P0=True, sidx0=4, maxpoints=128)
    self.assertTrue(res.success)
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
    env = env2pPT(pr, Tmin=193.15, stabkwargs=dict(method='qnss'))
    res = env.run(P0, T0, yi, 0., improve_P0=True, maxpoints=69)
    self.assertTrue(res.success)
    if plotting:
      self.plot(res, -80., 10., 0., 30.)
    pass

  def test_04(self):
    P0 = 15e6
    T0 = 300.
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
    env = env2pPT(pr, Tmin=258.15, stabkwargs=dict(method='qnss'), maxiter=10)
    res = env.run(P0, T0, yi, 0., improve_P0=True, sidx0=5, maxpoints=277,
                  cfmax=0.05)
    self.assertTrue(res.success)
    if plotting:
      self.plot(res, -20., 500., 0., 40.)
    pass



if __name__ == '__main__':
  unittest.main(verbosity=0)
