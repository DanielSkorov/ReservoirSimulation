import sys

sys.path.append('../_src/')

import logging

logger = logging.getLogger('bound')
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
    P0 = 2e6
    T0 = 173.15
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
    env = env2pPT(pr)
    res = env.run(P0, T0, yi, 0.)
    self.assertTrue(res.success)
    if plotting:
      self.plot(res, -100., 140., 0., 20.)
    pass



if __name__ == '__main__':
  unittest.main(verbosity=0)
