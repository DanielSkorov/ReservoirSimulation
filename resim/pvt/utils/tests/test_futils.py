import sys
import logging
import unittest

from numpy import (
  allclose as np_allclose,
  asfortranarray as np_asfortranarray,
)

from numpy.linalg import (
  solve as np_lusolver,
)

from numpy.random import (
  uniform as np_uniform,
)

from resim.pvt.utils import (
  futils,
)


logger = logging.getLogger('utils')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


lusolver = futils.linalg.lusolver


class futils_linalg(unittest.TestCase):

  def test_01(self):
    A = np_uniform(size=(10, 10))
    b = np_uniform(size=(10,))
    x = np_lusolver(A, -b)
    fA = np_asfortranarray(A)
    fb = np_asfortranarray(b)
    fx, singular = lusolver(fA, -fb)
    self.assertTrue(np_allclose(x, fx) and not singular)
    pass


if __name__ == '__main__':
  unittest.main(verbosity=0)
