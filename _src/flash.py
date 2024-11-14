import numpy as np

from functools import (
  partial,
)

from typing import (
  Callable,
)

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  EosptType,
)


class stabilityPT(object):
  """Stability test based on the Gibbs energy analysis.

  Checks the tangent-plane distance (TPD) in local minima of
  the Gibbs energy function.

  Arguments
  ---------
  eos : EosptType
    An initialized instance of a PT-based equation of state.
  """
  def __init__(
    self,
    eos: EosptType,
    eps: ScalarType = np.float64(1e-4),
    tol: ScalarType = np.float64(1e-6),
    Niter: int = 50,
  ) -> None:
    self.eos = eos
    self.eps = -eps
    self.tol = tol
    self.Niter = Niter
    pass

  def check_qnss(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[bool, MatrixType | None]:
    """QNSS-method for stability test

    Performs the quasi-Newton successive substitution (QNSS) method to
    find local minima of the Gibbs energy function from different
    initial guesses. Calculates the TPD value at found local minima.
    For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

    Arguments
    ---------
      P : numpy.float64
        Pressure of a mixture [Pa].

      T : numpy.float64
        Temperature of a mixture [K].

      yi : numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
        mole fractions of `(Nc,)` components.

    Returns a tuple of a boolean (True if a system is stable, False
    otherwise) and an array of k-values at a local minima that can be
    used as an initial guess for flash calculations.
    """
    # print(f'*** Stability Test ***\n{P = }\n{T = }\n{yi = }')
    kvji = self.eos.get_kvguess(P, T)
    hi = self.eos.get_lnphii(P, T, yi, False) + np.log(yi)
    plnphii = partial(self.eos.get_lnphii, P=P, T=T, check_input=False)
    pcondit = partial(self._condit_qnss, tol=self.tol, Niter=self.Niter)
    pupdate = partial(self._update_qnss, hi=hi, yi=yi, plnphii=plnphii)
    # print('Sarting the kv-loop...')
    kvi: VectorType
    for j, kvi in enumerate(kvji):
      # print(f'\n\tThe kv-loop iteration number = {j}')
      # print(f'\tInitial k-values = {kvi}')
      ni = kvi * yi
      gi = np.log(ni) + plnphii(yi=ni/ni.sum()) - hi
      # print(f'\t{gi = }')
      # print(f'\tTPD = {-np.log(ni.sum())}')
      carry = (1, kvi, gi, np.float64(1.))
      # print('\tStarting the solution loop...')
      while pcondit(carry):
        carry = pupdate(carry)
      i, kvi, gi, _ = carry
      ni = kvi * yi
      TPD = -np.log(ni.sum())
      if (TPD < self.eps) & (i < self.Niter):
        # print(f'The final kv-loop iteration number = {j}')
        # print(f'{TPD = }')
        # print(f'The one-phase state is stable: False')
        # print('*** End of Stability Test ***')
        xi = ni / ni.sum()
        return False, np.vstack([xi / yi, yi / xi])
    else:
      # print(f'The final kv-loop iteration number = {j}')
      # print(f'{TPD = }')
      # print(f'The one-phase state is stable: True')
      # print('*** End of Stability Test ***')
      return True, None

  @staticmethod
  def _condit_qnss(
    carry: tuple[int, VectorType, VectorType, ScalarType],
    tol: ScalarType,
    Niter: int,
  ) -> np.bool:
    i, ki, gi, _ = carry
    return (i < Niter) & (np.linalg.norm(gi) > tol)

  @staticmethod
  def _update_qnss(
    carry: tuple[int, VectorType, VectorType, ScalarType],
    hi: VectorType,
    yi: VectorType,
    plnphii: Callable[[VectorType], VectorType],
  ) -> tuple[int, VectorType, VectorType, ScalarType]:
    i, kvi, gi_, lmbd = carry
    # print(f'\n\t\tIteration #{i}')
    dlnkvi = -lmbd * gi_
    max_dlnkvi = np.abs(dlnkvi).max()
    if max_dlnkvi > 6.:
      relax = 6. / max_dlnkvi
      lmbd *= relax
      dlnkvi *= relax
    # print(f'\t\t{lmbd = }')
    # print(f'\t\t{dlnkvi = }')
    kvi *= np.exp(dlnkvi)
    # print(f'\t\t{kvi = }')
    ni = kvi * yi
    gi = np.log(ni) + plnphii(yi=ni/ni.sum()) - hi
    # print(f'\t\t{gi = }')
    # print(f'\t\tTPD = {-np.log(ni.sum())}')
    lmbd *= np.abs(dlnkvi.dot(gi_) / dlnkvi.dot(gi - gi_))
    if lmbd > 30.:
      lmbd = np.float64(30.)
    return i + 1, kvi, gi, lmbd

