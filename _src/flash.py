import numpy as np
import numpy.typing as npt

from functools import (
  partial,
)

from typing import (
  Callable,
)

from custom_types import (
  EOS_PT,
)


class stabilityPT(object):
  def __init__(
    self,
    eos: EOS_PT,
    eps: np.float64 = np.float64(1e-4),
    tol: np.float64 = np.float64(1e-6),
    Niter: int = 50,
  ) -> None:
    self.eos = eos
    self.eps = -eps
    self.tol = tol
    self.Niter = Niter
    pass

  def get_stability(
    self,
    P: np.float64,
    T: np.float64,
    yi: np.float64,
  ) -> tuple[bool, npt.NDArray[np.float64] | None]:
    # print(f'*** Stability Test ***\n{P = }\n{T = }\n{yi = }')
    kvji = self.eos.get_kvguess(P, T)
    hi = self.eos.get_lnphii(P, T, yi) + np.log(yi)
    plnphii = partial(self.eos.get_lnphii, P=P, T=T, check_input=False)
    pcondit = partial(self._condit, tol=self.tol, Niter=self.Niter)
    pupdate = partial(self._update, hi=hi, yi=yi, plnphii=plnphii)
    # print('Sarting the kv-loop...')
    for j, kvi in enumerate(kvji):
      # print(f'\n\tThe kv-loop iteration number = {j}')
      # print(f'\tInitial k-values = {kvi}')
      ni = kvi * yi
      gi = np.log(ni) + plnphii(yi=ni/ni.sum()) - hi
      # print(f'\t{gi = }')
      # print(f'\tTPD = {-np.log(ni.sum())}')
      carry = (1, kvi, gi, 1.)
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
        kvi_guess = np.vstack([xi / yi, yi / xi])
        return False, kvi_guess
    else:
      # print(f'The final kv-loop iteration number = {j}')
      # print(f'{TPD = }')
      # print(f'The one-phase state is stable: True')
      # print('*** End of Stability Test ***')
      return True, None

  @staticmethod
  def _condit(
    carry: tuple[
      int,
      npt.NDArray[np.float64],
      npt.NDArray[np.float64],
      np.float64,
    ],
    tol: np.float64,
    Niter: int,
  ) -> bool:
    i, ki, gi, _ = carry
    return (i < Niter) & (np.linalg.norm(gi) > tol)

  @staticmethod
  def _update(
    carry: tuple[
      int,
      npt.NDArray[np.float64],
      npt.NDArray[np.float64],
      np.float64,
    ],
    hi: npt.NDArray[np.float64],
    yi: npt.NDArray[np.float64],
    plnphii: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
  ) -> tuple[
    int,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    np.float64,
  ]:
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
      lmbd = 30.
    return i + 1, kvi, gi, lmbd

