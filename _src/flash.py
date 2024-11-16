import logging

from functools import (
  partial,
)

import numpy as np

from stability import (
  stabilityPT,
)

from rr import (
  solve2p_FGH,
)

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  EosptType,
)


logger = logging.getLogger('flash')


class flash2pPT(object):
  """Performs ...

  Arguments
  ---------

  Returns ....

  Raises ...
  """
  def __init__(
    self,
    eos: EosptType,
    tol: ScalarType = np.float64(1e-6),
    Niter: int = 50,
    skip_stab: bool = False,
  ) -> None:
    self.eos = eos
    self.tol = tol
    self.Niter = Niter
    self.skip_stab = skip_stab
    self.stability = stabilityPT(eos)
    pass

  def run(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
    method: str = 'qnss',
  ) -> tuple[VectorType, MatrixType, VectorType]:
    if method == 'qnss':
      return self._run_qnss(P, T, yi)
    elif method == 'bfgs':
      return self._run_bfgs(P, T, yi)
    else:
      raise ValueError(f'The unknown method: {method}.')

  def _run_bfgs(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, MatrixType, VectorType]:
    raise NotImplementedError(
      'The BFGS-method for stability testing is not implemented yet.'
    )
    return (
      np.array([0., 0.]),
      np.zeros(shape=(self.eos.Nc, self.eos.Nc), dtype=yi.dtype),
      np.array([0., 0.]),
    )

  def _run_qnss(
    self,
    P: ScalarType,
    T: ScalarType,
    yi: VectorType,
  ) -> tuple[VectorType, MatrixType, VectorType]:
    logger.debug(
      'Flash Calculation (QNSS-method)'
      '\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
      P, T, yi,
    )
    if self.skip_stab:
      kvji = self.eos.get_kvguess(P, T)
    else:
      stab, kvji, Z = self.stability._run_qnss(P, T, yi)
      if stab:
        return (
          np.array([1., 0.]),
          np.vstack([yi, np.zeros_like(yi)]),
          np.array([Z, 0.]),
        )
    plnphii = partial(self.eos.get_lnphii_Z, P=P, T=T)
    kvik: VectorType
    for i, kvik in enumerate(kvji):
      logger.debug('The kv-loop iteration number = %s', i)
      k = 0
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = plnphii(yi=yli)
      lnphivi, Zv = plnphii(yi=yvi)
      gi = np.log(kvik) + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tFv = %s\n\tlmbd = %s',
        0, kvik, gi, Fv, 1.,
      )
      lmbd = 1.
      while (gnorm > self.tol) & (k < self.Niter):
        dlnkvi = -lmbd * gi
        max_dlnkvi = np.abs(dlnkvi).max()
        if max_dlnkvi > 6.:
          relax = 6. / max_dlnkvi
          lmbd *= relax
          dlnkvi *= relax
        k += 1
        tkm1 = dlnkvi.dot(gi)
        kvik *= np.exp(dlnkvi)
        Fv = solve2p_FGH(kvik, yi)
        yli = yi / ((kvik - 1.) * Fv + 1.)
        yvi = yli * kvik
        lnphili, Zl = plnphii(yi=yli)
        lnphivi, Zv = plnphii(yi=yvi)
        gi = np.log(kvik) + lnphivi - lnphili
        gnorm = np.linalg.norm(gi)
        logger.debug(
          'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tFv = %s\n\tlmbd = %s',
          k, kvik, gi, Fv, lmbd,
        )
        if (gnorm < self.tol):
          break
        lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
        if lmbd > 30.:
          lmbd = 30.
      if k < self.Niter:
        rhol = yli.dot(self.eos.mwi) / Zl
        rhov = yvi.dot(self.eos.mwi) / Zv
        if rhov < rhol:
          return (
            np.array([Fv, 1. - Fv]),
            np.vstack([yvi, yli]),
            np.array([Zv, Zl]),
          )
        else:
          return (
            np.array([1. - Fv, Fv]),
            np.vstack([yli, yvi]),
            np.array([Zl, Zv]),
          )
    raise ValueError(
      "The solution of the equilibrium was not found. "
      "Try to increase the number of iterations or choose an another method."
    )


