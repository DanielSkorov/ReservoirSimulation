import logging

from functools import (
  partial,
)

import numpy as np

from stability import (
  _stabPT_ss,
  _stabPT_qnss,
)

from rr import (
  solve2p_FGH,
)

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  EOSPTType,
)


logger = logging.getLogger('flash')


class FlashResult(dict):
  """Container for flash calculation outputs with pretty-printing.

  Attributes:
  -----------
  yji: ndarray, shape (Np, Nc)
    Mole fractions of components in each phase. Two-dimensional
    array of real elements of size `(Np, Nc)`, where `Np` is
    the number of phases and `Nc` is the number of components.

  Fj: ndarray, shape (Np,)
    Phase mole fractions. Array of real elements of size `(Np,)`,
    where `Np` is the number of phases.

  Zj: ndarray, shape (Np,)
    Compressibility factors for each phase. Array of real elements of
    size `(Np,)`, where `Np` is the number of phases.

  kvji: ndarray, shape (Np-1, Nc)
    K-values of components in non-reference phases. Two-dimensional
    array of real elements of size `(Np-1, Nc)`, where `Np` is
    the number of phases and `Nc` is the number of components.

  gnorm: float
    Norm of a vector of equilibrium equations.

  success: bool
    Whether or not the procedure exited successfully.
  """
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self):
    with np.printoptions(linewidth=np.inf):
      s = (f"Phase composition:\n{self.yji}\n"
           f"Phase mole fractions:\n{self.Fj}\n"
           f"Phase compressibility factors:\n{self.Zj}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class flash2pPT(object):
  """Two-phase flash calculations.

  Performs two-phase flash calculations for isobaric-isothermal systems.

  Arguments
  ---------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

      - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`, where
        `P: float` is pressure [Pa], `T: float` is temperature [K],
        and `yi: ndarray`, shape `(Nc,)` is an array of components
        mole fractions, `Nc` is the number of components. This method
        is used to generate initial guesses of k-values.

      - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
        This method should return a tuple of logarithms of the fugacity
        coefficients of components and the phase compressibility factor.

    If the solution method would be one of `'newton'` or `'ss-newton'`
    then it also must have:

      - `getPT_lnphii_Z_dnj(P, T, yi) -> tuple[ndarray, float, ndarray]`
        This method should return a tuple of logarithms of the fugacity
        coefficients, the mixture compressibility factor, and partial
        derivatives of logarithms of the fugacity coefficients with
        respect to components mole numbers which are an `ndarray` of
        shape `(Nc, Nc)`.

    Also, this instance must have attributes:

      - `mwi: ndarray`
        Vector of components molecular weights of shape `(Nc,)`.

      - `name: str`
        The EOS name (for proper logging).

  flashmethod: str
    Type of flash calculations solver. Should be one of:

      - `'ss'` (Successive Substitution method),
      - `'qnss'` (Quasi-Newton Successive Substitution method),
      - `'bfgs'` (Currently raises `NotImplementedError`),
      - `'newton'` (Currently raises `NotImplementedError`),
      - `'ss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  stabmethod: str
    Type of stability tests sovler. Should be one of:

      - `'ss'` (Successive Substitution method),
      - `'qnss'` (Quasi-Newton Successive Substitution method),
      - `'bfgs'` (Currently raises `NotImplementedError`),
      - `'newton'` (Currently raises `NotImplementedError`),
      - `'ss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of solver iterations. Default is `50`.

  runstab: bool
    If `True` then the algorithm will perform the stability test, for
    which initial guesses of k-values will be calculated by the method
    of an eos instance and taken from previous flash calculations if
    the flag `useprev` was set `True`. Initial guesses of k-values for
    flash calculations will be taken from the stability test results.
    Default is `True`.

  level: int
    Regulates a set of initial k-values obtained by the method
    `eos.getPT_kvguess(P, T, yi, level)`. Default is `0`.

  stabkwargs: dict
    Dictionary that used to regulate the stability test procedure.
    Default is an empty dictionary.

  useprev: bool
    Allows to preseve previous calculation results (if the solution was
    found) and to use them as the first initial guess for the next run.
    Default is `False`.

  negativeflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`. The value of
    this flag can be changed to `False` if the one-phase state will
    be unstable.

  Methods
  -------
  run(P, T, yi) -> FlashResult
    This method performs two-phase flash calculation for given pressure
    `P: float` in [Pa], temperature `T: float` in [K], composition
    `yi: ndarray` of `Nc` components, and returns flash calculation
    results as an instance of `FlashResult`.
  """
  def __init__(
    self,
    eos: EOSPTType,
    flashmethod: str = 'ss',
    stabmethod: str = 'ss',
    tol: ScalarType = 1e-5,
    maxiter: int = 50,
    runstab: bool = True,
    level: int = 0,
    stabkwargs: dict = {},
    useprev: bool = False,
    negativeflash: bool = True,
  ) -> None:
    self.eos = eos
    self.runstab = runstab
    self.level = level
    self.useprev = useprev
    self.preserved: bool = False
    self.prevkvji: None | MatrixType = None
    self.negativeflash = negativeflash
    if flashmethod == 'ss':
      self.flashsolver = partial(_flash2pPT_ss,
                                 eos=eos, tol=tol, maxiter=maxiter)
    elif flashmethod == 'qnss':
      self.flashsolver = partial(_flash2pPT_qnss,
                                 eos=eos, tol=tol, maxiter=maxiter)
    elif flashmethod == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for flash calculations is not implemented yet.'
      )
    elif flashmethod == 'newton':
      raise NotImplementedError(
        "The Newton's method for flash calculations is not implemented yet."
      )
    elif flashmethod == 'ss-newton':
      raise NotImplementedError(
        'The SS-Newton method for flash calculations is not implemented yet.'
      )
    else:
      raise ValueError(f'The unknown flash-method: {flashmethod}.')
    if stabmethod == 'ss':
      self.stabsolver = partial(_stabPT_ss, eos=eos, **stabkwargs)
    elif stabmethod == 'qnss':
      self.stabsolver = partial(_stabPT_qnss, eos=eos, **stabkwargs)
    self._1pstab_yi = np.zeros_like(eos.mwi)
    self._1pstab_Fj = np.array([1., 0.])
    pass

  def run(self, P: ScalarType, T: ScalarType, yi: VectorType) -> FlashResult:
    """Performs flash calculations for given pressure, temperature and
    composition.

    Arguments
    ---------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Flash calculation results as an instance of `FlashResult` object.
    Important attributes are: `yji` the component mole fractions in
    each phase, `Fj` the phase mole fractions, `Zj` the compressibility
    factors of each phase, `success` a Boolean flag indicating if the
    calculation completed successfully.
    """
    kvji0 = self.eos.getPT_kvguess(P, T, yi, self.level)
    if self.useprev and self.preserved:
      kvji0 = *self.prevkvji, *kvji0
    if self.runstab:
      stab = self.stabsolver(P, T, yi, kvji0)
      if stab.stable:
        return FlashResult(yji=np.vstack([yi, self._1pstab_yi]),
                           Fj=self._1pstab_Fj, Zj=np.array([stab.Z, 0.]),
                           gnorm=-1, success=True)
      else:
        self.negativeflash = False
      if self.useprev and self.preserved:
        kvji0 = *self.prevkvji, *stab.kvji
      else:
        kvji0 = stab.kvji
    flash = self.flashsolver(P, T, yi, kvji0, negativeflash=self.negativeflash)
    if flash.success and self.useprev:
      self.prevkvji = flash.kvji
      self.preserved = True
    return flash


def _flash2pPT_ss(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 30,
  negativeflash: bool = True,
) -> FlashResult:
  """Successive substitution method for two-phase flash calculations.

  Arguments
  ---------
  P: float
    Pressure of a mixture [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[ndarray]
    Initial guesses for k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

      - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
        This method should return a tuple of logarithms of the fugacity
        coefficients of components and the phase compressibility factor.

    Also, this instance must have attributes:

      - `mwi: ndarray`
        Vector of components molecular weights of shape `(Nc,)`.

      - `name: str`
        The EOS name (for proper logging).

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of solver iterations. Default is `50`.

  negativeflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are: `yji` the component mole fractions
  in each phase, `Fj` the phase mole fractions, `Zj` the compressibility
  factors of each phase, `success` a Boolean flag indicating if the
  calculation completed successfully.
  """
  logger.debug(
    'Flash Calculation (SS-method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  plnphii = partial(eos.getPT_lnphii_Z, P=P, T=T)
  i: int
  kvik: VectorType
  for i, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', i)
    k: int = 0
    kvik = kvi0.flatten()
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = plnphii(yi=yli)
    lnphivi, Zv = plnphii(yi=yvi)
    lnkvik = np.log(kvik)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tFv = %s', 0, kvik, gi, Fv,
    )
    while (gnorm > tol) & (k < maxiter):
      k += 1
      lnkvik -= gi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = plnphii(yi=yli)
      lnphivi, Zv = plnphii(yi=yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tFv = %s', k, kvik, gi, Fv,
      )
    if ((gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Fv))
        & ((Fv < 1.) & (1. - Fv < 1.) | negativeflash)):
      rhol = yli.dot(eos.mwi) / Zl
      rhov = yvi.dot(eos.mwi) / Zv
      kvji = np.atleast_2d(kvik)
      if rhov < rhol:
        yji = np.vstack([yvi, yli])
        Fj = np.array([Fv, 1. - Fv])
        Zj = np.array([Zv, Zl])
      else:
        yji = np.vstack([yli, yvi])
        Fj = np.array([1. - Fv, Fv])
        Zj = np.array([Zl, Zv])
      return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                         success=True)
  else:
    logger.warning(
      "Two-phase flash calculations terminates unsuccessfully. "
      "The solution method was SS, EOS: %s. Parameters:"
      "\n\tP = %s, T = %s\n\tyi = %s\n\tkvji = %s.",
      eos.name, P, T, yi, kvji0,
    )
    return FlashResult(yji=np.vstack([yvi, yli]), Fj=np.array([Fv, 1. - Fv]),
                       Zj=np.array([Zv, Zl]), kvji=np.atleast_2d(kvik),
                       gnorm=gnorm, success=False)


def _flash2pPT_qnss(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 30,
  negativeflash: bool = True,
) -> FlashResult:
  """QNSS-method for two-phase flash calculations.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of non-linear equations.
  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Arguments
  ---------
  P: float
    Pressure of a mixture [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvji0: tuple[ndarray]
    Initial guesses for k-values of components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

      - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
        This method should return a tuple of logarithms of the fugacity
        coefficients of components and the phase compressibility factor.

    Also, this instance must have attributes:

      - `mwi: ndarray`
        Vector of components molecular weights of shape `(Nc,)`.

      - `name: str`
        The EOS name (for proper logging).

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of solver iterations. Default is `50`.

  negativeflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are: `yji` the component mole fractions
  in each phase, `Fj` the phase mole fractions, `Zj` the compressibility
  factors of each phase, `success` a Boolean flag indicating if the
  calculation completed successfully.
  """
  logger.debug(
    'Flash Calculation (QNSS-method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  plnphii = partial(eos.getPT_lnphii_Z, P=P, T=T)
  i: int
  kvi0: VectorType
  for i, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', i)
    k: int = 0
    kvik = kvi0.flatten()
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = plnphii(yi=yli)
    lnphivi, Zv = plnphii(yi=yvi)
    lnkvik = np.log(kvik)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    lmbd: ScalarType = 1.
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tFv = %s\n\tlmbd = %s',
      0, kvik, gi, Fv, lmbd,
    )
    while (gnorm > tol) & (k < maxiter):
      dlnkvi = -lmbd * gi
      max_dlnkvi = np.abs(dlnkvi).max()
      if max_dlnkvi > 6.:
        relax = 6. / max_dlnkvi
        lmbd *= relax
        dlnkvi *= relax
      k += 1
      tkm1 = dlnkvi.dot(gi)
      lnkvik += dlnkvi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = plnphii(yi=yli)
      lnphivi, Zv = plnphii(yi=yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgi = %s\n\tFv = %s\n\tlmbd = %s',
        k, kvik, gi, Fv, lmbd,
      )
      if (gnorm < tol):
        break
      lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if ((gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Fv))
        & ((Fv < 1.) & (1. - Fv < 1.) | negativeflash)):
      rhol = yli.dot(eos.mwi) / Zl
      rhov = yvi.dot(eos.mwi) / Zv
      kvji = np.atleast_2d(kvik)
      if rhov < rhol:
        yji = np.vstack([yvi, yli])
        Fj = np.array([Fv, 1. - Fv])
        Zj = np.array([Zv, Zl])
      else:
        yji = np.vstack([yli, yvi])
        Fj = np.array([1. - Fv, Fv])
        Zj = np.array([Zl, Zv])
      return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                         success=True)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully. "
      "The solution method was QNSS, EOS: %s. Parameters:"
      "\n\tP = %s, T = %s\n\tyi = %s\n\tkvji = %s.",
      eos.name, P, T, yi, kvji0,
    )
    return FlashResult(yji=np.vstack([yvi, yli]), Fj=np.array([Fv, 1. - Fv]),
                       Zj=np.array([Zv, Zl]), kvji=np.atleast_2d(kvik),
                       gnorm=gnorm, success=False)

