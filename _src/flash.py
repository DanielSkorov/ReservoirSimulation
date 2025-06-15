import logging

from functools import (
  partial,
)

import numpy as np

from stability import (
  _stabPT_ss,
  _stabPT_qnss,
  _stabPT_newt,
  _stabPT_ssnewt,
  _stabPT_qnssnewt,
)

from rr import (
  solve2p_FGH,
)

from typing import Callable

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  EOSPTType,
)


logger = logging.getLogger('flash')


class FlashResult(dict):
  """Container for flash calculation outputs with pretty-printing.

  Attributes
  ----------
  yji: ndarray, shape (Np, Nc)
    Mole fractions of components in each phase. Two-dimensional
    array of real elements of size `(Np, Nc)`, where `Np` is
    the number of phases and `Nc` is the number of components.

  Fj: ndarray, shape (Np,)
    Phase mole fractions. Array of real elements of size `(Np,)`,
    where `Np` is the number of phases.

  Zj: ndarray, shape (Np,)
    Compressibility factors of each phase. Array of real elements of
    size `(Np,)`, where `Np` is the number of phases.

  kvji: ndarray, shape (Np-1, Nc)
    K-values of components in non-reference phases. Two-dimensional
    array of real elements of size `(Np-1, Nc)`, where `Np` is
    the number of phases and `Nc` is the number of components.

  gnorm: float
    The norm of the vector of equilibrium equations.

  success: bool
    A boolean flag indicating whether or not the procedure exited
    successfully.
  """
  def __getattr__(self, name: str) -> object:
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self) -> str:
    with np.printoptions(linewidth=np.inf):
      s = (f"Phase composition:\n{self.yji}\n"
           f"Phase mole fractions:\n{self.Fj}\n"
           f"Phase compressibility factors:\n{self.Zj}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class flash2pPT(object):
  """Two-phase flash calculations.

  Performs two-phase flash calculations for a given pressure [Pa],
  temperature [K] and composition of a mixture.

  Parameters
  ----------
  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P, T, yi, level) -> tuple[ndarray]`, where
      `P: float` is pressure [Pa], `T: float` is temperature [K],
      and `yi: ndarray`, shape `(Nc,)` is an array of components
      mole fractions, `Nc` is the number of components. This method
      is used to generate initial guesses of k-values.

    - `getPT_lnphii_Z(P, T, yi) -> tuple[ndarray, float]`
      For a given pressure [Pa], temperature [K] and composition
      (ndarray of shape `(Nc,)`), this method must return a tuple that
      contains:

      - a vector of logarithms of the fugacity coefficients of
        components (ndarray of shape `(Nc,)`),
      - the phase compressibility factor of the mixture.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition
      (ndarray of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients (ndarray of shape
        `(Nc,)`),
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers (ndarray of shape
        `(Nc, Nc)`) .

    Also, this instance must have attributes:

    - `mwi: ndarray`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  flashmethod: str
    Type of flash calculations solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Newton's method with preceding successive
      substitution iterations for initial guess improvement),
    - `'qnss-newton'` (Newton's method with preceding quasi-newton
      successive substitution iterations for initial guess improvement).

    Default is `'ss'`.

  stabmethod: str
    Type of stability tests sovler. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Newton's method with preceding successive
      substitution iterations for initial guess improvement),
    - `'qnss-newton'` (Newton's method with preceding quasi-newton
      successive substitution iterations for initial guess improvement).

    Default is `'ss'`.

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

  kwargs: dict
    Other arguments for a phase split solver. It may contain such
    arguments as `tol`, `maxiter` and others, depending on the selected
    phase split solver.

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
    runstab: bool = True,
    level: int = 0,
    stabkwargs: dict = {},
    useprev: bool = False,
    negativeflash: bool = True,
    **kwargs,
  ) -> None:
    self.eos = eos
    self.runstab = runstab
    self.level = level
    self.useprev = useprev
    self.preserved = False
    self.prevkvji: None | MatrixType = None
    self.negativeflash = negativeflash
    if flashmethod == 'ss':
      self.flashsolver = partial(_flash2pPT_ss, eos=eos, **kwargs)
    elif flashmethod == 'qnss':
      self.flashsolver = partial(_flash2pPT_qnss, eos=eos, **kwargs)
    elif flashmethod == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for flash calculations is not implemented yet.'
      )
    elif flashmethod == 'newton':
      self.flashsolver = partial(_flash2pPT_newt, eos=eos, **kwargs)
    elif flashmethod == 'ss-newton':
      self.flashsolver = partial(_flash2pPT_ssnewt, eos=eos, **kwargs)
    elif flashmethod == 'qnss-newton':
      self.flashsolver = partial(_flash2pPT_qnssnewt, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown flash-method: {flashmethod}.')
    if stabmethod == 'ss':
      self.stabsolver = partial(_stabPT_ss, eos=eos, **stabkwargs)
    elif stabmethod == 'qnss':
      self.stabsolver = partial(_stabPT_qnss, eos=eos, **stabkwargs)
    elif stabmethod == 'newton':
      self.stabsolver = partial(_stabPT_newt, eos=eos, **stabkwargs)
    elif stabmethod == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for the stability test is not implemented yet.'
      )
    elif stabmethod == 'ss-newton':
      self.stabsolver = partial(_stabPT_ssnewt, eos=eos, **stabkwargs)
    elif stabmethod == 'qnss-newton':
      self.stabsolver = partial(_stabPT_qnssnewt, eos=eos, **stabkwargs)
    self._1pstab_yi = np.zeros_like(eos.mwi)
    self._1pstab_Fj = np.array([1., 0.])
    pass

  def run(self, P: ScalarType, T: ScalarType, yi: VectorType) -> FlashResult:
    """Performs flash calculations for given pressure, temperature and
    composition.

    Parameters
    ----------
    P: float
      Pressure of a mixture [Pa].

    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    Flash calculation results as an instance of `FlashResult` object.
    Important attributes are:

    - `yji` the component mole fractions in each phase,
    - `Fj` the phase mole fractions,
    - `Zj` the compressibility factors of each phase,
    - `success` a boolean flag indicating if the calculation completed
      successfully.
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
    flash = self.flashsolver(P, T, yi, kvji0,
                             negativeflash=self.negativeflash)
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
  """Successive substitution method for two-phase flash calculations
  using a PT-based equation of state.

  Parameters
  ----------
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
      For a given pressure [Pa], temperature [K] and composition
      (ndarray of shape `(Nc,)`), this method must return a tuple that
      contains:

      - a vector of logarithms of the fugacity coefficients of
        components (ndarray of shape `(Nc,)`),
      - the phase compressibility factor of the mixture.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  negativeflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:

  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.debug(
    'Flash Calculation (SS-method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  for i, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', i)
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
    lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
      k, kvik, gnorm, Fv,
    )
    while gnorm > tol and k < maxiter:
      k += 1
      lnkvik -= gi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
      lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
        k, kvik, gnorm, Fv,
      )
    if (gnorm < tol and np.isfinite(kvik).all() and np.isfinite(Fv)
        and (0. < Fv < 1. or negativeflash)):
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
      logger.info(
        'Two-phase flash P = %s Pa, T = %s K, yi = %s:\n\t'
        'Fj = %s\n\tyji = %s\n\tgnorm = %s\n\tNiter = %s',
        P, T, yi, Fj, yji, gnorm, k,
      )
      return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                         success=True)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully. "
      "The solution method was SS, EOS: %s. Parameters:"
      "\n\tP = %s Pa, T = %s K\n\tyi = %s\n\tkvji = %s.",
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
  """QNSS-method for two-phase flash calculations using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of nonlinear equations.
  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
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
      For a given pressure [Pa], temperature [K] and composition
      (ndarray of shape `(Nc,)`), this method must return a tuple that
      contains:

      - a vector of logarithms of the fugacity coefficients of
        components (ndarray of shape `(Nc,)`),
      - the phase compressibility factor of the mixture.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  negativeflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:

  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.debug(
    'Flash Calculation (QNSS-method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  for i, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', i)
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
    lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    lmbd = 1.
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s\n\tlmbd = %s',
      k, kvik, gnorm, Fv, lmbd,
    )
    while gnorm > tol and k < maxiter:
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
      lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
      lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s\n\tlmbd = %s',
        k, kvik, gnorm, Fv, lmbd,
      )
      if gnorm < tol:
        break
      lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if (gnorm < tol and np.isfinite(kvik).all() and np.isfinite(Fv)
        and (0. < Fv < 1. or negativeflash)):
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
      logger.info(
        'Two-phase flash P = %s Pa, T = %s K, yi = %s:\n\t'
        'Fj = %s\n\tyji = %s\n\tgnorm = %s\n\tNiter = %s',
        P, T, yi, Fj, yji, gnorm, k,
      )
      return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                         success=True)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully. "
      "The solution method was QNSS, EOS: %s. Parameters:"
      "\n\tP = %s Pa, T = %s K\n\tyi = %s\n\tkvji = %s.",
      eos.name, P, T, yi, kvji0,
    )
    return FlashResult(yji=np.vstack([yvi, yli]), Fj=np.array([Fv, 1. - Fv]),
                       Zj=np.array([Zv, Zl]), kvji=np.atleast_2d(kvik),
                       gnorm=gnorm, success=False)


def _flash2pPT_newt(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 30,
  negativeflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> FlashResult:
  """Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient.

  Parameters
  ----------
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

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition
      (ndarray of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients (ndarray of shape
        `(Nc,)`),
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers (ndarray of shape
        `(Nc, Nc)`) .

    Also, this instance must have attributes:

    - `mwi: ndarray`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate Newton's method successfully if the norm of the gradient
    of Gibbs energy function is less than `tol`. Default is `1e-6`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  negativeflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:

  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.debug(
    "Flash Calculation (Newton's method)\n\tP = %s Pa\n\tT = %s K\n\tyi = %s",
    P, T, yi,
  )
  U = np.full(shape=(eos.Nc, eos.Nc), fill_value=-1.)
  for i, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', i)
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    if np.isclose(Fv, 0.):
      Fv = 1e-8
    elif np.isclose(Fv, 1.):
      Fv = 0.99999999
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
    lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
      k, kvik, gnorm, Fv,
    )
    while gnorm > tol and k < maxiter:
      ui = yi / (yli * yvi) - 1.
      FvFl = 1. / (Fv * (1. - Fv))
      np.fill_diagonal(U, ui)
      H = U * FvFl + (dlnphividnj + dlnphilidnj)
      dnvi = linsolver(H, -gi)
      dlnkvi = U.dot(dnvi) * FvFl
      k += 1
      lnkvik += dlnkvi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
      lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
      gi = lnkvik + lnphivi - lnphili
      gnormkp1 = np.linalg.norm(gi)
      if gnormkp1 < gnorm or forcenewton:
        gnorm = gnormkp1
        logger.debug(
          'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
          k, kvik, gnorm, Fv,
        )
      else:
        # TODO: implement TR-step
        lnkvik -= gi
        kvik = np.exp(lnkvik)
        Fv = solve2p_FGH(kvik, yi)
        yli = yi / ((kvik - 1.) * Fv + 1.)
        yvi = yli * kvik
        lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
        lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
        gi = lnkvik + lnphivi - lnphili
        gnorm = np.linalg.norm(gi)
        logger.debug(
          'Iteration (SS) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
          k, kvik, gnorm, Fv,
        )
    if (gnorm < tol and np.isfinite(kvik).all() and np.isfinite(Fv)
        and (0. < Fv < 1. or negativeflash)):
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
      logger.info(
        'Two-phase flash P = %s Pa, T = %s K, yi = %s:\n\t'
        'Fj = %s\n\tyji = %s\n\tgnorm = %s\n\tNiter = %s',
        P, T, yi, Fj, yji, gnorm, k,
      )
      return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                         success=True)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully. "
      "The solution method was Newton (forced: %s), EOS: %s. Parameters:"
      "\n\tP = %s Pa, T = %s K\n\tyi = %s\n\tkvji = %s.",
      forcenewton, eos.name, P, T, yi, kvji0,
    )
    return FlashResult(yji=np.vstack([yvi, yli]), Fj=np.array([Fv, 1. - Fv]),
                       Zj=np.array([Zv, Zl]), kvji=np.atleast_2d(kvik),
                       gnorm=gnorm, success=False)


def _flash2pPT_ssnewt(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 30,
  tol_ss: ScalarType = 1e-2,
  maxiter_ss: int = 10,
  negativeflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> FlashResult:
  """Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient. Successive substitution iterations
  precede Newton's method to improve initial guesses of k-values.

  Parameters
  ----------
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
      For a given pressure [Pa], temperature [K] and composition
      (ndarray of shape `(Nc,)`), this method must return a tuple that
      contains:

      - a vector of logarithms of the fugacity coefficients of
        components (ndarray of shape `(Nc,)`),
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition
      (ndarray of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients (ndarray of shape
        `(Nc,)`),
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers (ndarray of shape
        `(Nc, Nc)`) .

    Also, this instance must have attributes:

    - `mwi: ndarray`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  tol_ss: float
    Switch to Newton's method if the norm of the vector of equilibrium
    equations is less than `tol_ss`. Default is `1e-2`.

  maxiter_ss: int
    The maximum number of the successive substitution iterations.
    Default is `10`.

  negativeflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:

  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.debug(
    'Flash Calculation (SS-Newton method)\n\t'
    'P = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  for i, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', i)
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
    lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration (SS) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
      k, kvik, gnorm, Fv,
    )
    while gnorm > tol_ss and k < maxiter_ss:
      k += 1
      lnkvik -= gi
      kvik = np.exp(lnkvik)
      Fv = solve2p_FGH(kvik, yi)
      yli = yi / ((kvik - 1.) * Fv + 1.)
      yvi = yli * kvik
      lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
      lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration (SS) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
        k, kvik, gnorm, Fv,
      )
    if np.isfinite(kvik).all() and np.isfinite(Fv):
      if gnorm < tol:
        if 0. < Fv < 1. or negativeflash:
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
          logger.info(
            'Two-phase flash P = %s Pa, T = %s K, yi = %s:\n\t'
            'Fj = %s\n\tyji = %s\n\tgnorm = %s\n\tNiter = %s',
            P, T, yi, Fj, yji, gnorm, k,
          )
          return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                             success=True)
      else:
        U = np.full(shape=(eos.Nc, eos.Nc), fill_value=-1.)
        lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
        lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
        logger.debug(
          'Iteration (Newton) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
          k, kvik, gnorm, Fv,
        )
        while gnorm > tol and k < maxiter:
          ui = yi / (yli * yvi) - 1.
          FvFl = 1. / (Fv * (1. - Fv))
          np.fill_diagonal(U, ui)
          H = U * FvFl + (dlnphividnj + dlnphilidnj)
          dnvi = linsolver(H, -gi)
          dlnkvi = U.dot(dnvi) * FvFl
          k += 1
          lnkvik += dlnkvi
          kvik = np.exp(lnkvik)
          Fv = solve2p_FGH(kvik, yi)
          yli = yi / ((kvik - 1.) * Fv + 1.)
          yvi = yli * kvik
          lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1.-Fv)
          lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
          gi = lnkvik + lnphivi - lnphili
          gnormkp1 = np.linalg.norm(gi)
          if gnormkp1 < gnorm or forcenewton:
            gnorm = gnormkp1
            logger.debug(
              'Iteration (Newton) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
              k, kvik, gnorm, Fv,
            )
          else:
            # TODO: implement TR-step
            lnkvik -= gi
            kvik = np.exp(lnkvik)
            Fv = solve2p_FGH(kvik, yi)
            yli = yi / ((kvik - 1.) * Fv + 1.)
            yvi = yli * kvik
            lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P,T, yli, 1.-Fv)
            lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P,T, yvi, Fv)
            gi = lnkvik + lnphivi - lnphili
            gnorm = np.linalg.norm(gi)
            logger.debug(
              'Iteration (SS) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
              k, kvik, gnorm, Fv,
            )
        if (gnorm < tol and np.isfinite(kvik).all() and np.isfinite(Fv)
            and (0. < Fv < 1. or negativeflash)):
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
          logger.info(
            'Two-phase flash P = %s Pa, T = %s K, yi = %s:\n\t'
            'Fj = %s\n\tyji = %s\n\tgnorm = %s\n\tNiter = %s',
            P, T, yi, Fj, yji, gnorm, k,
          )
          return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                             success=True)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully. "
      "The solution method was SS-Newton (forced: %s), EOS: %s. Parameters:"
      "\n\tP = %s Pa, T = %s K\n\tyi = %s\n\tkvji = %s.",
      forcenewton, eos.name, P, T, yi, kvji0,
    )
    return FlashResult(yji=np.vstack([yvi, yli]), Fj=np.array([Fv, 1. - Fv]),
                       Zj=np.array([Zv, Zl]), kvji=np.atleast_2d(kvik),
                       gnorm=gnorm, success=False)


def _flash2pPT_qnssnewt(
  P: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvji0: tuple[VectorType],
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 30,
  tol_qnss: ScalarType = 1e-2,
  maxiter_qnss: int = 10,
  negativeflash: bool = True,
  forcenewton: bool = False,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
) -> FlashResult:
  """Performs minimization of the Gibbs energy function using Newton's
  method and a PT-based equation of state. A switch to the successive
  substitution iteration is implemented if Newton's method does not
  decrease the norm of the gradient. Quasi-Newton Successive Substitution
  (QNSS) iterations precede Newton's method to improve initial guesses
  of k-values.

  For the details of the QNSS-method see 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
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
      For a given pressure [Pa], temperature [K] and composition
      (ndarray of shape `(Nc,)`), this method must return a tuple that
      contains:

      - a vector of logarithms of the fugacity coefficients of
        components (ndarray of shape `(Nc,)`),
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition
      (ndarray of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients (ndarray of shape
        `(Nc,)`),
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers (ndarray of shape
        `(Nc, Nc)`) .

    Also, this instance must have attributes:

    - `mwi: ndarray`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate successfully if the norm of the equilibrium equations
    vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  tol_qnss: float
    Switch to Newton's method if the norm of the vector of equilibrium
    equations is less than `tol_qnss`. Default is `1e-2`.

  maxiter_qnss: int
    The maximum number of quasi-newton successive substitution
    iterations. Default is `10`.

  negativeflash: bool
    A flag indicating if unphysical phase mole fractions can be
    considered as a correct solution. Default is `True`.

  forcenewton: bool
    A flag indicating whether it is allowed to ignore the condition to
    switch from Newton's method to successive substitution iterations.
    Default is `False`.

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    a vector `b` of shape `(Nc,)` and finds a vector `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  Flash calculation results as an instance of `FlashResult` object.
  Important attributes are:

  - `yji` the component mole fractions in each phase,
  - `Fj` the phase mole fractions,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation completed
    successfully.
  """
  logger.debug(
    'Flash Calculation (QNSS-Newton method)\n\t'
    'P = %s Pa\n\tT = %s K\n\tyi = %s',
    P, T, yi,
  )
  for i, kvi0 in enumerate(kvji0):
    logger.debug('The kv-loop iteration number = %s', i)
    k = 0
    kvik = kvi0.flatten()
    lnkvik = np.log(kvik)
    Fv = solve2p_FGH(kvik, yi)
    yli = yi / ((kvik - 1.) * Fv + 1.)
    yvi = yli * kvik
    lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
    lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
    gi = lnkvik + lnphivi - lnphili
    gnorm = np.linalg.norm(gi)
    lmbd = 1.
    logger.debug(
      'Iteration (QNSS) #%s:\n\t'
      'kvi = %s\n\tgnorm = %s\n\tFv = %s\n\tlmbd = %s',
      k, kvik, gnorm, Fv, lmbd,
    )
    while gnorm > tol_qnss and k < maxiter_qnss:
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
      lnphili, Zl = eos.getPT_lnphii_Z(P, T, yli)
      lnphivi, Zv = eos.getPT_lnphii_Z(P, T, yvi)
      gi = lnkvik + lnphivi - lnphili
      gnorm = np.linalg.norm(gi)
      logger.debug(
        'Iteration (QNSS) #%s:\n\t'
        'kvi = %s\n\tgnorm = %s\n\tFv = %s\n\tlmbd = %s',
        k, kvik, gnorm, Fv, lmbd,
      )
      if gnorm < tol:
        break
      lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
      if lmbd > 30.:
        lmbd = 30.
    if np.isfinite(kvik).all() and np.isfinite(Fv):
      if gnorm < tol:
        if 0. < Fv < 1. or negativeflash:
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
          logger.info(
            'Two-phase flash P = %s Pa, T = %s K, yi = %s:\n\t'
            'Fj = %s\n\tyji = %s\n\tgnorm = %s\n\tNiter = %s',
            P, T, yi, Fj, yji, gnorm, k,
          )
          return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                             success=True)
      else:
        U = np.full(shape=(eos.Nc, eos.Nc), fill_value=-1.)
        lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1. - Fv)
        lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
        logger.debug(
          'Iteration (Newton) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
          k, kvik, gnorm, Fv,
        )
        while gnorm > tol and k < maxiter:
          ui = yi / (yli * yvi) - 1.
          FvFl = 1. / (Fv * (1. - Fv))
          np.fill_diagonal(U, ui)
          H = U * FvFl + (dlnphividnj + dlnphilidnj)
          dnvi = linsolver(H, -gi)
          dlnkvi = U.dot(dnvi) * FvFl
          k += 1
          lnkvik += dlnkvi
          kvik = np.exp(lnkvik)
          Fv = solve2p_FGH(kvik, yi)
          yli = yi / ((kvik - 1.) * Fv + 1.)
          yvi = yli * kvik
          lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P, T, yli, 1.-Fv)
          lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P, T, yvi, Fv)
          gi = lnkvik + lnphivi - lnphili
          gnormkp1 = np.linalg.norm(gi)
          if gnormkp1 < gnorm or forcenewton:
            gnorm = gnormkp1
            logger.debug(
              'Iteration (Newton) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
              k, kvik, gnorm, Fv,
            )
          else:
            # TODO: implement TR-step
            lnkvik -= gi
            kvik = np.exp(lnkvik)
            Fv = solve2p_FGH(kvik, yi)
            yli = yi / ((kvik - 1.) * Fv + 1.)
            yvi = yli * kvik
            lnphili, Zl, dlnphilidnj = eos.getPT_lnphii_Z_dnj(P,T, yli, 1.-Fv)
            lnphivi, Zv, dlnphividnj = eos.getPT_lnphii_Z_dnj(P,T, yvi, Fv)
            gi = lnkvik + lnphivi - lnphili
            gnorm = np.linalg.norm(gi)
            logger.debug(
              'Iteration (SS) #%s:\n\tkvi = %s\n\tgnorm = %s\n\tFv = %s',
              k, kvik, gnorm, Fv,
            )
        if (gnorm < tol and np.isfinite(kvik).all() and np.isfinite(Fv)
            and (0. < Fv < 1. or negativeflash)):
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
          logger.info(
            'Two-phase flash P = %s Pa, T = %s K, yi = %s:\n\t'
            'Fj = %s\n\tyji = %s\n\tgnorm = %s\n\tNiter = %s',
            P, T, yi, Fj, yji, gnorm, k,
          )
          return FlashResult(yji=yji, Fj=Fj, Zj=Zj, kvji=kvji, gnorm=gnorm,
                             success=True)
  else:
    logger.warning(
      "Two-phase flash calculation terminates unsuccessfully. "
      "The solution method was QNSS-Newton (forced: %s), EOS: %s. Parameters:"
      "\n\tP = %s Pa, T = %s K\n\tyi = %s\n\tkvji = %s.",
      forcenewton, eos.name, P, T, yi, kvji0,
    )
    return FlashResult(yji=np.vstack([yvi, yli]), Fj=np.array([Fv, 1. - Fv]),
                       Zj=np.array([Zv, Zl]), kvji=np.atleast_2d(kvik),
                       gnorm=gnorm, success=False)

