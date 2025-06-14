import logging

from functools import (
  partial,
)

import numpy as np

from utils import (
  mineig_rayquot,
)

from typing import Callable

from custom_types import (
  ScalarType,
  VectorType,
  MatrixType,
  EOSPTType,
  EOSVTType,
)

from stability import (
  StabResult,
  stabilityPT,
)


logger = logging.getLogger('bound')


def getVT_Tspinodal(
  V: ScalarType,
  yi: VectorType,
  eos: EOSVTType,
  T0: None | ScalarType = None,
  zeta0i : None | VectorType = None,
  multdT0: ScalarType = 1e-5,
  tol: ScalarType = 1e-5,
  maxiter: int = 25,
) -> tuple[ScalarType, VectorType]:
  """Calculates the spinodal temperature for a given volume
  and composition of a mixture.

  Parameters
  ----------
  V: float
    Volume of a mixture [m3].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  eos: EOSVTType
    An initialized instance of a VT-based equation of state.

  T0: None | float
    An initial guess of the spinodal temperature [K]. If it equals
    `None` a pseudocritical temperature will used.

  zeta0i: None | ndarray, shape (Nc,)
    An initial guess for the eigenvector corresponding to the lowest
    eigenvalue of the matrix of second partial derivatives of the
    Helmholtz energy function with respect to component mole
    numbers. If it equals `None` then `yi` will be used instead.

  multdT0: float
    A multiplier used to compute the temperature shift to estimate
    the partial derivative of the lowest eigenvalue with respect to
    temperature at the first iteration. Default is `1e-5`.

  tol: float
    Terminate successfully if the absolute value of the lowest
    eigenvalue is less then `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of iterations. Default is `25`.

  Returns
  -------
  A tuple of the spinodal temperature and the eigenvector.

  Raises
  ------
  The `ValueError` if the solution was not found.
  """
  logger.debug(
    'The spinodal temperature calculation procedure\n\tV = %s m3\n\tyi = %s',
    V, yi,
  )
  k = 0
  if T0 is None:
    Tk = 1.3 * yi.dot(eos.Tci)
  else:
    Tk = T0
  if zeta0i is None:
    zeta0i = yi
  Q = eos.getVT_lnfi_dnj(V, Tk, yi)[1]
  Nitrq, zetai, lmbdk = mineig_rayquot(Q, zeta0i)
  dT = multdT0 * Tk
  QdT = eos.getVT_lnfi_dnj(V, Tk + dT, yi)[1]
  lmbdkdT = np.linalg.eigvals(QdT).min()
  dlmbddT = (lmbdkdT - lmbdk) / dT
  dT = -lmbdk / dlmbddT
  logger.debug(
    'Iteration #%s:\n\tlmbd = %s\n\tzetai = %s\n\tT = %s\n\tdT = %s',
    k, lmbdk, zetai, Tk, dT,
  )
  k += 1
  while (np.abs(lmbdk) > tol) & (k < maxiter):
    Tkp1 = Tk + dT
    Q = eos.getVT_lnfi_dnj(V, Tkp1, yi)[1]
    Nitrq, zetai, lmbdkp1 = mineig_rayquot(Q, zetai)
    dlmbddT = (lmbdkp1 - lmbdk) / dT
    dT = -lmbdkp1 / dlmbddT
    Tk = Tkp1
    lmbdk = lmbdkp1
    logger.debug(
      'Iteration #%s:\n\tlmbd = %s\n\tzetai = %s\n\tT = %s\n\tdT = %s',
      k, lmbdk, zetai, Tk, dT,
    )
    k += 1
  if np.abs(lmbdk) < tol:
    return Tk, zetai
  logger.warning(
    "The spinodal temperature was not found using %s:"
    "\n\tV = %s\n\tyi = %s\n\tT0 = %s"
    "\n\tzeta0i = %s\n\tmultdT0 = %s\n\ttol = %s\n\tmaxiter = %s",
    eos.name, V, yi, T0, zeta0i, multdT0, tol, maxiter,
  )
  raise ValueError(
    "The spinodal temperature solution procedure completed unsuccessfully. "
    "Try to increase the number of iterations or change the initial guess."
  )


def getVT_PcTc(
  yi: VectorType,
  eos: EOSVTType,
  v0: None | ScalarType = None,
  T0: None | ScalarType = None,
  kappa0: None | ScalarType = 3.5,
  multdV0: ScalarType = 1e-5,
  krange: tuple[ScalarType] = (1.1, 5.),
  kstep: ScalarType = .1,
  tol: ScalarType = 1e-5,
  maxiter: int = 25,
) -> tuple[ScalarType, ScalarType]:
  """Calculates the critical pressure and temperature of a mixture.

  Parameters
  ----------
  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  eos: EOSVTType
    An initialized instance of a VT-based equation of state.

  v0: float
    An initial guess of the critical molar volume of a mixture [m3/mol].
    Default is `None` which means that the gridding procedure or the
    `kappa0` will be used instead.

  kappa0: float | None
    An initial guess for the relation of the critical molar volume
    to the minimal possible volume provided by an equation of state.
    Default is `None` which means that the gridding procedure or the
    `v0` will be used instead.

  krange: tuple[float]
    A range of possible values of the relation of the molar volume
    to the minimal possible volume provided by an equation of state
    for the gridding procedure. The gridding procedure will be used
    if both initial guesses `v0` and `kappa0` are equal `None`.
    Default is `(1.1, 5.0)`.

  kstep: float
    A step which is used to perform the gridding procedure to find
    an initial guess of the relation of the molar volume to the minimal
    possible volume. Default is `0.1`.

   tol: float
    Terminate successfully if the absolute value of the primary
    variable change is less than `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of iterations. Default is `25`.

  Returns
  -------
  A tuple of the critical pressure and temperature.

  Raises
  ------
  The `ValueError` if the solution was not found.
  """
  logger.debug('The critical point calculation procedure\n\tyi = %s', yi)
  k = 0
  if T0 is None:
    T = 1.3 * yi.dot(eos.Tci)
  else:
    T = T0
  if v0 is None:
    if kappa0 is None:
      raise NotImplementedError(
        'The gridding procedure is not implemented yet.'
      )
    else:
      vmin = eos.getVT_vmin(T, yi)
      vk = kappa0 * vmin
  else:
    vk = v0
  T, zetaik = getVT_Tspinodal(vk, yi, eos, T, yi)
  vmin = eos.getVT_vmin(T, yi)
  kappak = vk / vmin
  Ck = (kappak - 1.)**2 * eos.getVT_d3F(vk, T, yi, zetaik)
  dv = multdV0 * vk
  vks = vk + dv
  kappaks = vks / vmin
  Cks = (kappaks - 1.)**2 * eos.getVT_d3F(vks, T, yi, zetaik)
  dCdkappa = (Cks - Ck) / (kappaks - kappak)
  dkappa = -Ck / dCdkappa
  logger.debug(
    'Iteration #%s:\n\tkappa = %s\n\tT = %s\n\tC = %s\n\tdkappa = %s',
    k, kappak, T, Ck, dkappa,
  )
  k += 1
  while (np.abs(dkappa) > tol) & (k < maxiter):
    kappakp1 = kappak + dkappa
    vkp1 = kappakp1 * vmin
    if vkp1 < vmin:
      vkp1 = (vk + vmin) / 2.
      kappakp1 = vkp1 / vmin
      dkappa = kappakp1 - kappak
    T, zetaikp1 = getVT_Tspinodal(vkp1, yi, eos, T, zetaik)
    if zetaikp1[0] * zetaik[0] < 0.:
      zetaikp1 *= -1.
    Ckp1 = (kappakp1 - 1.)**2 * eos.getVT_d3F(vkp1, T, yi, zetaikp1)
    dCdkappa = (Ckp1 - Ck) / dkappa
    dkappa = -Ckp1 / dCdkappa
    vmin = eos.getVT_vmin(T, yi)
    vk = vkp1
    kappak = kappakp1
    Ck = Ckp1
    zetaik = zetaikp1
    logger.debug(
      'Iteration #%s:\n\tkappa = %s\n\tT = %s\n\tC = %s\n\tdkappa = %s',
      k, kappak, T, Ck, dkappa,
    )
    k += 1
  if np.abs(dkappa) < tol:
    Pc = eos.getVT_P(vk, T, yi)
    logger.info(
      'Critical point for yi = %s:\n\tPc = %s Pa\n\tTc = %s K'
      '\n\tdkappa = %s\n\tNiter = %s',
      yi, Pc, T, dkappa, k,
    )
    return Pc, T
  logger.warning(
    "The critical point was not found using %s:"
    "\n\tyi = %s\n\tv0 = %s m3/mol\n\tT0 = %s K\n\tkappa0 = %s\n\t"
    "multdV0 = %s\n\tkrange = %s\n\tkstep = %s\n\ttol = %s\n\tmaxiter = %s",
    eos.name, yi, v0, T0, kappa0, multdV0, krange, kstep, tol, maxiter,
  )
  raise ValueError(
    "The critical point solution procedure completed unsuccessfully. "
    "Try to increase the number of iterations or change the initial guess."
  )


class SatResult(dict):
  """Container for saturation point calculation outputs with
  pretty-printing.

  Attributes
  ----------
  P: float
    Saturation pressure [Pa].

  T: float
    Saturation temperature [K].

  yji: ndarray, shape (Np, Nc)
    Equilibrium composition of two phases for the saturation point.
    Two-dimensional array of real elements of size `(2, Nc)`, where `Np`
    is the number of phases and `Nc` is the number of components.

  Zj: ndarray, shape (Np,)
    Compressibility factors of each phase. Array of real elements of
    size `(Np,)`, where `Np` is the number of phases.

  success: bool
    Whether or not the procedure exited successfully.
  """
  def __getattr__(self, name: str) -> object:
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self) -> str:
    with np.printoptions(linewidth=np.inf):
      s = (f"Saturation pressure: {self.P} Pa\n"
           f"Saturation temperature: {self.T} K\n"
           f"Phase composition:\n{self.yji}\n"
           f"Phase compressibility factors:\n{self.Zj}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class PsatPT(object):
  """Saturation pressure calculation.

  Performs saturation pressure calculation using PT-based equations of
  state.

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
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor.

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_lnphii_Z_dnj_dP(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  method: str
    Type of the solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Currently raises `NotImplementedError`),
    - `'qnss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  step: float
    To specify the confidence interval for the saturation pressure
    calculation, the preliminary search is performed. This parameter
    regulates the step of this search in fraction units. For example,
    if it is necessary to find the upper bound of the confidence
    interval, then the next value of pressure will be calculated from
    the previous one using the formula: `Pnext = Pprev * (1. + step)`.
    Default is `0.1`.

  lowerlimit: float
    During the preliminary search, the pressure can not drop below
    the lower limit. Otherwise, the `ValueError` will be rised.
    Default is `1.` [Pa].

  upperlimit: float
    During the preliminary search, the pressure can not exceed the
    upper limit. Otherwise, the `ValueError` will be rised.
    Default is `1e8` [Pa].

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval of the saturation pressure. This dictionary is used to
    specify arguments for the stability test procedure. Default is an
    empty dictionary.

  kwargs: dict
    Other arguments for a Psat-solver. It may contain such arguments
    as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others depending
    on the selected solver.

  Methods
  -------
  run(T, yi, P0) -> SatResult
    This method performs the saturation pressure calculation for given
    temperature `T: float` in [K], composition `yi: ndarray` of `Nc`
    components and the initial guess `P0: float` in [Pa]. This method
    returns saturation pressure calculation results as an instance of
    `SatResult`.
  """
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'ss',
    upper: bool = True,
    step: ScalarType = 0.1,
    lowerlimit: ScalarType = 1.,
    upperlimit: ScalarType = 1e8,
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.upper = upper
    self.step = step
    self.lowerlimit = lowerlimit
    self.upperlimit = upperlimit
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.psatsolver = partial(_PsatPT_ss, eos=eos, upper=upper, **kwargs)
    elif method == 'qnss':
      self.psatsolver = partial(_PsatPT_qnss, eos=eos, upper=upper, **kwargs)
    elif method == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for the saturation pressure calculation is not '
        'implemented yet.'
      )
    elif method == 'newton':
      self.psatsolver = partial(_PsatPT_newtA, eos=eos, upper=upper, **kwargs)
    elif method == 'newton-b':
      self.psatsolver = partial(_PsatPT_newtB, eos=eos, upper=upper, **kwargs)
    elif method == 'newton-c':
      self.psatsolver = partial(_PsatPT_newtC, eos=eos, upper=upper, **kwargs)
    elif method == 'ss-newton':
      raise NotImplementedError(
        'The SS-Newton method for the saturation pressure calculation is '
        'not implemented yet.'
      )
    elif method == 'qnss-newton':
      raise NotImplementedError(
        'The QNSS-Newton method for the saturation pressure calculation is '
        'not implemented yet.'
      )
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    T: ScalarType,
    yi: VectorType,
    P0: ScalarType,
  ) -> SatResult:
    """Performs the saturation pressure calculation for a given
    temperature and composition. To improve an initial guess, the
    preliminary search is performed.

    Parameters
    ----------
    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    P0: float
      Initial guess of the saturation pressure [Pa].

    Returns
    -------
    Saturation pressure calculation results as an instance of the
    `SatResult`. Important attributes are:

    - `P` the saturation pressure in [Pa],
    - `T` the saturation temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase,
    - `success` a boolean flag indicating if the calculation completed
      successfully.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    stab = self.stabsolver.run(P0, T, yi)
    logger.debug(
      'For the initial guess P0 = %s Pa, the one-phase state is stable: %s',
      P0, stab.stable,
    )
    if stab.stable and self.upper:
      logger.debug(
        'Finding the two-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Pmax = P0
      c = 1. - self.step
      Pmin = c * P0
      stabmin = self.stabsolver.run(Pmin, T, yi)
      logger.debug(
        'Pmin = %s Pa, the one-phase state is stable: %s',
        Pmin, stabmin.stable,
      )
      if stabmin.stable:
        Pmax = Pmin
      while stabmin.stable and Pmin > self.lowerlimit:
        Pmin *= c
        stabmin = self.stabsolver.run(Pmin, T, yi)
        logger.debug(
          'Pmin = %s Pa, the one-phase state is stable: %s',
          Pmin, stabmin.stable,
        )
        if stabmin.stable:
          Pmax = Pmin
      if Pmin < self.lowerlimit:
        raise ValueError(
          'The two-phase region was not identified. It could be because of \n'
          'its narrowness or absence. Try to change the value of P0 or \n'
          'stability test parameters using the `stabkwargs` argument of \n'
          'this class. It also might be helpful to reduce the value of the \n'
          'argument `step`.'
        )
      else:
        P0 = Pmin
        stab = stabmin
    elif not stab.stable and self.upper:
      logger.debug(
        'Finding the one-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Pmin = P0
      c = 1. + self.step
      Pmax = c * P0
      stabmax = self.stabsolver.run(Pmax, T, yi)
      logger.debug(
        'Pmax = %s Pa, the one-phase state is stable: %s',
        Pmax, stabmax.stable,
      )
      if not stabmax.stable:
        Pmin = Pmax
      while not stabmax.stable and Pmax < self.upperlimit:
        Pmax *= c
        stabmax = self.stabsolver.run(Pmax, T, yi)
        logger.debug(
          'Pmax = %s Pa, the one-phase state is stable: %s',
          Pmax, stabmax.stable,
        )
        if not stabmax.stable:
          Pmin = Pmax
          stab = stabmax
      if Pmax > self.upperlimit:
        raise ValueError(
          'The one-phase region was not identified. Try to change the \n'
          'value of P0 and `upperlimit` parameter of this class.'
        )
      else:
        P0 = Pmin
    elif stab.stable and not self.upper:
      logger.debug(
        'Finding the two-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Pmin = P0
      c = 1. + self.step
      Pmax = c * P0
      stabmax = self.stabsolver.run(Pmax, T, yi)
      logger.debug(
        'Pmax = %s Pa, the one-phase state is stable: %s',
        Pmax, stabmax.stable,
      )
      while stabmax.stable and Pmax < self.upperlimit:
        Pmax *= c
        stabmax = self.stabsolver.run(Pmax, T, yi)
        logger.debug(
          'Pmax = %s Pa, the one-phase state is stable: %s',
          Pmax, stabmax.stable,
        )
        if stabmax.stable:
          Pmin = Pmax
      if Pmax > self.upperlimit:
        raise ValueError(
          'The two-phase region was not identified. It could be because of \n'
          'its narrowness or absence. Try to change the value of P0 or \n'
          'stability test parameters using the `stabkwargs` argument of \n'
          'this class. It also might be helpful to reduce the value of the \n'
          'argument `step`.'
        )
      else:
        P0 = Pmax
        stab = stabmax
    else:
      logger.debug(
        'Finding the one-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Pmax = P0
      c = 1. - self.step
      Pmin = c * P0
      stabmin = self.stabsolver.run(Pmin, T, yi)
      logger.debug(
        'Pmin = %s Pa, the one-phase state is stable: %s',
        Pmin, stabmin.stable,
      )
      while not stabmin.stable and Pmin > self.lowerlimit:
        Pmin *= c
        stabmin = self.stabsolver.run(Pmin, T, yi)
        logger.debug(
          'Pmin = %s Pa, the one-phase state is stable: %s',
          Pmin, stabmin.stable,
        )
        if not stabmin.stable:
          Pmax = Pmin
          stab = stabmin
      if Pmin < self.lowerlimit:
        raise ValueError(
          'The one-phase region was not identified. Try to change the \n'
          'value of P0 and `lowerlimit` parameter of this class.'
        )
      else:
        P0 = Pmax
    return self.psatsolver(P0, T, yi, stab.kvji[0], Pmin=Pmin, Pmax=Pmax)


def _solveTPDeqPT_forP(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  yti: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Pmax0: ScalarType = 1e8,
  Pmin0: ScalarType = 1.,
  increasing: bool = True,
) -> tuple[ScalarType, VectorType, VectorType,
           ScalarType, ScalarType, ScalarType]:
  """Solves the TPD-equation using the PT-based equations of state for
  pressure at a constant temperature. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  yti: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Pmax0: float
    The initial upper bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `1e8` [Pa].

  Pmin0: float
    The initial lower bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `1.0` [Pa].

  increasing: bool
    A flag that indicates if the TPD-equation vs pressure is the
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation pressure,
  - an array of the natural logarithms of fugacity coefficients of
    components in the trial phase,
  - the same for the initial composition of the mixture,
  - compressibility factors for both mixtures,
  - value of the tangent-plane distance.
  """
  k = 0
  Pk = P0
  Pmin = Pmin0
  Pmax = Pmax0
  lnkvi = np.log(yti / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphiti, Zt, dlnphitidP = eos.getPT_lnphii_Z_dP(Pk, T, yti)
  TPD = yti.dot(lnkvi + lnphiti - lnphiyi)
  while k < maxiter and np.abs(TPD) > tol:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Pmin = Pk
    else:
      Pmax = Pk
    dTPDdP = yti.dot(dlnphitidP - dlnphiyidP)
    dP = - TPD / dTPDdP
    Pkp1 = Pk + dP
    if Pkp1 > Pmax or Pkp1 < Pmin:
      Pkp1 = .5 * (Pmin + Pmax)
      dP = Pkp1 - Pk
    if np.abs(dP) < 1e-8:
      break
    Pk = Pkp1
    k += 1
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphiti, Zt, dlnphitidP = eos.getPT_lnphii_Z_dP(Pk, T, yti)
    TPD = yti.dot(lnkvi + lnphiti - lnphiyi)
  return Pk, lnphiti, lnphiyi, Zt, Zy, TPD


def _PsatPT_ss(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Pmax: ScalarType = 1e8,
  Pmin: ScalarType = 1.,
  upper: bool = True,
) -> SatResult:
  """Successive substitution (SS) method for the saturation pressure
  calculation using a PT-based equation of state.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Pmax: float
    The upper bound for the TPD-equation solver. Default is `1e8` [Pa].

  Pmin: float
    The lower bound for the TPD-equation solver. Default is `1.0` [Pa].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    'Saturation pressure calculation using the SS-method:\n'
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa',
    P0, T, yi, Pmin, Pmax,
  )
  solverTPDeq = partial(_solveTPDeqPT_forP, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Pmin0=Pmin, Pmax0=Pmax,
                        increasing=upper)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  xi = ni / ni.sum()
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  gi = lnkvik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  TPD = ni.dot(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tTPD = %s\n\tP = %s Pa',
    k, kvik, gnorm, TPD, Pk,
  )
  while ((gnorm > tol) or (np.abs(TPD) > tol_tpd)) and (k < maxiter):
    lnkvik -= gi
    k += 1
    kvik = np.exp(lnkvik)
    ni = kvik * yi
    xi = ni / ni.sum()
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkvik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tTPD = %s\n\tP = %s Pa',
      k, kvik, gnorm, TPD, Pk,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Pk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, k,
    )
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation using the SS-method terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa",
      eos.name, P0, T, yi, Pmin, Pmax,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PsatPT_qnss(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Pmax: ScalarType = 1e8,
  Pmin: ScalarType = 1.,
  upper: bool = True,
) -> SatResult:
  """QNSS-method for the saturation pressure calculation using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of nonlinear equations.
  For the details of the QNSS-method see: 10.1016/0378-3812(84)80013-8
  and 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Pmax: float
    The upper bound for the TPD-equation solver. Default is `1e8` [Pa].

  Pmin: float
    The lower bound for the TPD-equation solver. Default is `1.0` [Pa].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    'Saturation pressure calculation using the QNSS-method:\n'
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa',
    P0, T, yi, Pmin, Pmax,
  )
  solverTPDeq = partial(_solveTPDeqPT_forP, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Pmin0=Pmin, Pmax0=Pmax,
                        increasing=upper)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  xi = ni / ni.sum()
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  gi = np.log(kvik) + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  TPD = ni.dot(gi)
  lmbd = 1.
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tTPD = %s\n\tP = %s Pa',
    k, kvik, gnorm, TPD, Pk,
  )
  while ((gnorm > tol) or (np.abs(TPD) > tol_tpd)) and (k < maxiter):
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
    ni = kvik * yi
    xi = ni / ni.sum()
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkvik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tTPD = %s\n\tPk = %s',
      k, kvik, gnorm, TPD, Pk,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Pk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, k,
    )
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation using the QNSS-method terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa",
      eos.name, P0, T, yi, Pmin, Pmax,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _improveP0_newt(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  yti: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Pmax0: ScalarType = 1e8,
  Pmin0: ScalarType = 1.,
  increasing: bool = True,
) -> tuple[ScalarType, VectorType, VectorType,
           ScalarType, ScalarType, ScalarType]:
  """Improves initial guess of the saturation pressure by solving
  the TPD-equation using the PT-based equations of state for pressure
  at a constant temperature. The TPD-equation is the equation of
  equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation. The
  algorithm is the same as for the `_solveTPDeqPT_forP` but it differs
  in returned result.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  yti: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Pmax0: float
    The initial upper bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `1e8` [Pa].

  Pmin0: float
    The initial lower bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `1.0` [Pa].

  increasing: bool
    A flag that indicates if the TPD-equation vs pressure is the
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation pressure,
  - an array of the natural logarithms of fugacity coefficients of
    components in the mixture,
  - compressibility factor of the mixture,
  - an array of partial derivatives of logarithms of the fugacity
    coefficients of components with respect to pressure.
  """
  k = 0
  Pk = P0
  Pmin = Pmin0
  Pmax = Pmax0
  lnkvi = np.log(yti / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphiti, Zt, dlnphitidP = eos.getPT_lnphii_Z_dP(Pk, T, yti)
  TPD = yti.dot(lnkvi + lnphiti - lnphiyi)
  while k < maxiter and np.abs(TPD) > tol:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Pmin = Pk
    else:
      Pmax = Pk
    dTPDdP = yti.dot(dlnphitidP - dlnphiyidP)
    dP = - TPD / dTPDdP
    Pkp1 = Pk + dP
    if Pkp1 > Pmax or Pkp1 < Pmin:
      Pkp1 = .5 * (Pmin + Pmax)
      dP = Pkp1 - Pk
    if np.abs(dP) < 1e-8:
      break
    Pk = Pkp1
    k += 1
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphiti, Zt, dlnphitidP = eos.getPT_lnphii_Z_dP(Pk, T, yti)
    TPD = yti.dot(lnkvi + lnphiti - lnphiyi)
  return Pk, lnphiyi, Zy, dlnphiyidP



def _PsatPT_newtA(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  Pmax: ScalarType = 1e8,
  Pmin: ScalarType = 1.,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation pressure by solving a system of
  nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the sum of mole numbers of components in
  the trial phase equals unity.

  The formulation of the system of nonlinear equations is based on the
  paper of M.L. Michelsen: 10.1016/0378-3812(80)80001-X.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dnj_dP(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Pmax: float
    The pressure upper bound. Default is `1e8` [Pa].

  Pmin: float
    The pressure lower bound. Default is `1.0` [Pa].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. The TPD-equation is the
    equation of equality to zero of the tangent-plane distance, which
    determines the second phase appearance or disappearance. This
    parameter is used by the algorithm of the saturation pressure
    initial guess improvement. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation pressure initial guess
    improvement. Default is `8`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. This parameter is used by the algorithm of
    the saturation pressure initial guess improvement.
    Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    "Saturation pressure calculation using Newton's method (A-form):\n"
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa',
    P0, T, yi, Pmin, Pmax,
  )
  Nc = eos.Nc
  J = np.empty(shape=(Nc + 1, Nc + 1))
  J[-1,-1] = 0.
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphiyi, Zy, dlnphiyidP = _improveP0_newt(P0, T, yi, xi, eos, tol_tpd,
                                                maxiter_tpd, Pmax, Pmin,
                                                upper)
  lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(Pk, T, xi,
                                                                   n)
  gi[:Nc] = lnkvik + lnphixi - lnphiyi
  gi[-1] = n - 1.
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tP = %s Pa\n\tgnorm = %s',
    k, kvik, Pk, gnorm,
  )
  while (gnorm > tol) and (k < maxiter):
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = ni
    J[:Nc,-1] = Pk * (dlnphixidP - dlnphiyidP)
    try:
      dlnkvilnP = linsolver(J, -gi)
    except:
      dlnkvilnP = -gi
    k += 1
    lnkvik += dlnkvilnP[:-1]
    Pkp1 = Pk * np.exp(dlnkvilnP[-1])
    if Pkp1 > Pmax:
      Pk = .5 * (Pk + Pmax)
    elif Pkp1 < Pmin:
      Pk = .5 * (Pmin + Pk)
    else:
      Pk = Pkp1
    kvik = np.exp(lnkvik)
    ni = kvik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(Pk, T,
                                                                     xi, n)
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    gi[:Nc] = lnkvik + lnphixi - lnphiyi
    gi[-1] = n - 1.
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tP = %s Pa\n\tgnorm = %s',
      k, kvik, Pk, gnorm,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Pk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, k,
    )
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation using Newton's method (A-form) "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa",
      eos.name, P0, T, yi, Pmin, Pmax,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PsatPT_newtB(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  Pmax: ScalarType = 1e8,
  Pmin: ScalarType = 1.,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation pressure by solving a system of
  nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the tangent-plane distance is zero at
  the saturation curve.

  The formulation of the system of nonlinear equations is based on the
  paper of L.X. Nghiem and Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dnj_dP(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to pressure.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Pmax: float
    The pressure upper bound. Default is `1e8` [Pa].

  Pmin: float
    The pressure lower bound. Default is `1.0` [Pa].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. The TPD-equation is the
    equation of equality to zero of the tangent-plane distance, which
    determines the second phase appearance or disappearance. This
    parameter is used by the algorithm of the saturation pressure
    initial guess improvement. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation pressure initial guess
    improvement. Default is `8`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. This parameter is used by the algorithm of
    the saturation pressure initial guess improvement.
    Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    "Saturation pressure calculation using Newton's method (B-form):\n"
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa',
    P0, T, yi, Pmin, Pmax,
  )
  Nc = eos.Nc
  J = np.empty(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphiyi, Zy, dlnphiyidP = _improveP0_newt(P0, T, yi, xi, eos, tol_tpd,
                                                maxiter_tpd, Pmax, Pmin,
                                                upper)
  lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(Pk, T, xi,
                                                                   n)
  gi[:Nc] = np.log(xi / yi) + lnphixi - lnphiyi
  gi[-1] = xi.dot(gi[:Nc])
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tP = %s Pa\n\tgnorm = %s',
    k, kvik, Pk, gnorm,
  )
  while (gnorm > tol) and (k < maxiter):
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = xi * (gi[:-1] - xi.dot(gi[:-1]))
    J[:Nc,-1] = Pk * (dlnphixidP - dlnphiyidP)
    J[-1,-1] = xi.dot(J[:Nc,-1])
    try:
      dlnkvilnP = linsolver(J, -gi)
    except:
      dlnkvilnP = -gi
    k += 1
    lnkvik += dlnkvilnP[:-1]
    Pkp1 = Pk * np.exp(dlnkvilnP[-1])
    if Pkp1 > Pmax:
      Pk = .5 * (Pk + Pmax)
    elif Pkp1 < Pmin:
      Pk = .5 * (Pmin + Pk)
    else:
      Pk = Pkp1
    kvik = np.exp(lnkvik)
    ni = kvik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(Pk, T,
                                                                     xi, n)
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    gi[:Nc] = np.log(xi / yi) + lnphixi - lnphiyi
    gi[-1] = xi.dot(gi[:Nc])
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tP = %s Pa\n\tgnorm = %s',
      k, kvik, Pk, gnorm,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Pk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, k,
    )
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation using Newton's method (B-form) "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa",
      eos.name, P0, T, yi, Pmin, Pmax,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _PsatPT_newtC(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Pmax: ScalarType = 1e8,
  Pmin: ScalarType = 1.,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation pressure by solving a system of
  nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the tangent-plane distance is zero at
  the saturation curve. The TPD-equation is solved in the inner
  while-loop.

  The formulation of the system of nonlinear equations is based on the
  paper of L.X. Nghiem and Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to pressure.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Pmax: float
    The pressure upper bound. Default is `1e8` [Pa].

  Pmin: float
    The pressure lower bound. Default is `1.0` [Pa].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    "Saturation pressure calculation using Newton's method (C-form):\n"
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa',
    P0, T, yi, Pmin, Pmax,
  )
  solverTPDeq = partial(_solveTPDeqPT_forP, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Pmin0=Pmin, Pmax0=Pmax,
                        increasing=upper)
  I = np.eye(eos.Nc)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  _, _, dlnphixidnj = eos.getPT_lnphii_Z_dnj(Pk, T, xi, n)
  gi = lnkvik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  TPD = ni.dot(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tP = %s Pa\n\tgnorm = %s\n\tTPD = %s',
    k, kvik, Pk, gnorm, TPD,
  )
  while ((gnorm > tol) or (np.abs(TPD) > tol_tpd)) and (k < maxiter):
    J = I + ni * dlnphixidnj
    try:
      dlnkvi = linsolver(J, -gi)
    except:
      dlnkvi = -gi
    k += 1
    lnkvik += dlnkvi
    kvik = np.exp(lnkvik)
    ni = kvik * yi
    n = ni.sum()
    xi = ni / n
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    _, _, dlnphixidnj = eos.getPT_lnphii_Z_dnj(Pk, T, xi, n)
    gi = lnkvik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tP = %s Pa\n\tgnorm = %s\n\tTPD = %s',
      k, kvik, Pk, gnorm, TPD
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Pk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation pressure for T = %s K, yi = %s:\n\t'
      'Ps = %s Pa\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      T, yi, Pk, xi, gnorm, k,
    )
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation using Newton's method (C-form) "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa",
      eos.name, P0, T, yi, Pmin, Pmax,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


class TsatPT(object):
  """Saturation temperature calculation.

  Performs saturation temperature calculation using PT-based equations
  of state.

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
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor.

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_lnphii_Z_dnj_dT(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  method: str
    Type of the solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Currently raises `NotImplementedError`),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Currently raises `NotImplementedError`),
    - `'ss-newton'` (Currently raises `NotImplementedError`),
    - `'qnss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  step: float
    To specify the confidence interval for the saturation temperature
    calculation, the preliminary search is performed. This parameter
    regulates the step of this search in fraction units. For example,
    if it is necessary to find the upper bound of the confidence
    interval, then the next value of temperature will be calculated from
    the previous one using the formula: `Tnext = Tprev * (1. + step)`.
    Default is `0.1`.

  lowerlimit: float
    During the preliminary search, the temperature can not drop below
    the lower limit. Otherwise, the `ValueError` will be rised.
    Default is `173.15` [K].

  upperlimit: float
    During the preliminary search, the temperature can not exceed the
    upper limit. Otherwise, the `ValueError` will be rised.
    Default is `973.15` [K].

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval of the saturation temperature. This dictionary is used to
    specify arguments for the stability test procedure. Default is an
    empty dictionary.

  kwargs: dict
    Other arguments for a Tsat-solver. It may contain such arguments
    as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others depending
    on the selected solver.

  Methods
  -------
  run(P, yi, T0) -> SatResult
    This method performs the saturation temperature calculation for
    given pressure `P: float` in [Pa], composition `yi: ndarray` of `Nc`
    components and the initial guess `T0: float` in [K]. This method
    returns saturation temperature calculation results as an instance of
    `SatResult`.
  """
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'ss',
    upper: bool = True,
    step: ScalarType = 0.1,
    lowerlimit: ScalarType = 173.15,
    upperlimit: ScalarType = 973.15,
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.upper = upper
    self.step = step
    self.lowerlimit = lowerlimit
    self.upperlimit = upperlimit
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.tsatsolver = partial(_TsatPT_ss, eos=eos, upper=upper, **kwargs)
    elif method == 'qnss':
      self.tsatsolver = partial(_TsatPT_qnss, eos=eos, upper=upper, **kwargs)
    elif method == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for the saturation temperature calculation is not '
        'implemented yet.'
      )
    elif method == 'newton':
      self.tsatsolver = partial(_TsatPT_newtA, eos=eos, upper=upper, **kwargs)
    elif method == 'newton-b':
      self.tsatsolver = partial(_TsatPT_newtB, eos=eos, upper=upper, **kwargs)
    elif method == 'newton-c':
      self.tsatsolver = partial(_TsatPT_newtC, eos=eos, upper=upper, **kwargs)
    elif method == 'ss-newton':
      raise NotImplementedError(
        'The SS-Newton method for the saturation temperature calculation is '
        'not implemented yet.'
      )
    elif method == 'qnss-newton':
      raise NotImplementedError(
        'The QNSS-Newton method for the saturation temperature calculation '
        'is not implemented yet.'
      )
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: ScalarType,
    yi: VectorType,
    T0: ScalarType,
  ) -> SatResult:
    """Performs the saturation temperature calculation for a given
    pressure and composition. To improve an initial guess, the
    preliminary search is performed.

    Parameters
    ----------
    P: float
      Pressure of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    T0: float
      Initial guess of the saturation temperature [Pa].

    Returns
    -------
    Saturation temperature calculation results as an instance of the
    `SatResult`. Important attributes are:

    - `P` the saturation pressure in [Pa],
    - `T` the saturation temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase,
    - `success` a boolean flag indicating if the calculation completed
      successfully.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    stab = self.stabsolver.run(P, T0, yi)
    logger.debug(
      'For the initial guess T0 = %s K, the one-phase state is stable: %s',
      T0, stab.stable,
    )
    if stab.stable and self.upper:
      logger.debug(
        'Finding the two-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Tmax = T0
      c = 1. - self.step
      Tmin = c * T0
      stabmin = self.stabsolver.run(P, Tmin, yi)
      logger.debug(
        'Tmin = %s K, the one-phase state is stable: %s',
        Tmin, stabmin.stable,
      )
      if stabmin.stable:
        Tmax = Tmin
      while stabmin.stable and Tmin > self.lowerlimit:
        Tmin *= c
        stabmin = self.stabsolver.run(P, Tmin, yi)
        logger.debug(
          'Tmin = %s K, the one-phase state is stable: %s',
          Tmin, stabmin.stable,
        )
        if stabmin.stable:
          Tmax = Tmin
      if Tmin < self.lowerlimit:
        raise ValueError(
          'The two-phase region was not identified. It could be because of \n'
          'its narrowness or absence. Try to change the value of T0 or \n'
          'stability test parameters using the `stabkwargs` argument of \n'
          'this class. It also might be helpful to reduce the value of the \n'
          'argument `step`.'
        )
      else:
        T0 = Tmin
        stab = stabmin
    elif not stab.stable and self.upper:
      logger.debug(
        'Finding the one-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Tmin = T0
      c = 1. + self.step
      Tmax = c * T0
      stabmax = self.stabsolver.run(P, Tmax, yi)
      logger.debug(
        'Tmax = %s K, the one-phase state is stable: %s',
        Tmax, stabmax.stable,
      )
      if not stabmax.stable:
        Tmin = Tmax
      while not stabmax.stable and Tmax < self.upperlimit:
        Tmax *= c
        stabmax = self.stabsolver.run(P, Tmax, yi)
        logger.debug(
          'Tmax = %s K, the one-phase state is stable: %s',
          Tmax, stabmax.stable,
        )
        if not stabmax.stable:
          Tmin = Tmax
          stab = stabmax
      if Tmax > self.upperlimit:
        raise ValueError(
          'The one-phase region was not identified. Try to change the \n'
          'value of T0 and `upperlimit` parameter of this class.'
        )
      else:
        T0 = Tmin
    elif stab.stable and not self.upper:
      logger.debug(
        'Finding the two-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Tmin = T0
      c = 1. + self.step
      Tmax = c * T0
      stabmax = self.stabsolver.run(P, Tmax, yi)
      logger.debug(
        'Tmax = %s K, the one-phase state is stable: %s',
        Tmax, stabmax.stable,
      )
      while stabmax.stable and Tmax < self.upperlimit:
        Tmax *= c
        stabmax = self.stabsolver.run(P, Tmax, yi)
        logger.debug(
          'Tmax = %s K, the one-phase state is stable: %s',
          Tmax, stabmax.stable,
        )
        if stabmax.stable:
          Tmin = Tmax
      if Tmax > self.upperlimit:
        raise ValueError(
          'The two-phase region was not identified. It could be because of \n'
          'its narrowness or absence. Try to change the value of T0 or \n'
          'stability test parameters using the `stabkwargs` argument of \n'
          'this class. It also might be helpful to reduce the value of the \n'
          'argument `step`.'
        )
      else:
        T0 = Tmax
        stab = stabmax
    else:
      logger.debug(
        'Finding the one-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Tmax = T0
      c = 1. - self.step
      Tmin = c * T0
      stabmin = self.stabsolver.run(P, Tmin, yi)
      logger.debug(
        'Tmin = %s K, the one-phase state is stable: %s',
        Tmin, stabmin.stable,
      )
      while not stabmin.stable and Tmin > self.lowerlimit:
        Tmin *= c
        stabmin = self.stabsolver.run(P, Tmin, yi)
        logger.debug(
          'Tmin = %s K, the one-phase state is stable: %s',
          Tmin, stabmin.stable,
        )
        if not stabmin.stable:
          Tmax = Tmin
          stab = stabmin
      if Tmin < self.lowerlimit:
        raise ValueError(
          'The one-phase region was not identified. Try to change the \n'
          'value of T0 and `lowerlimit` parameter of this class.'
        )
      else:
        T0 = Tmax
    return self.tsatsolver(P, T0, yi, stab.kvji[0], Tmin=Tmin, Tmax=Tmax)


def _solveTPDeqPT_forT(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  yti: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Tmax0: ScalarType = 973.15,
  Tmin0: ScalarType = 173.15,
  increasing: bool = True,
) -> tuple[ScalarType, VectorType, VectorType,
           ScalarType, ScalarType, ScalarType]:
  """Solves the TPD-equation using the PT-based equations of state for
  temperature at a constant pressure. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation.

  Parameters
  ----------
  T0: float
    Initial guess of the saturation temperature [K].

  P: float
    Pressure of a mixture [Pa].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  yti: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Tmax0: float
    The initial upper bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `973.15` [K].

  Tmin0: float
    The initial lower bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `173.15` [K].

  increasing: bool
    A flag that indicates if the TPD-equation vs temperature is the
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation temperature,
  - an array of the natural logarithms of fugacity coefficients of
    components in the trial phase,
  - the same for the initial composition of the mixture,
  - compressibility factors for both mixtures.
  - value of the tangent-plane distance.
  """
  k = 0
  Tmin = Tmin0
  Tmax = Tmax0
  Tk = T0
  lnkvi = np.log(yti / yi)
  lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
  lnphiti, Zt, dlnphitidT = eos.getPT_lnphii_Z_dT(P, Tk, yti)
  TPD = yti.dot(lnkvi + lnphiti - lnphiyi)
  while k < maxiter and np.abs(TPD) > tol:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Tmin = Tk
    else:
      Tmax = Tk
    dTPDdT = yti.dot(dlnphitidT - dlnphiyidT)
    dT = - TPD / dTPDdT
    Tkp1 = Tk + dT
    if Tkp1 > Tmax or Tkp1 < Tmin:
      Tkp1 = .5 * (Tmin + Tmax)
      dT = Tkp1 - Tk
    if np.abs(dT) < 1e-8:
      break
    Tk = Tkp1
    k += 1
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    lnphiti, Zt, dlnphitidT = eos.getPT_lnphii_Z_dT(P, Tk, yti)
    TPD = yti.dot(lnkvi + lnphiti - lnphiyi)
  return Tk, lnphiti, lnphiyi, Zt, Zy, TPD


def _TsatPT_ss(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tmax: ScalarType = 973.15,
  Tmin: ScalarType = 173.15,
  upper: bool = True,
) -> SatResult:
  """Successive substitution (SS) method for the saturation temperature
  calculation using a PT-based equation of state.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Tmax: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  Tmin: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    'Saturation temperature calculation using the SS-method:\n'
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K',
    P, T0, yi, Tmin, Tmax,
  )
  solverTPDeq = partial(_solveTPDeqPT_forT, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tmin0=Tmin, Tmax0=Tmax,
                        increasing=upper)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  xi = ni / ni.sum()
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkvik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tTPD = %s\n\tT = %s K',
    k, kvik, gnorm, TPD, Tk,
  )
  while ((gnorm > tol) or (np.abs(TPD) > tol_tpd)) and (k < maxiter):
    lnkvik -= gi
    k += 1
    kvik = np.exp(lnkvik)
    ni = kvik * yi
    xi = ni / ni.sum()
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkvik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tTPD = %s\n\tT = %s K',
      k, kvik, gnorm, TPD, Tk,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, k,
    )
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation using the SS-method terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K",
      eos.name, P, T0, yi, Tmin, Tmax,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TsatPT_qnss(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tmax: ScalarType = 973.15,
  Tmin: ScalarType = 173.15,
  upper: bool = True,
) -> SatResult:
  """Quasi-Newton Successive Substitution (QNSS) method for the
  saturation temperature calculation using a PT-based equation
  of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of nonlinear equations.
  For the details of the QNSS-method see: 10.1016/0378-3812(84)80013-8
  and 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Tmax: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  Tmin: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    'Saturation temperature calculation using the QNSS-method:\n'
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K',
    P, T0, yi, Tmin, Tmax,
  )
  solverTPDeq = partial(_solveTPDeqPT_forT, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tmin0=Tmin, Tmax0=Tmax,
                        increasing=upper)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  xi = ni / ni.sum()
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkvik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  TPD = ni.dot(gi)
  lmbd = 1.
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tTPD = %s\n\tT = %s K',
    k, kvik, gnorm, TPD, Tk,
  )
  while ((gnorm > tol) or (np.abs(TPD) > tol_tpd)) and (k < maxiter):
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
    ni = kvik * yi
    xi = ni / ni.sum()
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkvik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tTPD = %s\n\tT = %s K',
      k, kvik, gnorm, TPD, Tk,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, k,
    )
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation using the QNSS-method terminates "
      "unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K",
      eos.name, P, T0, yi, Tmin, Tmax,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _improveT0_newt(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  yti: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Tmax0: ScalarType = 973.15,
  Tmin0: ScalarType = 173.15,
  increasing: bool = True,
) -> tuple[ScalarType, VectorType, ScalarType, VectorType]:
  """Improves initial guess of the saturation temperature by solving
  the TPD-equation using the PT-based equations of state for temperature
  at a constant pressure. The TPD-equation is the equation of equality
  to zero of the tangent-plane distance, which determines the phase
  appearance or disappearance. A combination of the bisection method
  with Newton's method is used to solve the TPD-equation. The algorithm
  is the same as for the `_solveTPDeqPT_forT` but it differs in returned
  result.

  Parameters
  ----------
  T0: float
    Initial guess of the saturation temperature [K].

  P: float
    Pressure of a mixture [Pa].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  yti: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

  maxiter: int
    The maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Tmax0: float
    The initial upper bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `973.15` [K].

  Tmin0: float
    The initial lower bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `173.15` [K].

  increasing: bool
    A flag that indicates if the TPD-equation vs temperature is the
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation temperature,
  - an array of the natural logarithms of fugacity coefficients of
    components in the mixture,
  - compressibility factors of the mixture,
  - an array of partial derivatives of logarithms of the fugacity
    coefficients of components with respect to temperature.
  """
  k = 0
  Tmin = Tmin0
  Tmax = Tmax0
  Tk = T0
  lnkvi = np.log(yti / yi)
  lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
  lnphiti, Zt, dlnphitidT = eos.getPT_lnphii_Z_dT(P, Tk, yti)
  TPD = yti.dot(lnkvi + lnphiti - lnphiyi)
  while k < maxiter and np.abs(TPD) > tol:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Tmin = Tk
    else:
      Tmax = Tk
    dTPDdT = yti.dot(dlnphitidT - dlnphiyidT)
    dT = - TPD / dTPDdT
    Tkp1 = Tk + dT
    if Tkp1 > Tmax or Tkp1 < Tmin:
      Tkp1 = .5 * (Tmin + Tmax)
      dT = Tkp1 - Tk
    if np.abs(dT) < 1e-8:
      break
    Tk = Tkp1
    k += 1
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    lnphiti, Zt, dlnphitidT = eos.getPT_lnphii_Z_dT(P, Tk, yti)
    TPD = yti.dot(lnkvi + lnphiti - lnphiyi)
  return Tk, lnphiyi, Zy, dlnphiyidT


def _TsatPT_newtA(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  Tmax: ScalarType = 1e8,
  Tmin: ScalarType = 1.,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation temperature by solving a system
  of nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the sum of mole numbers of components in
  the trial phase equals unity.

  The formulation of the system of nonlinear equations is based on the
  paper of M.L. Michelsen: 10.1016/0378-3812(80)80001-X.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dnj_dT(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Tmax: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  Tmin: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. The TPD-equation is the
    equation of equality to zero of the tangent-plane distance, which
    determines the second phase appearance or disappearance. This
    parameter is used by the algorithm of the saturation temperature
    initial guess improvement. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation temperature initial guess
    improvement. Default is `8`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. This parameter is used by the algorithm of
    the saturation temperature initial guess improvement.
    Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    "Saturation temperature calculation using Newton's method (A-form):\n"
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K',
    P, T0, yi, Tmin, Tmax,
  )
  Nc = eos.Nc
  J = np.empty(shape=(Nc + 1, Nc + 1))
  J[-1,-1] = 0.
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphiyi, Zy, dlnphiyidT = _improveT0_newt(P, T0, yi, xi, eos, tol_tpd,
                                                maxiter_tpd, Tmax, Tmin,
                                                upper)
  lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk, xi,
                                                                   n)
  gi[:Nc] = lnkvik + lnphixi - lnphiyi
  gi[-1] = n - 1.
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tT = %s K\n\tgnorm = %s',
    k, kvik, Tk, gnorm,
  )
  while (gnorm > tol) and (k < maxiter):
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = ni
    J[:Nc,-1] = Tk * (dlnphixidT - dlnphiyidT)
    try:
      dlnkvilnT = linsolver(J, -gi)
    except:
      dlnkvilnT = -gi
    k += 1
    lnkvik += dlnkvilnT[:-1]
    Tkp1 = Tk * np.exp(dlnkvilnT[-1])
    if Tkp1 > Tmax:
      Tk = .5 * (Tk + Tmax)
    elif Tkp1 < Tmin:
      Tk = .5 * (Tmin + Tk)
    else:
      Tk = Tkp1
    kvik = np.exp(lnkvik)
    ni = kvik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk,
                                                                     xi, n)
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    gi[:Nc] = lnkvik + lnphixi - lnphiyi
    gi[-1] = n - 1.
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tT = %s K\n\tgnorm = %s',
      k, kvik, Tk, gnorm,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, k,
    )
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation temperature calculation using Newton's method (A-form) "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K",
      eos.name, P, T0, yi, Tmin, Tmax,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TsatPT_newtB(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  Tmax: ScalarType = 1e8,
  Tmin: ScalarType = 1.,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation temperature by solving a system
  of nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the tangent-plane distance is zero at
  the saturation curve.

  The formulation of the system of nonlinear equations is based on the
  paper of L.X. Nghiem and Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dnj_dT(P, T, yi, n) -> tuple[ndarray, float,
                                                   ndarray, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to temperature.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Tmax: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  Tmin: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. The TPD-equation is the
    equation of equality to zero of the tangent-plane distance, which
    determines the second phase appearance or disappearance. This
    parameter is used by the algorithm of the saturation temperature
    initial guess improvement. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation temperature initial guess
    improvement. Default is `8`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. This parameter is used by the algorithm of
    the saturation temperature initial guess improvement.
    Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    "Saturation temperature calculation using Newton's method (B-form):\n"
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K',
    P, T0, yi, Tmin, Tmax,
  )
  Nc = eos.Nc
  J = np.empty(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphiyi, Zy, dlnphiyidT = _improveT0_newt(P, T0, yi, xi, eos, tol_tpd,
                                                maxiter_tpd, Tmax, Tmin,
                                                upper)
  lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk, xi,
                                                                   n)
  gi[:Nc] = np.log(xi / yi) + lnphixi - lnphiyi
  gi[-1] = xi.dot(gi[:Nc])
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tT = %s K\n\tgnorm = %s',
    k, kvik, Tk, gnorm,
  )
  while (gnorm > tol) and (k < maxiter):
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = ni
    J[:Nc,-1] = Tk * (dlnphixidT - dlnphiyidT)
    J[-1,-1] = xi.dot(J[:Nc,-1])
    try:
      dlnkvilnT = linsolver(J, -gi)
    except:
      dlnkvilnT = -gi
    k += 1
    lnkvik += dlnkvilnT[:-1]
    Tkp1 = Tk * np.exp(dlnkvilnT[-1])
    if Tkp1 > Tmax:
      Tk = .5 * (Tk + Tmax)
    elif Tkp1 < Tmin:
      Tk = .5 * (Tmin + Tk)
    else:
      Tk = Tkp1
    kvik = np.exp(lnkvik)
    ni = kvik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk,
                                                                     xi, n)
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    gi[:Nc] = np.log(xi / yi) + lnphixi - lnphiyi
    gi[-1] = xi.dot(gi[:Nc])
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tT = %s K\n\tgnorm = %s',
      k, kvik, Tk, gnorm,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, k,
    )
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation temperature calculation using Newton's method (B-form) "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K",
      eos.name, P, T0, yi, Tmin, Tmax,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)


def _TsatPT_newtC(
  P: ScalarType,
  T0: ScalarType,
  yi: VectorType,
  kvi0: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 20,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Tmax: ScalarType = 1e8,
  Tmin: ScalarType = 1.,
  linsolver: Callable[[MatrixType, VectorType], VectorType] = np.linalg.solve,
  upper: bool = True,
) -> SatResult:
  """This function calculates saturation temperature by solving a system
  of nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirement that the tangent-plane distance is zero at
  the saturation curve. The TPD-equation is solved in the inner
  while-loop.

  The formulation of the system of nonlinear equations is based on the
  paper of L.X. Nghiem and Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P: float
    Pressure of a mixture [Pa].

  T0: float
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: ndarray, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P, T, yi) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K] and phase composition,
      this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - an array of shape `(Nc,)` of partial derivatives of
        logarithms of the fugacity coefficients of components
        with respect to temperature.

    - `getPT_lnphii_Z_dnj(P, T, yi, n) -> tuple[ndarray, float, ndarray]`
      For a given pressure [Pa], temperature [K], phase composition and
      phase mole number [mol] this method should return a tuple of:

      - an array of shape `(Nc,)` of logarithms of the fugacity
        coefficients of components,
      - the phase compressibility factor,
      - a matrix of shape `(Nc, Nc)` of partial derivatives of
        logarithms of the fugacity coefficients of components with
        respect to their mole numbers.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: float
    Terminate the solver successfully if the norm of an array of
    nonlinear equations is less than `tol`. Default is `1e-5`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. Default is `1e-6`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `8`.

  Tmax: float
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  Tmin: float
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  linsolver: Callable[[ndarray, ndarray], ndarray]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation bound or the lower saturation bound.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries. Default is `True`.

  Returns
  -------
  Saturation temperature calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the saturation pressure in [Pa],
  - `T` the saturation temperature in [K],
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase,
  - `success` a boolean flag indicating if the calculation
    completed successfully.
  """
  logger.debug(
    "Saturation temperature calculation using Newton's method (C-form):\n"
    '\tP = %s Pa\n\tT0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K',
    P, T0, yi, Tmin, Tmax,
  )
  solverTPDeq = partial(_solveTPDeqPT_forT, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tmin0=Tmin, Tmax0=Tmax,
                        increasing=upper)
  I = np.eye(eos.Nc)
  k = 0
  kvik = kvi0
  lnkvik = np.log(kvik)
  ni = kvik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  _, _, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk, xi, n)
  gi = lnkvik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tT = %s K\n\tgnorm = %s',
    k, kvik, Tk, gnorm,
  )
  while (gnorm > tol) and (k < maxiter):
    J = I + ni * dlnphixidnj
    try:
      dlnkvi = linsolver(J, -gi)
    except:
      dlnkvi = -gi
    k += 1
    lnkvik += dlnkvi
    kvik = np.exp(lnkvik)
    ni = kvik * yi
    n = ni.sum()
    xi = ni / n
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    _, _, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(P, Tk, xi, n)
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    gi = lnkvik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tT = %s K\n\tgnorm = %s',
      k, kvik, Tk, gnorm,
    )
  if (gnorm < tol) & (np.isfinite(kvik).all()) & (np.isfinite(Tk)):
    rhoy = yi.dot(eos.mwi) / Zy
    rhox = xi.dot(eos.mwi) / Zx
    if rhoy < rhox:
      yji = np.vstack([yi, xi])
      Zj = np.array([Zy, Zx])
      lnphiji = np.vstack([lnphiyi, lnphixi])
    else:
      yji = np.vstack([xi, yi])
      Zj = np.array([Zx, Zy])
      lnphiji = np.vstack([lnphixi, lnphiyi])
    logger.info(
      'Saturation temperature for P = %s Pa, yi = %s:\n\t'
      'Ts = %s K\n\tyti = %s\n\tgnorm = %s\n\tNiter = %s',
      P, yi, Tk, xi, gnorm, k,
    )
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation temperature calculation using Newton's method (C-form) "
      "terminates unsuccessfully. EOS: %s. Parameters:"
      "\n\tP = %s Pa, T0 = %s K\n\tyi = %s\n\tTmin = %s K\n\tTmax = %s K",
      eos.name, P, T0, yi, Tmin, Tmax,
    )
    return SatResult(P=P, T=Tk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)
