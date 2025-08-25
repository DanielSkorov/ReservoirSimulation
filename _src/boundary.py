import logging

from functools import (
  partial,
)

import numpy as np

from utils import (
  pyrqi,
)

from typing import (
  Callable,
  Iterable,
)

from custom_types import (
  Scalar,
  Vector,
  Matrix,
  Eos,
  SolutionNotFoundError,
)

from stability import (
  StabResult,
  stabilityPT,
)

from flash import (
  FlashResult,
  flash2pPT,
)


logger = logging.getLogger('bound')


class CritEosVT(Eos):
  Tci: Vector

  def getVT_lnfi_dnj(
    self,
    V: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar,
  ) -> tuple[Vector, Matrix]: ...

  def getVT_d3F(
    self,
    V: Scalar,
    T: Scalar,
    yi: Vector,
    zti: Vector,
    n: Scalar,
  ) -> Scalar: ...

  def getVT_vmin(
    self,
    T: Scalar,
    yi: Vector,
  ) -> Scalar: ...


class PsatEosPT(Eos):

  def getPT_kvguess(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, ...]: ...

  def getPT_lnphii_Z(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar]: ...

  def getPT_lnphii_Z_dP(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar, Vector]: ...

  def getPT_lnphii_Z_dnj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar,
  ) -> tuple[Vector, Scalar, Matrix]: ...

  def getPT_lnphii_Z_dnj_dP(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar,
  ) -> tuple[Vector, Scalar, Matrix, Vector]: ...


class TsatEosPT(Eos):

  def getPT_kvguess(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, ...]: ...

  def getPT_lnphii_Z(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar]: ...

  def getPT_lnphii_Z_dT(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar, Vector]: ...

  def getPT_lnphii_Z_dnj(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar,
  ) -> tuple[Vector, Scalar, Matrix]: ...

  def getPT_lnphii_Z_dnj_dT(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    n: Scalar,
  ) -> tuple[Vector, Scalar, Matrix, Vector]: ...


class PmaxEosPT(PsatEosPT):

  def getPT_lnphii_Z_dT_d2T(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar, Vector, Vector]: ...


class TmaxEosPT(TsatEosPT):

  def getPT_lnphii_Z_dP_d2P(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar, Vector, Vector]: ...


class Env2pEosPT(PsatEosPT):

  def getPT_lnphii_Z_dP_dT_dyj(
    self,
    P: Scalar,
    T:  Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar, Vector, Vector, Matrix]: ...

  def getPT_lnphii_Z_dP_dT(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
  ) -> tuple[Vector, Scalar, Vector, Vector]: ...


def getVT_Tspinodal(
  V: Scalar,
  yi: Vector,
  eos: CritEosVT,
  T0: None | Scalar = None,
  zeta0i: None | Vector = None,
  multdT0: Scalar = 1e-5,
  tol: Scalar = 1e-5,
  maxiter: int = 25,
) -> tuple[Scalar, Vector]:
  """Calculates the spinodal temperature for a given volume
  and composition of the mixture.

  Parameters
  ----------
  V: Scalar
    Volume of the mixture [m3].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  eos: CritEosVT
    An initialized instance of a VT-based equation of state. Must have
    the following methods:

    - `getVT_lnfi_dnj(V: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Matrix]`
      For a given volume [m3], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacities of components as a `Vector` of
        shape `(Nc,)`,
      - partial derivatives of logarithms of fugacities of components
        with respect to their mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  T0: None | Scalar
    An initial guess of the spinodal temperature [K]. If it equals
    `None`, a pseudocritical temperature multiplied by a factor `1.3`
    will be used.

  zeta0i: None | Vector, shape (Nc,)
    An initial guess for the eigenvector corresponding to the lowest
    eigenvalue of the matrix of second partial derivatives of the
    Helmholtz energy function with respect to component mole
    numbers. If it equals `None` then `yi` will be used instead.

  multdT0: Scalar
    A multiplier used to compute the temperature shift to estimate
    the partial derivative of the lowest eigenvalue with respect to
    temperature at the first iteration. Default is `1e-5`.

  tol: Scalar
    Terminate successfully if the absolute value of the lowest
    eigenvalue is less then `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of iterations. Default is `25`.

  Returns
  -------
  A tuple of the spinodal temperature and the eigenvector.

  Raises
  ------
  `SolutionNotFoundError` if the solution was not found.
  """
  logger.debug('Calculating the spinodal temperature.')
  Nc = eos.Nc
  logger.debug(
    '%3s%9s' + Nc * '%9s' + '%9s%11s',
    'Nit', 'eigval', *['zeta%s' % s for s in range(Nc)], 'T, K', 'dT, K',
  )
  tmpl = '%3s%9.4f' + Nc * '%9.4f' + '%9.2f%11.2e'
  k = 0
  if T0 is None:
    Tk = 1.3 * yi.dot(eos.Tci)
  else:
    Tk = T0
  if zeta0i is None:
    zeta0i = yi
  Q = eos.getVT_lnfi_dnj(V, Tk, yi)[1]
  zetai, lmbdk = pyrqi(Q, zeta0i)
  dT = multdT0 * Tk
  QdT = eos.getVT_lnfi_dnj(V, Tk + dT, yi)[1]
  lmbdkdT = np.linalg.eigvals(QdT).min()
  dlmbddT = (lmbdkdT - lmbdk) / dT
  dT = -lmbdk / dlmbddT
  repeat = lmbdk < -tol or lmbdk > tol
  logger.debug(tmpl, k, lmbdk, *zetai, Tk, dT)
  while repeat and k < maxiter:
    Tkp1 = Tk + dT
    Q = eos.getVT_lnfi_dnj(V, Tkp1, yi)[1]
    zetai, lmbdkp1 = pyrqi(Q, zetai)
    dlmbddT = (lmbdkp1 - lmbdk) / dT
    dT = -lmbdkp1 / dlmbddT
    k += 1
    Tk = Tkp1
    lmbdk = lmbdkp1
    repeat = lmbdk < -tol or lmbdk > tol
    logger.debug(tmpl, k, lmbdk, *zetai, Tk, dT)
  if not repeat:
    return Tk, zetai
  logger.warning(
    "The spinodal temperature was not found. The EOS was: %s.\nParameters:\n"
    "V = %s\nyi = %s\nT0 = %s\nzeta0i = %s\nmultdT0 = %s",
    eos.name, V, yi, T0, zeta0i, multdT0,
  )
  raise SolutionNotFoundError(
    "The spinodal temperature solution procedure completed unsuccessfully.\n"
    "Try to increase the number of iterations or change the initial guess."
  )


def getVT_PcTc(
  yi: Vector,
  eos: CritEosVT,
  v0: None | Scalar = None,
  T0: None | Scalar = None,
  kappa0: None | Scalar = 3.5,
  multdV0: Scalar = 1e-5,
  krange: tuple[Scalar, Scalar] = (1.1, 5.),
  kstep: Scalar = 0.1,
  tol: Scalar = 1e-5,
  maxiter: int = 25,
) -> tuple[Scalar, Scalar]:
  """Calculates the critical pressure and temperature of a mixture.

  Parameters
  ----------
  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  eos: CritEosVT
    An initialized instance of a VT-based equation of state. Must have
    the following methods:

    - `getVT_lnfi_dnj(V: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Matrix]`
      For a given volume [m3], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacities of components as a `Vector` of
        shape `(Nc,)`,
      - partial derivatives of logarithms of fugacities of components
        with respect to their mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    - `getVT_d3F(V: Scalar, T: Scalar, yi: Vector,
                 zti: Vector, n: Scalar) -> Scalar`
      For a given volume [m3], temperature [K], mole composition
      (`Vector` of shape `(Nc,)`), component mole number changes
      (`Vector` of shape `(Nc,)`), and mixture mole number, this method
      must return the cubic form of the Helmholtz energy Taylor series
      decomposition.

    - `getVT_vmin(T: Scalar, yi: Vector) -> Scalar`
      For a given temperature [K] and mole composition (`Vector` of
      shape `(Nc,)`), this method must return the minimum molar
      volume of the mixture in [m3/mol].

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

    - `name: str`
      The EOS name (for proper logging).

  v0: Scalar
    An initial guess of the critical molar volume of a mixture [m3/mol].
    Default is `None` which means that the gridding procedure or the
    `kappa0` will be used instead.

  kappa0: Scalar | None
    An initial guess for the relation of the critical molar volume
    to the minimal possible volume provided by an equation of state.
    Default is `None` which means that the gridding procedure or the
    `v0` will be used instead.

  krange: tuple[Scalar, Scalar]
    A range of possible values of the relation of the molar volume
    to the minimal possible volume provided by an equation of state
    for the gridding procedure. The gridding procedure will be used
    if both initial guesses `v0` and `kappa0` are equal `None`.
    Default is `(1.1, 5.0)`.

  kstep: Scalar
    A step which is used to perform the gridding procedure to find
    an initial guess of the relation of the molar volume to the minimal
    possible volume. Default is `0.1`.

   tol: Scalar
    Terminate successfully if the absolute value of the primary
    variable change is less than `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of iterations. Default is `25`.

  Returns
  -------
  A tuple of the critical pressure and temperature.

  Raises
  ------
  `SolutionNotFoundError` if the solution was not found.
  """
  logger.info('Calculating the critical point.')
  Nc = eos.Nc
  logger.debug('%3s%9s%9s%9s%11s', 'Nit', 'kappa', 'T, K', 'C', 'dkappa')
  tmpl = '%3s%9.4f%9.2f%9.2f%11.2e'
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
  repeat = dkappa < -tol or dkappa > tol
  logger.debug(tmpl, k, kappak, T, Ck, dkappa)
  k += 1
  while repeat and k < maxiter:
    kappakp1 = kappak + dkappa
    vkp1 = kappakp1 * vmin
    if vkp1 < vmin:
      vkp1 = (vk + vmin) * .5
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
    repeat = dkappa < -tol or dkappa > tol
    logger.debug(tmpl, k, kappak, T, Ck, dkappa)
    k += 1
  if not repeat:
    Pc = eos.getVT_P(vk, T, yi)
    logger.info('Critical point is: Pc = %.1f Pa, Tc = %.2f K', Pc, T)
    return Pc, T
  logger.warning(
    "The critical point was not found. The EOS was %s.\nParameters:\n"
    "yi = %s\nv0 = %s m3/mol\nT0 = %s K\nkappa0 = %s\nmultdV0 = %s\n"
    "krange = %s\nkstep = %s\ntol = %s\nmaxiter = %s",
    eos.name, yi, v0, T0, kappa0, multdV0, krange, kstep, tol, maxiter,
  )
  raise SolutionNotFoundError(
    "The critical point solution procedure completed unsuccessfully.\n"
    "Try to increase the number of iterations or change the initial guess."
  )


class SatResult(dict):
  """Container for saturation point calculation outputs with
  pretty-printing.

  Attributes
  ----------
  P: Scalar
    Saturation pressure [Pa].

  T: Scalar
    Saturation temperature [K].

  yji: Matrix, shape (Np, Nc)
    Equilibrium composition of two phases for the saturation point.
    Two-dimensional array of real elements of size `(Np, Nc)`, where
    `Np` is the number of phases and `Nc` is the number of components.

  Zj: Vector, shape (Np,)
    Compressibility factors of each phase. Array of real elements of
    size `(Np,)`, where `Np` is the number of phases.

  g2: Scalar
    The sum of squared elements of a vector of equations.

  TPD: Scalar
    The tangent-plane distance at the solution.

  Niter: int
    The number of iterations.
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
           f"Phase compressibility factors:\n{self.Zj}\n")
    return s


class PsatPT(object):
  """Saturation pressure calculation.

  Performs saturation pressure calculation using PT-based equations of
  state.

  Parameters
  ----------
  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> tuple[Vector, ...]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as a tuple of `Vector` of shape `(Nc,)`.

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_lnphii_Z_dnj_dP(P: Scalar, T: Scalar, yi: Vector, n: Scalar,
       ) -> tuple[Vector, Scalar, Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
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
    - `'newton'` (Newton's method),
    - `'ss-newton'` (Currently raises `NotImplementedError`),
    - `'qnss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval of the saturation pressure. This dictionary is used to
    specify arguments for the stability test procedure. Default is an
    empty dictionary.

  **kwargs: dict
    Other arguments for a Psat-solver. It may contain such arguments
    as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others depending
    on the selected solver.

  Methods
  -------
  run(P: Scalar, T: Scalar, yi: Vector) -> SatResult
    This method performs the saturation pressure calculation for a given
    initial guess of saturation pressure [Pa], temperature in [K] and
    mole composition `yi` of shape `(Nc,)`. This method returns
    saturation pressure calculation results as an instance of
    `SatResult`.

  search(P: Scalar, T: Scalar, yi: Vector) -> tuple[Scalar, Vector,
                                                    Scalar, Scalar]
    This method performs the preliminary search to refine an initial
    guess of the saturation pressure and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the saturation pressure [Pa],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the saturation pressure [Pa],
    - the upper bound of the saturation pressure [Pa].

  gridding(P: Scalar, T: Scalar, yi: Vector) -> tuple[Scalar, Vector,
                                                      Scalar, Scalar]
    This method performs the gridding procedure to refine an initial
    guess of the saturation pressure and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the saturation pressure [Pa],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the saturation pressure [Pa],
    - the upper bound of the saturation pressure [Pa].
  """
  def __init__(
    self,
    eos: PsatEosPT,
    method: str = 'ss',
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.solver = partial(_PsatPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_PsatPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_PsatPT_newtA, eos=eos, **kwargs)
    elif method == 'newton-b':
      self.solver = partial(_PsatPT_newtB, eos=eos, **kwargs)
    elif method == 'newton-c':
      self.solver = partial(_PsatPT_newtC, eos=eos, **kwargs)
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
    P: Scalar,
    T: Scalar,
    yi: Vector,
    upper: bool = True,
    search: bool = True,
    **kwargs,
  ) -> SatResult:
    """Performs the saturation pressure calculation for known
    temperature and composition. To improve an initial guess, the
    preliminary search is performed.

    Parameters
    ----------
    P: Scalar
      Initial guess of the saturation pressure [Pa].

    T: Scalar
      Temperature of a mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondentherm serves as the dividing point between upper and
      lower phase boundaries. Default is `True`.

    search: bool
      A boolean flag that indicates whether the preliminary search or
      gridding procedure should be used to refine an initial guess of
      the saturation pressure calculation. It is recommended to use the
      gridding procedure if the initial guess of the saturation pressure
      is significantly inaccurate. Default is `True`.

    **kwargs: dict
      Parameters for the preliminary search or gridding procedure.
      It may contain key-value pairs to change default values for
      `Pmin`, `Pmax`, `Nnodes` and other depending on the `search`
      flag. Default is an empty dictionary.

    Returns
    -------
    Saturation pressure calculation results as an instance of the
    `SatResult`. Important attributes are:
    - `P` the saturation pressure in [Pa],
    - `T` the saturation temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase.

    Raises
    ------
    ValueError
      The `ValueError` exception will be raised if the one-phase or
      two-phase region was not found using a preliminary search.

    SolutionNotFoundError
      The `SolutionNotFoundError` exception will be raised if the
      saturation pressure calculation procedure terminates
      unsuccessfully.
    """
    if search:
      P0, kvi0, Plow, Pupp = self.search(P, T, yi, upper=upper, **kwargs)
    else:
      P0, kvi0, Plow, Pupp = self.gridding(P, T, yi, upper=upper, **kwargs)
    return self.solver(P0, T, yi, kvi0, Plow=Plow, Pupp=Pupp, upper=upper)

  def search(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    Pmin: Scalar = 1.,
    Pmax: Scalar = 1e8,
    upper: bool = True,
    step: Scalar = 0.1,
  ) -> tuple[Scalar, Vector, Scalar, Scalar]:
    """Performs the preliminary search to refine the initial guess of
    the saturation pressure.

    Parameters
    ----------
    P: Scalar
      The initial guess of the saturation pressure to be improved [Pa].

    T: Scalar
      Temperature of a mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Pmin: Scalar
      During the preliminary search, the pressure can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `1.` [Pa].

    Pmax: Scalar
      During the preliminary search, the pressure can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `1e8` [Pa].

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondentherm serves as the dividing point between upper and
      lower phase boundaries. Default is `True`.

    step: Scalar
      To specify the confidence interval for the saturation pressure
      calculation, the preliminary search is performed. This parameter
      regulates the step of this search in fraction units. For example,
      if it is necessary to find the upper bound of the confidence
      interval, then the next value of pressure will be calculated from
      the previous one using the formula: `Pnext = Pprev * (1. + step)`.
      Default is `0.1`.

    Returns
    -------
    A tuple of:
    - the improved initial guess for the saturation pressure [Pa],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the saturation pressure [Pa],
    - the upper bound of the saturation pressure [Pa].

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    stab = self.stabsolver.run(P, T, yi)
    logger.debug(
      'For the initial guess P = %.1f Pa, the one-phase state is stable: %s',
      P, stab.stable,
    )
    if stab.stable and upper:
      logger.debug(
        'Finding the two-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Pupp = P
      c = 1. - step
      Plow = c * P
      stabmin = self.stabsolver.run(Plow, T, yi)
      logger.debug(
        'Plow = %.1f Pa, the one-phase state is stable: %s',
        Plow, stabmin.stable,
      )
      if stabmin.stable:
        Pupp = Plow
      while stabmin.stable and Plow > Pmin:
        Plow *= c
        stabmin = self.stabsolver.run(Plow, T, yi)
        logger.debug(
          'Plow = %.1f Pa, the one-phase state is stable: %s',
          Plow, stabmin.stable,
        )
        if stabmin.stable:
          Pupp = Plow
      if Plow < Pmin:
        raise ValueError(
          'The two-phase region was not identified. It could be because of\n'
          'its narrowness or absence. Try to change the initial guess for\n'
          'pressure or stability test parameters using the `stabkwargs`.\n'
          'It also might be helpful to reduce the value of the `step`.'
        )
      else:
        P = Plow
        stab = stabmin
    elif not stab.stable and upper:
      logger.debug(
        'Finding the one-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Plow = P
      c = 1. + step
      Pupp = c * P
      stabmax = self.stabsolver.run(Pupp, T, yi)
      logger.debug(
        'Pupp = %.1f Pa, the one-phase state is stable: %s',
        Pupp, stabmax.stable,
      )
      if not stabmax.stable:
        Plow = Pupp
        stab = stabmax
      while not stabmax.stable and Pupp < Pmax:
        Pupp *= c
        stabmax = self.stabsolver.run(Pupp, T, yi)
        logger.debug(
          'Pupp = %.1f Pa, the one-phase state is stable: %s',
          Pupp, stabmax.stable,
        )
        if not stabmax.stable:
          Plow = Pupp
          stab = stabmax
      if Pupp > Pmax:
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for pressure and/or `Pmax` parameter.'
        )
      else:
        P = Plow
    elif stab.stable and not upper:
      logger.debug(
        'Finding the two-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Plow = P
      c = 1. + step
      Pupp = c * P
      stabmax = self.stabsolver.run(Pupp, T, yi)
      logger.debug(
        'Pupp = %.1f Pa, the one-phase state is stable: %s',
        Pupp, stabmax.stable,
      )
      while stabmax.stable and Pupp < Pmax:
        Pupp *= c
        stabmax = self.stabsolver.run(Pupp, T, yi)
        logger.debug(
          'Pupp = %.1f Pa, the one-phase state is stable: %s',
          Pupp, stabmax.stable,
        )
        if stabmax.stable:
          Plow = Pupp
      if Pupp > Pmax:
        raise ValueError(
          'The two-phase region was not identified. It could be because of\n'
          'its narrowness or absence. Try to change the initial guess for\n'
          'pressure or stability test parameters using the `stabkwargs`.\n'
          'It also might be helpful to reduce the value of the `step`.'
        )
      else:
        P = Pupp
        stab = stabmax
    else:
      logger.debug(
        'Finding the one-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Pupp = P
      c = 1. - step
      Plow = c * P
      stabmin = self.stabsolver.run(Plow, T, yi)
      logger.debug(
        'Plow = %.1f Pa, the one-phase state is stable: %s',
        Plow, stabmin.stable,
      )
      if not stabmin.stable:
        Pupp = Plow
        stab = stabmin
      while not stabmin.stable and Plow > Pmin:
        Plow *= c
        stabmin = self.stabsolver.run(Plow, T, yi)
        logger.debug(
          'Plow = %.1f Pa, the one-phase state is stable: %s',
          Plow, stabmin.stable,
        )
        if not stabmin.stable:
          Pupp = Plow
          stab = stabmin
      if Plow < Pmin:
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for pressure and/or `Pmin` parameter.'
        )
      else:
        P = Pupp
    return P, stab.kvji[0], Plow, Pupp

  def gridding(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    Pmin: Scalar = 1.,
    Pmax: Scalar = 1e8,
    upper: bool = True,
    Nnodes: int = 10,
    logspace: bool = False,
  ) -> tuple[Scalar, Vector, Scalar, Scalar]:
    """Performs the gridding procedure to refine the initial guess of
    the saturation pressure. Instead of evaluating all segments, the
    gridding procedure would terminate as soon as an interval exhibiting
    a change in stability is identified. It is recommended to use the
    gridding procedure if the initial guess of the saturation pressure
    is significantly inaccurate.

    Parameters
    ----------
    P: Scalar
      The initial guess of the saturation pressure to be improved [Pa].

    T: Scalar
      Temperature of a mixture [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Pmin: Scalar
      Min pressure for the gridding procedure. Default is `1.` [Pa].

    Pmax: Scalar
      Max pressure for the gridding procedure. Default is `1e8` [Pa].

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondentherm serves as the dividing point between upper and
      lower phase boundaries. This flag controls the start point: if it
      is set to `True`, then the start point would be `Pmax`; otherwise
      it would be `Pmin`. Default is `True`.

    Nnodes: int
      The number of points to construct a grid. Default is `10`.

    logspace: bool
      A boolean flag indicating whether the log10 grid should be used
      instead of linear. It is advisable to use this option only if
      the saturation pressure is expected to be very small. Default is
      `False`.

    Returns
    -------
    A tuple of:
    - the improved initial guess for the saturation pressure [Pa],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the saturation pressure [Pa],
    - the upper bound of the saturation pressure [Pa].

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    if logspace:
      if upper:
        PP = np.logspace(np.log10(Pmax), np.log10(Pmin), Nnodes,
                         endpoint=True)
      else:
        PP = np.logspace(np.log10(Pmin), np.log10(Pmax), Nnodes,
                         endpoint=True)
    else:
      if upper:
        PP = np.linspace(Pmax, Pmin, Nnodes, endpoint=True)
      else:
        PP = np.linspace(Pmin, Pmax, Nnodes, endpoint=True)
    prevP = PP[0]
    prevresult = self.stabsolver.run(prevP, T, yi)
    prevstable = prevresult.stable
    logger.debug(
      'For P = %.1f Pa, the one-phase state is stable: %s', prevP, prevstable,
    )
    for nextP in PP[1:]:
      result = self.stabsolver.run(nextP, T, yi)
      stable = result.stable
      logger.debug(
        'For P = %.1f Pa, the one-phase state is stable: %s', nextP, stable,
      )
      if not stable and prevstable or stable and not prevstable:
        if upper:
          Plow = nextP
          Pupp = prevP
        else:
          Plow = prevP
          Pupp = nextP
        if not stable:
          stabres = result
          P0 = nextP
        else:
          stabres = prevresult
          P0 = prevP
        return P0, stabres.kvji[0], Plow, Pupp
      prevstable = stable
      prevresult = result
      prevP = nextP
    raise ValueError(
      'A boundary of the two-phase region was not identified. It could be\n'
      'because of its narrowness or absence. Try to change the number of\n'
      'points for gridding or stability test parameters using the\n'
      '`stabkwargs` parameter.'
    )


def _PsatPT_solve_TPDeq_P(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  xi: Vector,
  eos: PsatEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 10,
  Plow0: Scalar = 1.,
  Pupp0: Scalar = 1e8,
  increasing: bool = True,
) -> tuple[Scalar, Vector, Vector, Scalar, Scalar, Scalar]:
  """Solves the TPD-equation using the PT-based equations of state for
  pressure at a constant temperature. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the saturation pressure [Pa].

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  tol: Scalar
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-8`.

  Plow0: Scalar
    The initial lower bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `1.0` [Pa].

  Pupp0: Scalar
    The initial upper bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `1e8` [Pa].

  increasing: bool
    A flag that indicates if the TPD-equation vs pressure is an
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation pressure,
  - natural logarithms of fugacity coefficients of components in the
    trial phase as a `Vector` of shape `(Nc,)`,
  - natural logarithms of fugacity coefficients of components in the
    mixture as a `Vector` of shape `(Nc,)`,
  - compressibility factors for both mixtures,
  - value of the tangent-plane distance.
  """
  # print('%3s%10s%10s' % ('Nit', 'P, Pa', 'TPD'))
  # tmpl = '%3s%10.2e%10.2e'
  k = 0
  Pk = P0
  Plow = Plow0
  Pupp = Pupp0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(tmpl % (k, Pk, TPD))
  while (TPD < -tol or TPD > tol) and k < maxiter:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Plow = Pk
    else:
      Pupp = Pk
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    dP = - TPD / dTPDdP
    Pkp1 = Pk + dP
    if Pkp1 > Pupp or Pkp1 < Plow:
      Pkp1 = .5 * (Plow + Pupp)
      dP = Pkp1 - Pk
    if dP > -1e-8 and dP < 1e-8:
      break
    Pk = Pkp1
    k += 1
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(tmpl % (k, Pk, TPD))
  return Pk, lnphixi, lnphiyi, Zx, Zy, TPD


def _PsatPT_ss(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: PsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 200,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Plow: Scalar = 1.,
  Pupp: Scalar = 1e8,
  upper: bool = True,
) -> SatResult:
  """Successive substitution (SS) method for the saturation pressure
  calculation using a PT-based equation of state.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `200`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-8`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `10`.

  Plow: Scalar
    The lower bound for the TPD-equation solver. Default is `1.0` [Pa].

  Pupp: Scalar
    The upper bound for the TPD-equation solver. Default is `1e8` [Pa].

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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info('Saturation pressure calculation using the SS-method.')
  Nc = eos.Nc
  logger.info('T = %.2f K, yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat, Pa', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%10.2e%11.2e'
  solverTPDeq = partial(_PsatPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Plow0=Plow, Pupp0=Pupp,
                        increasing=upper)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  TPD = xi.dot(gi - np.log(n))
  repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
  logger.debug(tmpl, k, *lnkik, Pk, g2, TPD)
  while repeat and k < maxiter:
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
    logger.debug(tmpl, k, *lnkik, Pk, g2, TPD)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk):
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
    logger.info('Saturation pressure is: %.1f Pa.', Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, Niter=k)
  logger.warning(
    "The SS-method for saturation pressure calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT = %s K\n"
    "yi = %s\nPlow = %s Pa\nPupp = %s Pa",
    eos.name, P0, T, yi, Plow, Pupp,
  )
  raise SolutionNotFoundError(
    'The saturation pressure calculation\nterminates unsuccessfully. Try '
    'to increase the maximum number of\nsolver iterations.'
  )


def _PsatPT_qnss(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: PsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 50,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Plow: Scalar = 1.,
  Pupp: Scalar = 1e8,
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
  P0: Scalar
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-8`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `10`.

  Plow: Scalar
    The lower bound for the TPD-equation solver. Default is `1.0` [Pa].

  Pupp: Scalar
    The upper bound for the TPD-equation solver. Default is `1e8` [Pa].

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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info('Saturation pressure calculation using the QNSS-method.')
  Nc = eos.Nc
  logger.info('T = %.2f K, yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat, Pa', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%10.2e%11.2e'
  solverTPDeq = partial(_PsatPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Plow0=Plow, Pupp0=Pupp,
                        increasing=upper)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  gi = np.log(kik) + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  TPD = xi.dot(gi - np.log(n))
  lmbd = 1.
  repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
  logger.debug(tmpl, k, *lnkik, Pk, g2, TPD)
  while repeat and k < maxiter:
    dlnki = -lmbd * gi
    max_dlnki = np.abs(dlnki).max()
    if max_dlnki > 6.:
      relax = 6. / max_dlnki
      lmbd *= relax
      dlnki *= relax
    k += 1
    tkm1 = dlnki.dot(gi)
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
    logger.debug(tmpl, k, *lnkik, Pk, g2, TPD)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk):
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
    logger.info('Saturation pressure is: %.1f Pa.', Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, Niter=k)
  logger.warning(
    "The QNSS-method for saturation pressure calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT = %s K\n"
    "yi = %s\nPlow = %s Pa\nPupp = %s Pa",
    eos.name, P0, T, yi, Plow, Pupp,
  )
  raise SolutionNotFoundError(
    'The saturation pressure calculation\nterminates unsuccessfully. Try '
    'to increase the maximum number of\nsolver iterations.'
  )


def _PsatPT_newt_improveP0(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  xi: Vector,
  eos: PsatEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 10,
  Plow0: Scalar = 1.,
  Pupp0: Scalar = 1e8,
  increasing: bool = True,
) -> tuple[Scalar, Vector, Scalar, Vector]:
  """Improves initial guess of the saturation pressure by solving
  the TPD-equation using the PT-based equations of state for pressure
  at a constant temperature. The TPD-equation is the equation of
  equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation. The
  algorithm is the same as for the `_PsatPT_solve_TPDeq_P` but it
  differs in returned result.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the saturation pressure [Pa].

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  tol: Scalar
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-8`.

  Plow0: Scalar
    The initial lower bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `1.0` [Pa].

  Pupp0: Scalar
    The initial upper bound of the saturation pressure [Pa]. If the
    saturation pressure at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `1e8` [Pa].

  increasing: bool
    A flag that indicates if the TPD-equation vs pressure is an
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation pressure,
  - natural logarithms of fugacity coefficients of components in the
    mixture as a `Vector` of shape `(Nc,)`,
  - compressibility factor of the mixture,
  - partial derivatives of logarithms of the fugacity coefficients of
    components in the mixture with respect to pressure as a `Vector`
    of shape `(Nc,)`.
  """
  # print('%3s%10s%10s' % ('Nit', 'P, Pa', 'TPD'))
  # tmpl = '%3s%10.2e%10.2e'
  k = 0
  Pk = P0
  Plow = Plow0
  Pupp = Pupp0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(tmpl % (k, Pk, TPD))
  while (TPD < -tol or TPD > tol) and k < maxiter:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Plow = Pk
    else:
      Pupp = Pk
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    dP = - TPD / dTPDdP
    Pkp1 = Pk + dP
    if Pkp1 > Pupp or Pkp1 < Plow:
      Pkp1 = .5 * (Plow + Pupp)
      dP = Pkp1 - Pk
    if dP > -1e-8 and dP < 1e-8:
      break
    Pk = Pkp1
    k += 1
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(tmpl % (k, Pk, TPD))
  return Pk, lnphiyi, Zy, dlnphiyidP


def _PsatPT_newtA(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: PsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 20,
  Plow: Scalar = 1.,
  Pupp: Scalar = 1e8,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
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
  P0: Scalar
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    - `getPT_lnphii_Z_dnj_dP(
         P: Scalar, T: Scalar, yi: Vector, n: Scalar,
       ) -> tuple[Vector, Scalar, Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements of
    the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Plow: Scalar
    The pressure lower bound. Default is `1.0` [Pa].

  Pupp: Scalar
    The pressure upper bound. Default is `1e8` [Pa].

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm of the saturation pressure
    initial guess improvement. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation pressure initial guess
    improvement. Default is `10`.

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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (A-form)."
  )
  Nc = eos.Nc
  logger.info('T = %.2f K, yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat, Pa', 'g2',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%10.2e'
  J = np.zeros(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphiyi, Zy, dlnphiyidP = _PsatPT_newt_improveP0(
    P0, T, yi, xi, eos, tol_tpd, maxiter_tpd, Plow, Pupp, upper,
  )
  lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(
    Pk, T, xi, n,
  )
  gi[:Nc] = lnkik + lnphixi - lnphiyi
  gi[-1] = n - 1.
  g2 = gi.dot(gi)
  logger.debug(tmpl, k, *lnkik, Pk, g2)
  while g2 > tol and k < maxiter:
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = ni
    J[:Nc,-1] = Pk * (dlnphixidP - dlnphiyidP)
    try:
      dlnkilnP = linsolver(J, -gi)
    except:
      dlnkilnP = -gi
    k += 1
    lnkik += dlnkilnP[:-1]
    Pkp1 = Pk * np.exp(dlnkilnP[-1])
    if Pkp1 > Pupp:
      Pk = .5 * (Pk + Pupp)
    elif Pkp1 < Plow:
      Pk = .5 * (Plow + Pk)
    else:
      Pk = Pkp1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(
      Pk, T, xi, n,
    )
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    gi[:Nc] = lnkik + lnphixi - lnphiyi
    gi[-1] = n - 1.
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnkik, Pk, g2)
  if g2 < tol and np.isfinite(g2) and np.isfinite(Pk):
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
    logger.info('Saturation pressure is: %.1f Pa.', Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji, g2=g2,
                     TPD=xi.dot(gi[:Nc] - np.log(n)), Niter=k)
  logger.warning(
    "Newton's method (A-form) for saturation pressure calculation termin"
    "ates unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT = %s K\n"
    "yi = %s\nPlow = %s Pa\nPupp = %s Pa",
    eos.name, P0, T, yi, Plow, Pupp,
  )
  raise SolutionNotFoundError(
    'The saturation pressure calculation\nterminates unsuccessfully. Try '
    'to increase the maximum number of\nsolver iterations.'
  )


def _PsatPT_newtB(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: PsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 20,
  Plow: Scalar = 1.,
  Pupp: Scalar = 1e8,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
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
  P0: Scalar
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    - `getPT_lnphii_Z_dnj_dP(
         P: Scalar, T: Scalar, yi: Vector, n: Scalar,
       ) -> tuple[Vector, Scalar, Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements of
    the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Plow: Scalar
    The pressure lower bound. Default is `1.0` [Pa].

  Pupp: Scalar
    The pressure upper bound. Default is `1e8` [Pa].

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm of the saturation pressure
    initial guess improvement. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation pressure initial guess
    improvement. Default is `10`.

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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (B-form)."
  )
  Nc = eos.Nc
  logger.info('T = %.2f K, yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat, Pa', 'g2',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%10.2e'
  J = np.empty(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphiyi, Zy, dlnphiyidP = _PsatPT_newt_improveP0(
    P0, T, yi, xi, eos, tol_tpd, maxiter_tpd, Plow, Pupp, upper,
  )
  lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(
    Pk, T, xi, n,
  )
  gi[:Nc] = lnkik + lnphixi - lnphiyi
  hi = gi[:Nc] - np.log(n)
  gi[-1] = xi.dot(hi)
  g2 = gi.dot(gi)
  logger.debug(tmpl, k, *lnkik, Pk, g2)
  while g2 > tol and k < maxiter:
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = xi * (hi - gi[-1])
    J[:Nc,-1] = Pk * (dlnphixidP - dlnphiyidP)
    J[-1,-1] = xi.dot(J[:Nc,-1])
    try:
      dlnkilnP = linsolver(J, -gi)
    except:
      dlnkilnP = -gi
    k += 1
    lnkik += dlnkilnP[:-1]
    Pkp1 = Pk * np.exp(dlnkilnP[-1])
    if Pkp1 > Pupp:
      Pk = .5 * (Pk + Pupp)
    elif Pkp1 < Plow:
      Pk = .5 * (Plow + Pk)
    else:
      Pk = Pkp1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidP = eos.getPT_lnphii_Z_dnj_dP(
      Pk, T, xi, n,
    )
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    gi[:Nc] = lnkik + lnphixi - lnphiyi
    hi = gi[:Nc] - np.log(n)
    gi[-1] = xi.dot(hi)
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnkik, Pk, g2)
  if g2 < tol and np.isfinite(g2) and np.isfinite(Pk):
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
    logger.info('Saturation pressure is: %.1f Pa.', Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=gi[-1], Niter=k)
  logger.warning(
    "Newton's method (B-form) for saturation pressure calculation termin"
    "ates unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT = %s K\n"
    "yi = %s\nPlow = %s Pa\nPupp = %s Pa",
    eos.name, P0, T, yi, Plow, Pupp,
  )
  raise SolutionNotFoundError(
    'The saturation pressure calculation\nterminates unsuccessfully. Try '
    'to increase the maximum number of\nsolver iterations.'
  )


def _PsatPT_newtC(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: PsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Plow: Scalar = 1.,
  Pupp: Scalar = 1e8,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
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
  P0: Scalar
    Initial guess of the saturation pressure [Pa]. It should be inside
    the two-phase region.

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: PsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements of
    the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `10`.

  Plow: Scalar
    The pressure lower bound. Default is `1.0` [Pa].

  Pupp: Scalar
    The pressure upper bound. Default is `1e8` [Pa].

  linsolver: Callable[[Matrix, Vector], Vector]
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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (C-form)."
  )
  Nc = eos.Nc
  logger.info('T = %.2f K, yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%10s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat, Pa', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%10.2e%11.2e'
  solverTPDeq = partial(_PsatPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Plow0=Plow, Pupp0=Pupp,
                        increasing=upper)
  I = np.eye(eos.Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P0, T, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  TPD = ni.dot(gi)
  repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
  logger.debug(tmpl, k, *lnkik, Pk, g2, TPD)
  while repeat and k < maxiter:
    dlnphixidnj = eos.getPT_lnphii_Z_dnj(Pk, T, xi, n)[2]
    J = I + ni * dlnphixidnj
    try:
      dlnki = linsolver(J, -gi)
    except:
      dlnki = -gi
    k += 1
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Pk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(Pk, T, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
    logger.debug(tmpl, k, *lnkik, Pk, g2, TPD)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk):
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
    logger.info('Saturation pressure is: %.1f Pa.', Pk)
    return SatResult(P=Pk, T=T, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, Niter=k)
  logger.warning(
    "Newton's method (C-form) for saturation pressure calculation termin"
    "ates unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT = %s K\n"
    "yi = %s\nPlow = %s Pa\nPupp = %s Pa",
    eos.name, P0, T, yi, Plow, Pupp,
  )
  raise SolutionNotFoundError(
    'The saturation pressure calculation\nterminates unsuccessfully. Try '
    'to increase the maximum number of\nsolver iterations.'
  )


class TsatPT(object):
  """Saturation temperature calculation.

  Performs saturation temperature calculation using PT-based equations
  of state.

  Parameters
  ----------
  eos: TsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> tuple[Vector, ...]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as a tuple of `Vector` of shape `(Nc,)`.

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    If the solution method would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then it also must have:

    - `getPT_lnphii_Z_dnj_dT(
         P: Scalar, T: Scalar, yi: Vector, n: Scalar,
       ) -> tuple[Vector, Scalar, Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
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
    - `'newton'` (Currently raises `NotImplementedError`),
    - `'ss-newton'` (Currently raises `NotImplementedError`),
    - `'qnss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval of the saturation temperature. This dictionary is used to
    specify arguments for the stability test procedure. Default is an
    empty dictionary.

  **kwargs: dict
    Other arguments for a Tsat-solver. It may contain such arguments
    as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others depending
    on the selected solver.

  Methods
  -------
  run(P: Scalar, T: Scalar, yi: Vector) -> SatResult
    This method performs the saturation temperature calculation for a
    given pressure [Pa], initial guess of saturation temperature in [K]
    and mole composition `yi` of shape `(Nc,)`. This method returns
    saturation temperature calculation results as an instance of
    `SatResult`.

  search(P: Scalar, T: Scalar, yi: Vector) -> tuple[Scalar, Vector,
                                                    Scalar, Scalar]
    This method performs the preliminary search to refine an initial
    guess of the saturation temperature and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the saturation temperature [K],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the saturation temperature [K],
    - the upper bound of the saturation temperature [K].

  gridding(P: Scalar, T: Scalar, yi: Vector) -> tuple[Scalar, Vector,
                                                      Scalar, Scalar]
    This method performs the gridding procedure to refine an initial
    guess of the saturation temperature and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the saturation temperature [K],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the saturation temperature [K],
    - the upper bound of the saturation temperature [K].
  """
  def __init__(
    self,
    eos: TsatEosPT,
    method: str = 'ss',
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.solver = partial(_TsatPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_TsatPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_TsatPT_newtA, eos=eos, **kwargs)
    elif method == 'newton-b':
      self.solver = partial(_TsatPT_newtB, eos=eos, **kwargs)
    elif method == 'newton-c':
      self.solver = partial(_TsatPT_newtC, eos=eos, **kwargs)
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
    P: Scalar,
    T: Scalar,
    yi: Vector,
    upper: bool = True,
    search: bool = True,
    **kwargs,
  ) -> SatResult:
    """Performs the saturation temperature calculation for known
    pressure and composition. To improve an initial guess, the
    preliminary search is performed.

    Parameters
    ----------
    P: Scalar
      Pressure of a mixture [Pa].

    T: Scalar
      Initial guess of the saturation temperature [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondenbar serves as the dividing point between upper and
      lower phase boundaries. Default is `True`.

    search: bool
      A boolean flag that indicates whether the preliminary search or
      gridding procedure should be used to refine an initial guess of
      the saturation temperature calculation. It is recommended to use
      the gridding procedure if the initial guess of the saturation
      temperature is significantly inaccurate. Default is `True`.

    **kwargs: dict
      Parameters for the preliminary search or gridding procedure.
      It may contain key-value pairs to change default values for
      `Tmin`, `Tmax`, `Nnodes` and other depending on the `search`
      flag. Default is an empty dictionary.

    Returns
    -------
    Saturation temperature calculation results as an instance of the
    `SatResult`. Important attributes are:
    - `P` the saturation pressure in [Pa],
    - `T` the saturation temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.

    SolutionNotFoundError
      The `SolutionNotFoundError` exception will be raised if the
      saturation pressure calculation procedure terminates
      unsuccessfully.
    """
    if search:
      T0, kvi0, Tlow, Tupp = self.search(P, T, yi, upper=upper, **kwargs)
    else:
      T0, kvi0, Tlow, Tupp = self.gridding(P, T, yi, upper=upper, **kwargs)
    return self.solver(P, T0, yi, kvi0, Tlow=Tlow, Tupp=Tupp, upper=upper)

  def search(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    Tmin: Scalar = 173.15,
    Tmax: Scalar = 973.15,
    upper: bool = True,
    step: Scalar = 0.1,
  ) -> tuple[Scalar, Vector, Scalar, Scalar]:
    """Performs a preliminary search to refine the initial guess of
    the saturation temperature.

    Parameters
    ----------
    P: Scalar
      Pressure of a mixture [Pa].

    T: Scalar
      The initial guess of the saturation temperature to be improved
      [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Tmin: Scalar
      During the preliminary search, the temperature can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `173.15` [K].

    Tmax: Scalar
      During the preliminary search, the temperature can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `973.15` [K].

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondenbar serves as the dividing point between upper and
      lower phase boundaries. Default is `True`.

    step: Scalar
      To specify the confidence interval for the saturation temperature
      calculation, the preliminary search is performed. This parameter
      regulates the step of this search in fraction units. For example,
      if it is necessary to find the upper bound of the confidence
      interval, then the next value of temperature will be calculated
      from the previous one using the formula:
      `Tnext = Tprev * (1. + step)`. Default is `0.1`.

    Returns
    -------
    A tuple of:
    - the improved initial guess for the saturation temperature [K],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the saturation temperature [K],
    - the upper bound of the saturation temperature [K].

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    stab = self.stabsolver.run(P, T, yi)
    logger.debug(
      'For the initial guess T = %.2f K, the one-phase state is stable: %s',
      T, stab.stable,
    )
    if stab.stable and upper:
      logger.debug(
        'Finding the two-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Tupp = T
      c = 1. - step
      Tlow = c * T
      stabmin = self.stabsolver.run(P, Tlow, yi)
      logger.debug(
        'Tlow = %.2f K, the one-phase state is stable: %s',
        Tlow, stabmin.stable,
      )
      if stabmin.stable:
        Tupp = Tlow
      while stabmin.stable and Tlow > Tmin:
        Tlow *= c
        stabmin = self.stabsolver.run(P, Tlow, yi)
        logger.debug(
          'Tlow = %.2f K, the one-phase state is stable: %s',
          Tlow, stabmin.stable,
        )
        if stabmin.stable:
          Tupp = Tlow
      if Tlow < Tmin:
        raise ValueError(
          'The two-phase region was not identified. It could be because of\n'
          'its narrowness or absence. Try to change the initial guess for\n'
          'temperature or stability test parameters using the `stabkwargs`.\n'
          'It also might be helpful to reduce the value of the `step`.'
        )
      else:
        T = Tlow
        stab = stabmin
    elif not stab.stable and upper:
      logger.debug(
        'Finding the one-phase region for the upper-bound curve by the '
        'preliminary search.'
      )
      Tlow = T
      c = 1. + step
      Tupp = c * T
      stabmax = self.stabsolver.run(P, Tupp, yi)
      logger.debug(
        'Tupp = %.2f K, the one-phase state is stable: %s',
        Tupp, stabmax.stable,
      )
      if not stabmax.stable:
        Tlow = Tupp
        stab = stabmax
      while not stabmax.stable and Tupp < Tmax:
        Tupp *= c
        stabmax = self.stabsolver.run(P, Tupp, yi)
        logger.debug(
          'Tupp = %.2f K, the one-phase state is stable: %s',
          Tupp, stabmax.stable,
        )
        if not stabmax.stable:
          Tlow = Tupp
          stab = stabmax
      if Tupp > Tmax:
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for temperature and/or `Tmax` parameter.'
        )
      else:
        T = Tlow
    elif stab.stable and not upper:
      logger.debug(
        'Finding the two-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Tlow = T
      c = 1. + step
      Tupp = c * T
      stabmax = self.stabsolver.run(P, Tupp, yi)
      logger.debug(
        'Tupp = %.2f K, the one-phase state is stable: %s',
        Tupp, stabmax.stable,
      )
      while stabmax.stable and Tupp < Tmax:
        Tupp *= c
        stabmax = self.stabsolver.run(P, Tupp, yi)
        logger.debug(
          'Tupp = %.2f K, the one-phase state is stable: %s',
          Tupp, stabmax.stable,
        )
        if stabmax.stable:
          Tlow = Tupp
      if Tupp > Tmax:
        raise ValueError(
          'The two-phase region was not identified. It could be because of\n'
          'its narrowness or absence. Try to change the initial guess for\n'
          'temperature or stability test parameters using the `stabkwargs`\n'
          'It also might be helpful to reduce the value of the `step`.'
        )
      else:
        T = Tupp
        stab = stabmax
    else:
      logger.debug(
        'Finding the one-phase region for the lower-bound curve by the '
        'preliminary search.'
      )
      Tupp = T
      c = 1. - step
      Tlow = c * T
      stabmin = self.stabsolver.run(P, Tlow, yi)
      logger.debug(
        'Tlow = %.2f K, the one-phase state is stable: %s',
        Tlow, stabmin.stable,
      )
      if not stabmin.stable:
        Tupp = Tlow
        stab = stabmin
      while not stabmin.stable and Tlow > Tmin:
        Tlow *= c
        stabmin = self.stabsolver.run(P, Tlow, yi)
        logger.debug(
          'Tlow = %.2f K, the one-phase state is stable: %s',
          Tlow, stabmin.stable,
        )
        if not stabmin.stable:
          Tupp = Tlow
          stab = stabmin
      if Tlow < Tmin:
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for temperature and/or `Tmin` parameter.'
        )
      else:
        T = Tupp
    return T, stab.kvji[0], Tlow, Tupp

  def gridding(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    Tmin: Scalar = 173.15,
    Tmax: Scalar = 973.15,
    upper: bool = True,
    Nnodes: int = 10,
  ) -> tuple[Scalar, Vector, Scalar, Scalar]:
    """Performs the gridding procedure to refine the initial guess of
    the saturation temperature. Instead of evaluating all segments,
    the gridding procedure would terminate as soon as an interval
    exhibiting a change in stability is identified. It is recommended
    to use the gridding procedure if the initial guess of the saturation
    temperature is significantly inaccurate.

    Parameters
    ----------
    P: Scalar
      Pressure of a mixture [Pa].

    T: Scalar
      The initial guess of the saturation temperature to be improved
      [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    Tmin: Scalar
      During the preliminary search, the temperature can not drop below
      the lower limit. Otherwise, the `ValueError` will be rised.
      Default is `173.15` [K].

    Tmax: Scalar
      During the preliminary search, the temperature can not exceed the
      upper limit. Otherwise, the `ValueError` will be rised.
      Default is `973.15` [K].

    upper: bool
      A boolean flag that indicates whether the desired value is located
      at the upper saturation bound or the lower saturation bound.
      The cricondenbar serves as the dividing point between upper and
      lower phase boundaries. This flag controls the start point: if it
      is set to `True`, then the start point would be `Tmax`; otherwise
      it would be `Tmin`. Default is `True`.

    Nnodes: int
      The number of points to construct a grid. Default is `10`.

    Returns
    -------
    A tuple of:
    - the improved initial guess for the saturation temperature [K],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the saturation temperature [K],
    - the upper bound of the saturation temperature [K].

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.
    """
    if upper:
      TT = np.linspace(Tmax, Tmin, Nnodes, endpoint=True)
    else:
      TT = np.linspace(Tmin, Tmax, Nnodes, endpoint=True)
    prevT = TT[0]
    prevresult = self.stabsolver.run(P, prevT, yi)
    prevstable = prevresult.stable
    logger.debug(
      'For T = %.2f K, the one-phase state is stable: %s', prevT, prevstable,
    )
    for nextT in TT[1:]:
      result = self.stabsolver.run(P, nextT, yi)
      stable = result.stable
      logger.debug(
        'For T = %.2f K, the one-phase state is stable: %s', nextT, stable,
      )
      if not stable and prevstable or stable and not prevstable:
        if upper:
          Tlow = nextT
          Tupp = prevT
        else:
          Tlow = prevT
          Tupp = nextT
        if not stable:
          stabres = result
          T0 = nextT
        else:
          stabres = prevresult
          T0 = prevT
        return T0, stabres.kvji[0], Tlow, Tupp
      prevstable = stable
      prevresult = result
      prevT = nextT
    raise ValueError(
      'A boundary of the two-phase region was not identified. It could be\n'
      'because of its narrowness or absence. Try to change the number of\n'
      'points for gridding or stability test parameters using the\n'
      '`stabkwargs` parameter.'
    )


def _TsatPT_solve_TPDeq_T(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  xi: Vector,
  eos: TsatEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 10,
  Tlow0: Scalar = 173.15,
  Tupp0: Scalar = 973.15,
  increasing: bool = True,
) -> tuple[Scalar, Vector, Vector, Scalar, Scalar, Scalar]:
  """Solves the TPD-equation using the PT-based equations of state for
  temperature at a constant pressure. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance, which determines
  the phase appearance or disappearance. A combination of the bisection
  method with Newton's method is used to solve the TPD-equation.

  Parameters
  ----------
  P: Scalar
    Pressure of a mixture [Pa].

  T0: Scalar
    Initial guess of the saturation temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: TsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  tol: Scalar
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-8`.

  Tlow0: Scalar
    The initial lower bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `173.15` [K].

  Tupp0: Scalar
    The initial upper bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `973.15` [K].

  increasing: bool
    A flag that indicates if the TPD-equation vs temperature is an
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
  # print('%3s%9s%10s' % ('Nit', 'T, K', 'TPD'))
  # tmpl = '%3s%9.2f%10.2e'
  k = 0
  Tlow = Tlow0
  Tupp = Tupp0
  Tk = T0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
  lnphixi, Zx, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(tmpl % (k, Tk, TPD))
  while (TPD < -tol or TPD > tol) and k < maxiter:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Tlow = Tk
    else:
      Tupp = Tk
    dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    dT = - TPD / dTPDdT
    Tkp1 = Tk + dT
    if Tkp1 > Tupp or Tkp1 < Tlow:
      Tkp1 = .5 * (Tlow + Tupp)
      dT = Tkp1 - Tk
    if dT > -1e-8 and dT < 1e-8:
      break
    Tk = Tkp1
    k += 1
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    lnphixi, Zx, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(tmpl % (k, Tk, TPD))
  return Tk, lnphixi, lnphiyi, Zx, Zy, TPD


def _TsatPT_ss(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: TsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 200,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Tlow: Scalar = 173.15,
  Tupp: Scalar = 973.15,
  upper: bool = True,
) -> SatResult:
  """Successive substitution (SS) method for the saturation temperature
  calculation using a PT-based equation of state.

  Parameters
  ----------
  P: Scalar
    Pressure of a mixture [Pa].

  T0: Scalar
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: TsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `200`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-8`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `10`.

  Tlow: Scalar
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: Scalar
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info("Saturation temperature calculation using the SS-method.")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, yi =' + Nc * '%7.4f', P, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Tsat, K', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%9.2f%10.2e%11.2e'
  solverTPDeq = partial(_TsatPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tlow0=Tlow, Tupp0=Tupp,
                        increasing=upper)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
  logger.debug(tmpl, k, *lnkik, Tk, g2, TPD)
  while repeat and k < maxiter:
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
    logger.debug(tmpl, k, *lnkik, Tk, g2, TPD)
  if not repeat and np.isfinite(g2) and np.isfinite(Tk):
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
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, Niter=k)
  logger.warning(
    "The SS-method for saturation temperature calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP = %s Pa\nT0 = %s K\n"
    "yi = %s\nTlow = %s K\nTupp = %s K",
    eos.name, P, T0, yi, Tlow, Tupp,
  )
  raise SolutionNotFoundError(
    'The saturation temperature calculation\nterminates unsuccessfully. '
    'Try to increase the maximum number of\nsolver iterations.'
  )


def _TsatPT_qnss(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: TsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 50,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Tlow: Scalar = 173.15,
  Tupp: Scalar = 973.15,
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
  P: Scalar
    Pressure of a mixture [Pa].

  T0: Scalar
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: TsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-8`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `10`.

  Tlow: Scalar
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: Scalar
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info("Saturation temperature calculation using the QNSS-method.")
  Nc = eos.Nc
  logger.info('P = %.1f Pa, yi =' + Nc * '%7.4f', P, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Tsat, K', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%9.2f%10.2e%11.2e'
  solverTPDeq = partial(_TsatPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tlow0=Tlow, Tupp0=Tupp,
                        increasing=upper)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  TPD = ni.dot(gi)
  lmbd = 1.
  repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
  logger.debug(tmpl, k, *lnkik, Tk, g2, TPD)
  while repeat and k < maxiter:
    dlnki = -lmbd * gi
    max_dlnki = np.abs(dlnki).max()
    if max_dlnki > 6.:
      relax = 6. / max_dlnki
      lmbd *= relax
      dlnki *= relax
    k += 1
    tkm1 = dlnki.dot(gi)
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    xi = ni / ni.sum()
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
    logger.debug(tmpl, k, *lnkik, Tk, g2, TPD)
  if not repeat and np.isfinite(g2) and np.isfinite(Tk):
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
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, Niter=k)
  logger.warning(
    "The QNSS-method for saturation temperature calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP = %s Pa\nT0 = %s K\n"
    "yi = %s\nTlow = %s K\nTupp = %s K",
    eos.name, P, T0, yi, Tlow, Tupp,
  )
  raise SolutionNotFoundError(
    'The saturation temperature calculation\nterminates unsuccessfully. '
    'Try to increase the maximum number of\nsolver iterations.'
  )


def _TsatPT_newt_improveT0(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  xi: Vector,
  eos: TsatEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 10,
  Tlow0: Scalar = 173.15,
  Tupp0: Scalar = 973.15,
  increasing: bool = True,
) -> tuple[Scalar, Vector, Scalar, Vector]:
  """Improves initial guess of the saturation temperature by solving
  the TPD-equation using the PT-based equations of state for temperature
  at a constant pressure. The TPD-equation is the equation of equality
  to zero of the tangent-plane distance, which determines the phase
  appearance or disappearance. A combination of the bisection method
  with Newton's method is used to solve the TPD-equation. The algorithm
  is the same as for the `_TsatPT_solve_TPDeq_T` but it differs in
  returned result.

  Parameters
  ----------
  T0: Scalar
    Initial guess of the saturation temperature [K].

  P: Scalar
    Pressure of a mixture [Pa].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: TsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  tol: Scalar
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-8`.

  Tlow0: Scalar
    The initial lower bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is less than the current
    lower bound, then the bisection update would be used. Default is
    `173.15` [K].

  Tupp0: Scalar
    The initial upper bound of the saturation temperature [K]. If the
    saturation temperature at any iteration is greater than the current
    upper bound, then the bisection update would be used. Default is
    `973.15` [K].

  increasing: bool
    A flag that indicates if the TPD-equation vs temperature is an
    increasing function. This parameter is used to control the
    bisection method update. Default is `True`.

  Returns
  -------
  A tuple of:
  - the saturation temperature,
  - natural logarithms of fugacity coefficients of components in the
    mixture as a `Vector` of shape `(Nc,)`,
  - compressibility factor of the mixture,
  - partial derivatives of logarithms of the fugacity coefficients of
    components in the mixture with respect to temperature as a `Vector`
    of shape `(Nc,)`.
  """
  # print('%3s%9s%10s' % ('Nit', 'T, K', 'TPD'))
  # tmpl = '%3s%9.2f%10.2e'
  k = 0
  Tlow = Tlow0
  Tupp = Tupp0
  Tk = T0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
  lnphixi, Zt, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(tmpl % (k, Tk, TPD))
  while (TPD < -tol or TPD > tol) and k < maxiter:
    if TPD < 0. and increasing or TPD > 0. and not increasing:
      Tlow = Tk
    else:
      Tupp = Tk
    dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    dT = - TPD / dTPDdT
    Tkp1 = Tk + dT
    if Tkp1 > Tupp or Tkp1 < Tlow:
      Tkp1 = .5 * (Tlow + Tupp)
      dT = Tkp1 - Tk
    if dT > -1e-8 and dT < 1e-8:
      break
    Tk = Tkp1
    k += 1
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    lnphixi, Zt, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(tmpl % (k, Tk, TPD))
  return Tk, lnphiyi, Zy, dlnphiyidT


def _TsatPT_newtA(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: TsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 20,
  Tlow: Scalar = 173.15,
  Tupp: Scalar = 973.15,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
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
  P: Scalar
    Pressure of a mixture [Pa].

  T0: Scalar
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: TsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    - `getPT_lnphii_Z_dnj_dT(
         P: Scalar, T: Scalar, yi: Vector, n: Scalar,
       ) -> tuple[Vector, Scalar, Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements of
    the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Tlow: Scalar
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: Scalar
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm of the saturation
    temperature initial guess improvement. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation temperature initial guess
    improvement. Default is `10`.

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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info(
    "Saturation temperature calculation using Newton's method (A-form)."
  )
  Nc = eos.Nc
  logger.info('P = %.1f Pa, yi =' + Nc * '%7.4f', P, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Tsat, K', 'g2',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%9.2f%10.2e'
  J = np.zeros(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphiyi, Zy, dlnphiyidT = _TsatPT_newt_improveT0(
    P, T0, yi, xi, eos, tol_tpd, maxiter_tpd, Tlow, Tupp, upper,
  )
  lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(
    P, Tk, xi, n,
  )
  gi[:Nc] = lnkik + lnphixi - lnphiyi
  gi[-1] = n - 1.
  g2 = gi.dot(gi)
  logger.debug(tmpl, k, *lnkik, Tk, g2)
  while g2 > tol and k < maxiter:
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = ni
    J[:Nc,-1] = Tk * (dlnphixidT - dlnphiyidT)
    try:
      dlnkilnT = linsolver(J, -gi)
    except:
      dlnkilnT = -gi
    k += 1
    lnkik += dlnkilnT[:-1]
    Tkp1 = Tk * np.exp(dlnkilnT[-1])
    if Tkp1 > Tupp:
      Tk = .5 * (Tk + Tupp)
    elif Tkp1 < Tlow:
      Tk = .5 * (Tlow + Tk)
    else:
      Tk = Tkp1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(
      P, Tk, xi, n,
    )
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    gi[:Nc] = lnkik + lnphixi - lnphiyi
    gi[-1] = n - 1.
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnkik, Tk, g2)
  if g2 < tol and np.isfinite(g2) and np.isfinite(Tk):
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
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji, g2=g2,
                     TPD=xi.dot(gi[:Nc] - np.log(n)), Niter=k)
  logger.warning(
    "Newton's method (A-form) for saturation temperature calculation termin"
    "ates unsuccessfully. EOS: %s.\nParameters:\nP = %s Pa\nT0 = %s K\n"
    "yi = %s\nTlow = %s K\nTupp = %s K",
    eos.name, P, T0, yi, Tlow, Tupp,
  )
  raise SolutionNotFoundError(
    'The saturation temperature calculation\nterminates unsuccessfully. '
    'Try to increase the maximum number of\nsolver iterations.'
  )


def _TsatPT_newtB(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: TsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 20,
  Tlow: Scalar = 173.15,
  Tupp: Scalar = 973.15,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
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
  P: Scalar
    Pressure of a mixture [Pa].

  T0: Scalar
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: TsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    - `getPT_lnphii_Z_dnj_dT(
         P: Scalar, T: Scalar, yi: Vector, n: Scalar,
       ) -> tuple[Vector, Scalar, Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements of
    the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `20`.

  Tlow: Scalar
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: Scalar
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc+1, Nc+1)` and
    an array `b` of shape `(Nc+1,)` and finds an array `x` of shape
    `(Nc+1,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm of the saturation
    temperature initial guess improvement. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm of the saturation temperature initial guess
    improvement. Default is `10`.

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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info(
    "Saturation temperature calculation using Newton's method (B-form)."
  )
  Nc = eos.Nc
  logger.info('P = %.1f Pa, yi =' + Nc * '%7.4f', P, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Tsat, K', 'g2',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%9.2f%10.2e'
  J = np.empty(shape=(Nc + 1, Nc + 1))
  gi = np.empty(shape=(Nc + 1,))
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphiyi, Zy, dlnphiyidT = _TsatPT_newt_improveT0(
    P, T0, yi, xi, eos, tol_tpd, maxiter_tpd, Tlow, Tupp, upper,
  )
  lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(
    P, Tk, xi, n,
  )
  gi[:Nc] = lnkik + lnphixi - lnphiyi
  hi = gi[:Nc] - np.log(n)
  gi[-1] = xi.dot(hi)
  g2 = gi.dot(gi)
  logger.debug(tmpl, k, *lnkik, Tk, g2)
  while g2 > tol and k < maxiter:
    J[:Nc,:Nc] = I + ni * dlnphixidnj
    J[-1,:Nc] = xi * (hi - gi[-1])
    J[:Nc,-1] = Tk * (dlnphixidT - dlnphiyidT)
    J[-1,-1] = xi.dot(J[:Nc,-1])
    try:
      dlnkilnT = linsolver(J, -gi)
    except:
      dlnkilnT = -gi
    k += 1
    lnkik += dlnkilnT[:-1]
    Tkp1 = Tk * np.exp(dlnkilnT[-1])
    if Tkp1 > Tupp:
      Tk = .5 * (Tk + Tupp)
    elif Tkp1 < Tlow:
      Tk = .5 * (Tlow + Tk)
    else:
      Tk = Tkp1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, Zx, dlnphixidnj, dlnphixidT = eos.getPT_lnphii_Z_dnj_dT(
      P, Tk, xi, n,
    )
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    gi[:Nc] = lnkik + lnphixi - lnphiyi
    hi = gi[:Nc] - np.log(n)
    gi[-1] = xi.dot(hi)
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnkik, Tk, g2)
  if g2 < tol and np.isfinite(g2) and np.isfinite(Tk):
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
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=gi[-1], Niter=k)
  logger.warning(
    "Newton's method (B-form) for saturation temperature calculation termin"
    "ates unsuccessfully. EOS: %s.\nParameters:\nP = %s Pa\nT0 = %s K\n"
    "yi = %s\nTlow = %s K\nTupp = %s K",
    eos.name, P, T0, yi, Tlow, Tupp,
  )
  raise SolutionNotFoundError(
    'The saturation temperature calculation\nterminates unsuccessfully. '
    'Try to increase the maximum number of\nsolver iterations.'
  )


def _TsatPT_newtC(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: TsatEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Tlow: Scalar = 173.15,
  Tupp: Scalar = 973.15,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
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
  P: Scalar
    Pressure of a mixture [Pa].

  T0: Scalar
    Initial guess of the saturation temperature [K]. It should be
    inside the two-phase region.

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: TsatEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements of
    the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of solver iterations. Default is `30`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `10`.

  Tlow: Scalar
    The lower bound for the TPD-equation solver.
    Default is `173.15` [K].

  Tupp: Scalar
    The upper bound for the TPD-equation solver.
    Default is `973.15` [K].

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
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
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the saturation pressure calculation
  procedure terminates unsuccessfully.
  """
  logger.info(
    "Saturation temperature calculation using Newton's method (C-form)."
  )
  Nc = eos.Nc
  logger.info('P = %.1f Pa, yi =' + Nc * '%7.4f', P, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%10s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Tsat, K', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%9.2f%10.2e%11.2e'
  solverTPDeq = partial(_TsatPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Tlow0=Tlow, Tupp0=Tupp,
                        increasing=upper)
  I = np.eye(Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
  logger.debug(tmpl, k, *lnkik, Tk, g2, TPD)
  while repeat and k < maxiter:
    dlnphixidnj = eos.getPT_lnphii_Z_dnj(P, Tk, xi, n)[2]
    J = I + ni * dlnphixidnj
    try:
      dlnki = linsolver(J, -gi)
    except:
      dlnki = -gi
    k += 1
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Tk, lnphixi, lnphiyi, Zx, Zy, TPD = solverTPDeq(P, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    repeat = g2 > tol or TPD < -tol_tpd or TPD > tol_tpd
    logger.debug(tmpl, k, *lnkik, Tk, g2, TPD)
  if not repeat and np.isfinite(g2) and np.isfinite(Tk):
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
    logger.info('Saturation temperature for P = %.1f Pa is: %.2f K.', P, Tk)
    return SatResult(P=P, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, Niter=k)
  logger.warning(
    "Newton's method (C-form) for saturation temperature calculation termin"
    "ates unsuccessfully. EOS: %s.\nParameters:\nP = %s Pa\nT0 = %s K\n"
    "yi = %s\nTlow = %s K\nTupp = %s K",
    eos.name, P, T0, yi, Tlow, Tupp,
  )
  raise SolutionNotFoundError(
    'The saturation temperature calculation\nterminates unsuccessfully. '
    'Try to increase the maximum number of\nsolver iterations.'
  )


class PmaxPT(PsatPT):
  """Cricondenbar calculation.

  Performs the cricondenbar point calculation using PT-based equations
  of state.

  Parameters
  ----------
  eos: PmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> tuple[Vector, ...]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as a tuple of `Vector` of shape `(Nc,)`.

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    - `getPT_lnphii_Z_dT_d2T(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to temperature as a
        `Vector` of shape `(Nc,)`.

    If the solution method would be `'newton'` then it also must have:

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
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
    - `'newton'` (Newton's method).

    Default is `'ss'`.

  step: Scalar
    To specify the confidence interval for pressure of the cricondenbar
    point calculation, the preliminary search is performed. This
    parameter regulates the step of this search in fraction units.
    During the preliminary search, the next value of pressure will be
    calculated from the previous one using the formula:
    `Pnext = Pprev * (1. + step)`. Default is `0.1`.

  lowerlimit: Scalar
    During the preliminary search, the pressure can not drop below
    the lower limit. Otherwise, the `ValueError` will be rised.
    Default is `1.` [Pa].

  upperlimit: Scalar
    During the preliminary search, the pressure can not exceed the
    upper limit. Otherwise, the `ValueError` will be rised.
    Default is `1e8` [Pa].

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval for pressure of the cricondenbar point. This dictionary is
    used to specify arguments for the stability test procedure. Default
    is an empty dictionary.

  **kwargs: dict
    Other arguments for a cricondebar-solver. It may contain such
    arguments as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others
    depending on the selected solver.

  Methods
  -------
  run(P: Scalar, T: Scalar, yi: Vector) -> SatResult
    This method performs the cricondenbar point calculation for a given
    initial guess of cricondenbar pressure [Pa], temperature in [K] and
    mole composition `yi` of shape `(Nc,)`. This method returns
    cricondenbar point calculation results as an instance of
    `SatResult`.

  search(P: Scalar, T: Scalar, yi: Vector) -> tuple[Scalar, Vector,
                                                    Scalar, Scalar]
    This method performs the preliminary search to refine an initial
    guess of the cricondenbar and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the cricondenbar pressure [Pa],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the cricondenbar pressure [Pa],
    - the upper bound of the cricondenbar pressure [Pa].

  gridding(P: Scalar, T: Scalar, yi: Vector) -> tuple[Scalar, Vector,
                                                      Scalar, Scalar]
    This method performs the gridding procedure to refine an initial
    guess of the cricondenbar and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the cricondenbar pressure [Pa],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the cricondenbar pressure [Pa],
    - the upper bound of the cricondenbar pressure [Pa].
  """
  def __init__(
    self,
    eos: PmaxEosPT,
    method: str = 'ss',
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.solver = partial(_PmaxPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_PmaxPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_PmaxPT_newtC, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    search: bool = True,
    **kwargs,
  ) -> SatResult:
    """Performs the cricondenbar point calculation for a mixture. To
    improve the initial guess of pressure, the preliminary search is
    performed.

    Parameters
    ----------
    P: Scalar
      Initial guess of the cricondenbar pressure [Pa].

    T: Scalar
      Initial guess of the cricondenbar temperature [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    search: bool
      A boolean flag that indicates whether the preliminary search or
      gridding procedure should be used to refine an initial guess of
      the cricondenbar calculation. It is advisable to use the gridding
      procedure if the initial guess of the cricondenbar is vastly
      inaccurate. Default is `True`.

    **kwargs: dict
      Parameters for the preliminary search or gridding procedure.
      It may contain key-value pairs to change default values for
      `Pmin`, `Pmax`, `Nnodes` and other depending on the `search`
      flag. Default is an empty dictionary.

    Returns
    -------
    Cricondenbar point calculation results as an instance of the
    `SatResult`. Important attributes are:
    - `P` the cricondenbar pressure in [Pa],
    - `T` the cricondenbar temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.

    SolutionNotFoundError
      The `SolutionNotFoundError` exception will be raised if the
      cricondenbar calculation procedure terminates unsuccessfully.
    """
    if search:
      P0, kvi0, Plow, Pupp = self.search(P, T, yi, upper=True, **kwargs)
    else:
      P0, kvi0, Plow, Pupp = self.gridding(P, T, yi, upper=True, **kwargs)
    return self.solver(P0, T, yi, kvi0, Plow=Plow, Pupp=Pupp)


def _PmaxPT_solve_TPDeq_P(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  xi: Vector,
  eos: PmaxEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 10,
) -> Scalar:
  """Solves the TPD-equation using the PT-based equations of state for
  pressure at a constant temperature. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance functon. Newton's
  method is used to solve the TPD-equation.

  Parameters
  ----------
  P0: Scalar
    Initial guess for pressure [Pa].

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: PmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

  tol: Scalar
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-8`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  Returns
  -------
  The root (pressure) of the TPD-equation.
  """
  # print('%3s%10s%10s' % ('Nit', 'P, Pa', 'TPD'))
  # tmpl = '%3s%10.2e%10.2e'
  k = 0
  Pk = P0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(tmpl % (k, Pk, TPD))
  while (TPD < -tol or TPD > tol) and k < maxiter:
    # dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    dTPDdlnP = Pk * xi.dot(dlnphixidP - dlnphiyidP)
    k += 1
    # Pk -= TPD / dTPDdP
    Pk *= np.exp(-TPD / dTPDdlnP)
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(tmpl % (k, Pk, TPD))
  return Pk


def _PmaxPT_solve_dTPDdTeq_T(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  xi: Vector,
  eos: PmaxEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 10,
) -> tuple[Scalar, Vector, Vector, Scalar, Scalar]:
  """Solves the cricondenbar equation using the PT-based equations of
  state for temperature at a constant pressure. The cricondenbar equation
  is the equation of equality to zero of the partial derivative of the
  tangent-plane distance with respect to temperature. Newton's method is
  used to solve the cricondenbar equation.

  Parameters
  ----------
  P: Scalar
    Pressure of a mixture [Pa].

  T0: Scalar
    Initial guess for temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: PmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT_d2T(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to temperature as a
        `Vector` of shape `(Nc,)`.

  tol: Scalar
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-8`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  Returns
  -------
  A tuple of:
  - the root (temperature) of the cricondenbar equation,
  - natural logarithms of fugacity coefficients of components in the
    trial phase as a `Vector` of shape `(Nc,)`,
  - natural logarithms of fugacity coefficients of components in the
    mixture as a `Vector` of shape `(Nc,)`,
  - compressibility factors for both mixtures.
  - value of the cricondenbar equation.
  """
  # print('%3s%9s%10s' % ('Nit', 'T, K', 'eq'))
  # tmpl = '%3s%9.2f%10.2e'
  k = 0
  Tk = T0
  lnphiyi, Zy, dlnphiyidT, d2lnphiyidT2 = eos.getPT_lnphii_Z_dT_d2T(P, Tk, yi)
  lnphixi, Zx, dlnphixidT, d2lnphixidT2 = eos.getPT_lnphii_Z_dT_d2T(P, Tk, xi)
  eq = xi.dot(dlnphixidT - dlnphiyidT)
  # print(tmpl % (k, Tk, eq))
  while (eq < -tol or eq > tol) and k < maxiter:
    deqdT = xi.dot(d2lnphixidT2 - d2lnphiyidT2)
    k += 1
    Tk -= eq / deqdT
    lnphiyi, Zy, dlnphiyidT, d2lnphiyidT2 = eos.getPT_lnphii_Z_dT_d2T(
      P, Tk, yi,
    )
    lnphixi, Zx, dlnphixidT, d2lnphixidT2 = eos.getPT_lnphii_Z_dT_d2T(
      P, Tk, xi,
    )
    eq = xi.dot(dlnphixidT - dlnphiyidT)
    # print(tmpl % (k, Tk, eq))
  return Tk, lnphixi, lnphiyi, Zx, Zy, eq


def _PmaxPT_ss(
  P0: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: PmaxEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 200,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Plow: Scalar = 1.,
  Pupp: Scalar = 1e8,
) -> SatResult:
  """Successive substitution (SS) method for the cricondenbar
  calculation using a PT-based equation of state. To find the
  cricondenbar, the algorithm solves a system of non-linear equations:

  - `Nc` equations of equilibrium of components in the mixture and in
    the trial phase,
  - the TPD-equation that represents the condition where the
    tangent-plane distance equals zero (this equation is linearized with
    pressure),
  - the cricondenbar equation which is the equation of equality to zero
    of the partial derivative of the tangent-plane distance function
    with respect to temperature (this equation is linearized with
    temperature).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the cricondenbar pressure [Pa].

  T0: Scalar
    Initial guess of the cricondenbar temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: PmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    - `getPT_lnphii_Z_dT_d2T(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to temperature as a
        `Vector` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `200`.

  tol_tpd: Scalar
    Terminate the TPD-equation and the cricondenbar equation solvers
    successfully if the absolute value of the equation is less than
    `tol_tpd`. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondenbar equation solvers. Default is `10`.

  Plow: Scalar
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1.0` [Pa].

  Pupp: Scalar
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1e8` [Pa].

  Returns
  -------
  The cricondenbar point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondenbar point,
  - `T` the temperature in [K] of the cricondenbar point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the cricondenbar calculation procedure
  terminates unsuccessfully.
  """
  logger.info('Cricondenbar calculation using the SS-method.')
  Nc = eos.Nc
  logger.info('yi =' + Nc * ' %6.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P, Pa', 'T, K', 'g2', 'TPD', 'dTPDdT',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%9.2f%10.2e%11.2e%11.2e'
  solverTPDeq = partial(_PmaxPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCBAReq = partial(_PmaxPT_solve_dTPDdTeq_T, eos=eos, tol=tol_tpd,
                         maxiter=maxiter_tpd)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Pk, *_, TPD = _PsatPT_solve_TPDeq_P(
    P0, T0, yi, xi, eos, tol_tpd, maxiter_tpd, Plow, Pupp, True,
  )
  Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdT < -tol_tpd or dTPDdT > tol_tpd)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdT)
  while repeat and k < maxiter:
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Pk = solverTPDeq(Pk, Tk, yi, xi)
    Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    TPD = xi.dot(gi - np.log(n))
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdT < -tol_tpd or dTPDdT > tol_tpd)
    logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdT)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk) and np.isfinite(Tk):
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
    logger.info('The cricondenbar: P = %.1f Pa, T = %.2f K.', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, dTPDdT=dTPDdT, Niter=k)
  logger.warning(
    "The SS-method for cricondenbar calculation terminates unsuccessfully. "
    "EOS: %s.\nParameters:\nP0 = %s Pa\nT0 = %s K\nyi = %s\nPlow = %s Pa\n"
    "Pupp = %s Pa",
    eos.name, P0, T0, yi, Plow, Pupp,
  )
  raise SolutionNotFoundError(
    'The cricondenbar calculation\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations and/or improve the '
    'initial guess.'
  )


def _PmaxPT_qnss(
  P0: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: PmaxEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 50,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Plow: Scalar = 1.,
  Pupp: Scalar = 1e8,
) -> SatResult:
  """Quasi-Newton Successive Substitution (SS) method for the
  cricondenbar calculation using a PT-based equation of state. To find
  the cricondenbar, the algorithm solves a system of non-linear
  equations:

  - `Nc` equations of equilibrium of components in the mixture and in
    the trial phase,
  - the TPD-equation that represents the condition where the
    tangent-plane distance equals zero (this equation is linearized with
    pressure),
  - the cricondenbar equation which is the equation of equality to zero
    of the partial derivative of the tangent-plane distance function
    with respect to temperature (this equation is linearized with
    temperature).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  For the details of the QNSS-method see: 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the cricondenbar pressure [Pa].

  T0: Scalar
    Initial guess of the cricondenbar temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: PmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    - `getPT_lnphii_Z_dT_d2T(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to temperature as a
        `Vector` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: Scalar
    Terminate the TPD-equation and the cricondenbar equation solvers
    successfully if the absolute value of the equation is less than
    `tol_tpd`. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondenbar equation solvers. Default is `10`.

  Plow: Scalar
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1.0` [Pa].

  Pupp: Scalar
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1e8` [Pa].

  Returns
  -------
  The cricondenbar point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondenbar point,
  - `T` the temperature in [K] of the cricondenbar point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the cricondenbar calculation procedure
  terminates unsuccessfully.
  """
  logger.info('Cricondenbar calculation using the QNSS-method.')
  Nc = eos.Nc
  logger.info('yi =' + Nc * '%7.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P, Pa', 'T, K', 'g2', 'TPD', 'dTPDdT'
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%9.2f%10.2e%11.2e%11.2e'
  solverTPDeq = partial(_PmaxPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCBAReq = partial(_PmaxPT_solve_dTPDdTeq_T, eos=eos, tol=tol_tpd,
                         maxiter=maxiter_tpd)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Pk, *_, TPD = _PsatPT_solve_TPDeq_P(P0, T0, yi, xi, eos, tol_tpd,
                                      maxiter_tpd, Plow, Pupp, True)
  Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  lmbd = 1.
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdT < -tol_tpd or dTPDdT > tol_tpd)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdT)
  while repeat and k < maxiter:
    dlnki = -lmbd * gi
    max_dlnki = np.abs(dlnki).max()
    if max_dlnki > 6.:
      relax = 6. / max_dlnki
      lmbd *= relax
      dlnki *= relax
    k += 1
    tkm1 = dlnki.dot(gi)
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Pk = solverTPDeq(Pk, Tk, yi, xi)
    Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    TPD = xi.dot(gi - np.log(n))
    lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdT < -tol_tpd or dTPDdT > tol_tpd)
    logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdT)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk) and np.isfinite(Tk):
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
    logger.info('The cricondenbar: P = %.1f Pa, T = %.2f K.', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, dTPDdT=dTPDdT, Niter=k)
  logger.warning(
    "The QNSS-method for cricondenbar calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT0 = %s K\n"
    "yi = %s\nPlow = %s Pa\nPupp = %s Pa",
    eos.name, P0, T0, yi, Plow, Pupp,
  )
  raise SolutionNotFoundError(
    'The cricondenbar calculation\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations and/or improve the '
    'initial guess.'
  )


def _PmaxPT_newtC(
  P0: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: PmaxEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Plow: Scalar = 1.,
  Pupp: Scalar = 1e8,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> SatResult:
  """This function calculates the cricondenbar point by solving a system
  of nonlinear equations using Newton's method. The system incorporates
  the condition of equal fugacity for all components in both phases, as
  well as the requirements that the tangent-plane distance and its
  partial derivative with respect to temperature are zero. The
  TPD- and cricondenbar equations are solved in the inner while-loop.
  The TPD-equation that represents the condition where the
  tangent-plane distance equals zero (this equation is linearized with
  pressure). The cricondenbar equation is the equation of equality to
  zero of the partial derivative of the tangent-plane distance function
  with respect to temperature (this equation is linearized with
  temperature).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the cricondenbar pressure [Pa].

  T0: Scalar
    Initial guess of the cricondenbar temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: PmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    - `getPT_lnphii_Z_dT_d2T(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to temperature as a
        `Vector` of shape `(Nc,)`.

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`,

    - `name: str`
      The EOS name (for proper logging),

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `30`.

  tol_tpd: Scalar
    Terminate the TPD-equation and the cricondenbar equation solvers
    successfully if the absolute value of the equation is less than
    `tol_tpd`. Default is `1e-8`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondenbar equation solvers. Default is `10`.

  Plow: Scalar
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1.0` [Pa].

  Pupp: Scalar
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `1e8` [Pa].

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  The cricondenbar point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondenbar point,
  - `T` the temperature in [K] of the cricondenbar point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the cricondenbar calculation procedure
  terminates unsuccessfully.
  """
  logger.info("Cricondenbar calculation using Newton's method (C-form).")
  Nc = eos.Nc
  logger.info('yi =' + Nc * '%7.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P, Pa', 'T, K', 'g2', 'TPD', 'dTPDdT'
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%9.2f%10.2e%11.2e%11.2e'
  solverTPDeq = partial(_PmaxPT_solve_TPDeq_P, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCBAReq = partial(_PmaxPT_solve_dTPDdTeq_T, eos=eos, tol=tol_tpd,
                         maxiter=maxiter_tpd)
  I = np.eye(eos.Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Pk, *_, TPD = _PsatPT_solve_TPDeq_P(
    P0, T0, yi, xi, eos,
    tol_tpd, maxiter_tpd, Plow, Pupp, True,
  )
  Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, T0, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdT < -tol_tpd or dTPDdT > tol_tpd)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdT)
  while repeat and k < maxiter:
    dlnphixidnj = eos.getPT_lnphii_Z_dnj(Pk, Tk, xi, n)[2]
    J = I + ni * dlnphixidnj
    try:
      dlnki = linsolver(J, -gi)
    except:
      dlnki = -gi
    k += 1
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Pk = solverTPDeq(Pk, Tk, yi, xi)
    Tk, lnphixi, lnphiyi, Zx, Zy, dTPDdT = solverCBAReq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    TPD = xi.dot(gi - np.log(n))
    g2 = gi.dot(gi)
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdT < -tol_tpd or dTPDdT > tol_tpd)
    logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdT)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk) and np.isfinite(Tk):
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
    logger.info('The cricondenbar: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, dTPDdT=dTPDdT, Niter=k)
  logger.warning(
    "Newton's method (C-form) for cricondenbar calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT0 = %s K\n"
    "yi = %s\nPlow = %s Pa\nPupp = %s Pa",
    eos.name, P0, T0, yi, Plow, Pupp,
  )
  raise SolutionNotFoundError(
    'The cricondenbar calculation\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations and/or improve the '
    'initial guess.'
  )


class TmaxPT(TsatPT):
  """Cricondentherm calculation.

  Performs the cricondentherm point calculation using PT-based equations
  of state.

  Parameters
  ----------
  eos: TmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> tuple[Vector, ...]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as a tuple of `Vector` of shape `(Nc,)`.

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    - `getPT_lnphii_Z_dP_d2P(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method should return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to pressure as a
        `Vector` of shape `(Nc,)`.

    If the solution method would be `'newton'` then it also must have:

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
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
    - `'newton'` (Newton's method).

    Default is `'ss'`.

  stabkwargs: dict
    The stability test procedure is used to locate the confidence
    interval for temperature of the cricondentherm point. This
    dictionary is used to specify arguments for the stability test
    procedure. Default is an empty dictionary.

  **kwargs: dict
    Other arguments for a cricondentherm-solver. It may contain such
    arguments as `tol`, `maxiter`, `tol_tpd`, `maxiter_tpd` or others
    depending on the selected solver.

  Methods
  -------
  run(P: Scalar, T: Scalar, yi: Vector) -> SatResult
    This method performs the cricondentherm point calculation for a
    given pressure [Pa], initial guess of cricondentherm temperature in
    [K] and mole composition `yi` of shape `(Nc,)`. This method returns
    cricondentherm point calculation results as an instance of
    `SatResult`.

  search(P: Scalar, T: Scalar, yi: Vector) -> tuple[Scalar, Vector,
                                                    Scalar, Scalar]
    This method performs the preliminary search to refine an initial
    guess of the cricondentherm and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the cricondentherm temperature [K],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the cricondentherm temperature [K],
    - the upper bound of the cricondentherm temperature [K].

  gridding(P: Scalar, T: Scalar, yi: Vector) -> tuple[Scalar, Vector,
                                                      Scalar, Scalar]
    This method performs the gridding procedure to refine an initial
    guess of the cricondentherm and find lower and upper bounds.
    It returns a tuple of:
    - the improved initial guess for the cricondentherm temperature [K],
    - the initial guess for k-values as a `Vector` of shape `(Nc,)`,
    - the lower bound of the cricondentherm temperature [K],
    - the upper bound of the cricondentherm temperature [K].
  """
  def __init__(
    self,
    eos: TmaxEosPT,
    method: str = 'ss',
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.solver = partial(_TmaxPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.solver = partial(_TmaxPT_qnss, eos=eos, **kwargs)
    elif method == 'newton':
      self.solver = partial(_TmaxPT_newtC, eos=eos, **kwargs)
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P: Scalar,
    T: Scalar,
    yi: Vector,
    search: bool = True,
    **kwargs,
  ) -> SatResult:
    """Performs the cricindentherm point calculation for a mixture. To
    improve an initial guess of temperature, the preliminary search is
    performed.

    Parameters
    ----------
    P: Scalar
      Initial guess of the cricondentherm pressure [Pa].

    T: Scalar
      Initial guess of the cricondentherm temperature [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components.

    search: bool
      A boolean flag that indicates whether the preliminary search or
      gridding procedure should be used to refine an initial guess of
      the cricondentherm calculation. It is advisable to use the
      gridding procedure if the initial guess of the cricondenbar is
      vastly inaccurate. Default is `True`.

    **kwargs: dict
      Parameters for the preliminary search or gridding procedure.
      It may contain key-value pairs to change default values for
      `Tmin`, `Tmax`, `Nnodes` and other depending on the `search`
      flag. Default is an empty dictionary.

    Returns
    -------
    Cricondentherm point calculation results as an instance of the
    `SatResult`. Important attributes are:
    - `P` the cricondentherm pressure in [Pa],
    - `T` the cricondentherm temperature in [K],
    - `yji` the component mole fractions in each phase,
    - `Zj` the compressibility factors of each phase,
    - `success` a boolean flag indicating if the calculation completed
      successfully.

    Raises
    ------
    ValueError
      The `ValueError` exception may be raised if the one-phase or
      two-phase region was not found using a preliminary search.

    SolutionNotFoundError
      The `SolutionNotFoundError` exception will be raised if the
      cricondentherm calculation procedure terminates unsuccessfully.
    """
    if search:
      T0, kvi0, Tlow, Tupp = self.search(P, T, yi, upper=True, **kwargs)
    else:
      T0, kvi0, Tlow, Tupp = self.gridding(P, T, yi, upper=True, **kwargs)
    return self.solver(P, T0, yi, kvi0, Tlow=Tlow, Tupp=Tupp)


def _TmaxPT_solve_TPDeq_T(
  P: Scalar,
  T0: Scalar,
  yi: Vector,
  xi: Vector,
  eos: TmaxEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 10,
) -> Scalar:
  """Solves the TPD-equation using the PT-based equations of state for
  temperature at a constant pressure. The TPD-equation is the equation
  of equality to zero of the tangent-plane distance functon. Newton's
  method is used to solve the TPD-equation.

  Parameters
  ----------
  P: Scalar
    Pressure of a mixture [Pa].

  T0: Scalar
    Initial guess for temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  xi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: TmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

  tol: Scalar
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-8`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  Returns
  -------
  The root (temperature) of the TPD-equation.
  """
  # print('%3s%9s%10s' % ('Nit', 'T, K', 'TPD'))
  # tmpl = '%3s%9.2f%10.2e'
  k = 0
  Tk = T0
  lnkvi = np.log(xi / yi)
  lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
  lnphixi, Zx, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  # print(tmpl % (k, Tk, TPD))
  while (TPD < -tol or TPD > tol) and k < maxiter:
    # dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    dTPDdlnT = Tk * xi.dot(dlnphixidT - dlnphiyidT)
    k += 1
    # Tk -= TPD / dTPDdT
    Tk *= np.exp(-TPD / dTPDdlnT)
    lnphiyi, Zy, dlnphiyidT = eos.getPT_lnphii_Z_dT(P, Tk, yi)
    lnphixi, Zx, dlnphixidT = eos.getPT_lnphii_Z_dT(P, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    # print(tmpl % (k, Tk, TPD))
  return Tk


def _TmaxPT_solve_dTPDdPeq_P(
  P0: Scalar,
  T: Scalar,
  yi: Vector,
  xi: Vector,
  eos: TmaxEosPT,
  tol: Scalar = 1e-8,
  maxiter: int = 10,
) -> tuple[Scalar, Vector, Vector, Scalar, Scalar]:
  """Solves the cricondentherm equation using the PT-based equations of
  state for pressure at a constant temperature. The cricondentherm
  equation is the equation of equality to zero of the partial derivative
  of the tangent-plane distance with respect to pressure. Newton's
  method is used to solve the cricondentherm equation.

  Parameters
  ----------
  P0: Scalar
    Initial guess for pressure [Pa].

  T: Scalar
    Temperature of a mixture [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the mixture.

  yti: Vector, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: TmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP_d2P(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method should return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to pressure as a
        `Vector` of shape `(Nc,)`.

  maxiter: int
    The maximum number of iterations. Default is `10`.

  tol: Scalar
    Terminate successfully if the absolute value of the relative
    pressure change is less than `tol`. Default is `1e-8`.

  Returns
  -------
  A tuple of:
  - the root (pressure) of the cricondentherm equation,
  - an array of the natural logarithms of fugacity coefficients of
    components in the trial phase,
  - the same for the initial composition of the mixture,
  - compressibility factors for both mixtures.
  - value of the cricondentherm equation.
  """
  # print('%3s%10s%10s' % ('Nit', 'P, Pa', 'eq'))
  # tmpl = '%3s%10.2e%10.2e'
  k = 0
  Pk = P0
  lnphiyi, Zy, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_Z_dP_d2P(Pk, T, yi)
  lnphixi, Zx, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_Z_dP_d2P(Pk, T, xi)
  eq = xi.dot(dlnphixidP - dlnphiyidP)
  deqdP = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
  dP = -eq / deqdP
  # print(tmpl % (k, Pk, eq))
  while np.abs(dP) / Pk > tol and k < maxiter:
    k += 1
    Pk += dP
    lnphiyi, Zy, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_Z_dP_d2P(
      Pk, T, yi,
    )
    lnphixi, Zx, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_Z_dP_d2P(
      Pk, T, xi,
    )
    eq = xi.dot(dlnphixidP - dlnphiyidP)
    deqdP = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
    dP = -eq / deqdP
    # print(tmpl % (k, Pk, eq))
  return Pk, lnphixi, lnphiyi, Zx, Zy, eq


def _TmaxPT_ss(
  P0: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: TmaxEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 200,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Tlow: Scalar = 173.15,
  Tupp: Scalar = 973.15,
) -> SatResult:
  """Successive substitution (SS) method for the cricondentherm
  calculation using a PT-based equation of state. To find the
  cricondentherm, the algorithm solves a system of non-linear equations:

  - `Nc` equations of equilibrium of components in the mixture and in
    the trial phase,
  - the TPD-equation that represents the condition where the
    tangent-plane distance equals zero (this equation is linearized with
    temperature),
  - the cricondentherm equation which is the equation of equality to
    zero of the partial derivative of the tangent-plane distance
    function with respect to pressure (this equation is linearized with
    pressure).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the cricondentherm pressure [Pa].

  T0: Scalar
    Initial guess of the cricondentherm temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: TmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    - `getPT_lnphii_Z_dP_d2P(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method should return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to pressure as a
        `Vector` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `200`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Terminate the
    cricondentherm equation solver successfully if the absolute value
    of the relative pressure change is less than `tol_tpd`. Default
    is `1e-8`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondentherm equation solvers. Default is `10`.

  Tlow: Scalar
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `173.15` [K].

  Tupp: Scalar
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `973.15` [K].

  Returns
  -------
  The cricondentherm point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondentherm point,
  - `T` the temperature in [K] of the cricondentherm point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the cricondentherm calculation procedure
  terminates unsuccessfully.
  """
  logger.info("Cricondentherm calculation using the SS-method.")
  Nc = eos.Nc
  logger.info('yi =' + Nc * '%7.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P, Pa', 'T, K', 'g2', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%9.2f%10.2e%11.2e%11.2e'
  solverTPDeq = partial(_TmaxPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCTHERMeq = partial(_TmaxPT_solve_dTPDdPeq_P, eos=eos, tol=tol_tpd,
                           maxiter=maxiter_tpd)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Tk, *_, TPD = _TsatPT_solve_TPDeq_T(
    P0, T0, yi, xi, eos, tol_tpd, maxiter_tpd, Tlow, Tupp, True,
  )
  Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(P0, Tk, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdP)
  while repeat and k < maxiter:
    lnkik -= gi
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Tk = solverTPDeq(Pk, Tk, yi, xi)
    Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    TPD = xi.dot(gi - np.log(n))
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
    logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdP)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk) and np.isfinite(Tk):
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
    logger.info('The cricondentherm: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, dTPDdP=dTPDdP, Niter=k)
  logger.warning(
    "The SS-method for cricondentherm calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT0 = %s K\n"
    "yi = %s\nTlow = %s K\nTupp = %s K",
    eos.name, P0, T0, yi, Tlow, Tupp,
  )
  raise SolutionNotFoundError(
    'The cricondentherm calculation\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations and/or improve the '
    'initial guess.'
  )


def _TmaxPT_qnss(
  P0: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: TmaxEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 50,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Tlow: Scalar = 173.15,
  Tupp: Scalar = 973.15,
) -> SatResult:
  """Quasi-Newton Successive substitution (QNSS) method for the
  cricondentherm calculation using a PT-based equation of state. To find
  the cricondentherm, the algorithm solves a system of non-linear
  equations:

  - `Nc` equations of equilibrium of components in the mixture and in
    the trial phase,
  - the TPD-equation that represents the condition where the
    tangent-plane distance equals zero (this equation is linearized with
    temperature),
  - the cricondentherm equation which is the equation of equality to
    zero of the partial derivative of the tangent-plane distance
    function with respect to pressure (this equation is linearized with
    pressure).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  For the details of the QNSS see: 10.1016/0378-3812(84)80013-8.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the cricondentherm pressure [Pa].

  T0: Scalar
    Initial guess of the cricondentherm temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: TmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    - `getPT_lnphii_Z_dP_d2P(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method should return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to pressure as a
        `Vector` of shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Terminate the
    cricondentherm equation solver successfully if the absolute value
    of the relative pressure change is less than `tol_tpd`. Default
    is `1e-8`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondentherm equation solvers. Default is `10`.

  Tlow: Scalar
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `173.15` [K].

  Tupp: Scalar
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `973.15` [K].

  Returns
  -------
  The cricondentherm point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondentherm point,
  - `T` the temperature in [K] of the cricondentherm point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the cricondentherm calculation procedure
  terminates unsuccessfully.
  """
  logger.info("Cricondentherm calculation using the QNSS-method.")
  Nc = eos.Nc
  logger.info('yi =' + Nc * '%7.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P, Pa', 'T, K', 'g2', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%9.2f%10.2e%11.2e%11.2e'
  solverTPDeq = partial(_TmaxPT_solve_TPDeq_T, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCTHERMeq = partial(_TmaxPT_solve_dTPDdPeq_P, eos=eos, tol=tol_tpd,
                           maxiter=maxiter_tpd)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  xi = ni / ni.sum()
  Tk, *_, TPD = _TsatPT_solve_TPDeq_T(
    P0, T0, yi, xi, eos, tol_tpd, maxiter_tpd, Tlow, Tupp, True,
  )
  Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(P0, Tk, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  lmbd = 1.
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdP)
  while repeat and k < maxiter:
    dlnki = -lmbd * gi
    max_dlnki = np.abs(dlnki).max()
    if max_dlnki > 6.:
      relax = 6. / max_dlnki
      lmbd *= relax
      dlnki *= relax
    k += 1
    tkm1 = dlnki.dot(gi)
    lnkik += dlnki
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Tk = solverTPDeq(Pk, Tk, yi, xi)
    Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    TPD = xi.dot(gi - np.log(n))
    lmbd *= np.abs(tkm1 / (dlnki.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
    logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdP)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk) and np.isfinite(Tk):
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
    logger.info('The cricondentherm: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, dTPDdP=dTPDdP, Niter=k)
  logger.warning(
    "The QNSS-method for cricondentherm calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT0 = %s K\n"
    "yi = %s\nTlow = %s K\nTupp = %s K",
    eos.name, P0, T0, yi, Tlow, Tupp,
  )
  raise SolutionNotFoundError(
    'The cricondentherm calculation\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations and/or improve the '
    'initial guess.'
  )


def _TmaxPT_newtC(
  P0: Scalar,
  T0: Scalar,
  yi: Vector,
  kvi0: Vector,
  eos: TmaxEosPT,
  tol: Scalar = 1e-16,
  maxiter: int = 30,
  tol_tpd: Scalar = 1e-8,
  maxiter_tpd: int = 10,
  Tlow: Scalar = 173.15,
  Tupp: Scalar = 973.15,
  linsolver: Callable[[Matrix, Vector], Vector] = np.linalg.solve,
) -> SatResult:
  """This function calculates the cricondenthern point by solving a
  system of nonlinear equations using Newton's method. The system
  incorporates the condition of equal fugacity for all components in
  both phases, as well as the requirements that the tangent-plane
  distance and its partial derivative with respect to pressure are
  zero. The TPD- and cricondentherm equations are solved in the inner
  while-loop. The TPD-equation that represents the condition where the
  tangent-plane distance equals zero (this equation is linearized with
  temperature). The cricondentherm equation is the equation of equality
  to zero of the partial derivative of the tangent-plane distance
  function with respect to pressure (this equation is linearized with
  pressure).

  For the details of the algorithm see the paper of L.X. Nghiem and
  Y.k. Li: 10.1016/0378-3812(85)90059-7.

  Parameters
  ----------
  P0: Scalar
    Initial guess of the cricondentherm pressure [Pa].

  T0: Scalar
    Initial guess of the cricondentherm temperature [K].

  yi: Vector, shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector, shape (Nc,)
    Initial guess of k-values of `Nc` components.

  eos: TmaxEosPT
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dT(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    - `getPT_lnphii_Z_dP_d2P(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method should return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`,
      - second partial derivatives of logarithms of the fugacity
        coefficients of components with respect to pressure as a
        `Vector` of shape `(Nc,)`.

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      An array of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging),

    - `Nc: int`
      The number of components in the system.

  tol: Scalar
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-16`.

  maxiter: int
    The maximum number of equilibrium equation solver iterations.
    Default is `30`.

  tol_tpd: Scalar
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Terminate the
    cricondentherm equation solver successfully if the absolute value
    of the relative pressure change is less than `tol_tpd`. Default
    is `1e-8`.

  maxiter_tpd: int
    The maximum number of iterations for the TPD-equation and
    cricondentherm equation solvers. Default is `10`.

  Tlow: Scalar
    The lower bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `173.15` [K].

  Tupp: Scalar
    The upper bound for the TPD-equation solver. This parameter is used
    only at the zeroth iteration. Default is `973.15` [K].

  linsolver: Callable[[Matrix, Vector], Vector]
    A function that accepts a matrix `A` of shape `(Nc, Nc)` and
    an array `b` of shape `(Nc,)` and finds an array `x` of shape
    `(Nc,)`, which is the solution of the system of linear equations
    `Ax = b`. Default is `numpy.linalg.solve`.

  Returns
  -------
  The cricondentherm point calculation results as an instance of the
  `SatResult`. Important attributes are:
  - `P` the pressure in [Pa] of the cricondentherm point,
  - `T` the temperature in [K] of the cricondentherm point,
  - `yji` the component mole fractions in each phase,
  - `Zj` the compressibility factors of each phase.

  Raises
  ------
  `SolutionNotFoundError` if the cricondentherm calculation procedure
  terminates unsuccessfully.
  """
  logger.info("Cricondentherm calculation using Newton's method (C-form).")
  Nc = eos.Nc
  logger.info('yi =' + Nc * '%7.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%9s%10s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P, Pa', 'T, K', 'g2', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%9.2f%10.2e%11.2e%11.2e'
  solverTPDeq = partial(_TmaxPT_solve_TPDeq_T, eos=eos,tol=tol_tpd,
                        maxiter=maxiter_tpd)
  solverCTHERMeq = partial(_TmaxPT_solve_dTPDdPeq_P, eos=eos, tol=tol_tpd,
                           maxiter=maxiter_tpd)
  I = np.eye(eos.Nc)
  k = 0
  kik = kvi0
  lnkik = np.log(kik)
  ni = kik * yi
  n = ni.sum()
  xi = ni / n
  Tk, *_, TPD = _TsatPT_solve_TPDeq_T(
    P0, T0, yi, xi, eos, tol_tpd, maxiter_tpd, Tlow, Tupp, True,
  )
  Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(P0, Tk, yi, xi)
  gi = lnkik + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
  logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdP)
  while repeat and k < maxiter:
    dlnphixidnj = eos.getPT_lnphii_Z_dnj(Pk, Tk, xi, n)[2]
    J = I + ni * dlnphixidnj
    try:
      dlnki = linsolver(J, -gi)
    except:
      dlnki = -gi
    lnkik += dlnki
    k += 1
    kik = np.exp(lnkik)
    ni = kik * yi
    n = ni.sum()
    xi = ni / n
    Tk = solverTPDeq(Pk, Tk, yi, xi)
    Pk, lnphixi, lnphiyi, Zx, Zy, dTPDdP = solverCTHERMeq(Pk, Tk, yi, xi)
    gi = lnkik + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    TPD = xi.dot(gi - np.log(n))
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
    logger.debug(tmpl, k, *lnkik, Pk, Tk, g2, TPD, dTPDdP)
  if not repeat and np.isfinite(g2) and np.isfinite(Pk) and np.isfinite(Tk):
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
    logger.info('The cricondentherm: P = %.1f Pa, T = %.2f K', Pk, Tk)
    return SatResult(P=Pk, T=Tk, lnphiji=lnphiji, Zj=Zj, yji=yji,
                     g2=g2, TPD=TPD, dTPDdP=dTPDdP, Niter=k)
  logger.warning(
    "Newton's method (C-form) for cricondentherm calculation terminates "
    "unsuccessfully. EOS: %s.\nParameters:\nP0 = %s Pa\nT0 = %s K\n"
    "yi = %s\nTlow = %s K\nTupp = %s K",
    eos.name, P0, T0, yi, Tlow, Tupp,
  )
  raise SolutionNotFoundError(
    'The cricondentherm calculation\nterminates unsuccessfully. Try to '
    'increase the maximum number of\nsolver iterations and/or improve the '
    'initial guess.'
  )


class EnvelopeResult(dict):
  """Container for phase envelope calculation outputs with
  pretty-printing.

  Attributes
  ----------
  Pk: Vector, shape (Ns,)
    This array includes pressures [Pa] of the phase envelope that
    consists of `Ns` states (points).

  Tk: Vector, shape (Ns,)
    This array includes temperatures [K] of the phase envelope that
    consists of `Ns` states (points).

  kvki: Matrix, shape (Ns, Nc)
    Equilibrium k-values of components along the phase envelope.

  Zkj: Matrix, shape (Ns, Np)
    This two-dimensional array represents compressibility factors of
    phases that are in equilibrium along the `Np`-phase envelope
    consisting of `Ns` states (points).

  Pc: list[Scalar]
    Pressure(s) of critical point(s) [Pa]. It might be empty if none of
    the critical points was found.

  Tc: list[Scalar]
    Temperature(s) of critical point(s) [K]. It might be empty if none
    of the critical points was found.

  Pcb: list[Scalar]
    Pressure(s) of cricondenbar point(s) [Pa]. It might be empty if none
    of the cricondenbars was found.

  Tcb: list[Scalar]
    Temperature(s) of cricondenbar point(s) [K]. It might be empty if
    none of the cricondenbars was found.

  Pct: list[Scalar]
    Pressure(s) of cricondentherm point(s) [Pa]. It might be empty if
    none of the cricondentherms was found.

  Tct: list[Scalar]
    Temperature(s) of cricondentherm point(s) [K]. It might be empty if
    none of the cricondentherms was found.

  succeed: bool
    A flag indicating whether the phase envelope construction procedure
    terminated successfully.
  """
  def __getattr__(self, name: str) -> object:
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self) -> str:
    with np.printoptions(linewidth=np.inf):
      s = (f"Critical point: {self.Pc} Pa, {self.Tc} K\n"
           f"Cricondenbar: {self.Pcb} Pa, {self.Tcb} K\n"
           f"Cricondentherm: {self.Pct} Pa, {self.Tct} K\n")
    return s


def _env2pPT_step(Niter: int, dx: Vector) -> Scalar:
  """Function for step control in the phase envelope construction
  procedure. The step size miltipliers were taken from the paper of
  L. Xu (doi: 10.1016/j.geoen.2023.212058).

  Parameters
  ----------
  Niter: int
    The number of iterations required to achieve the current solution.

  dx: Vector, shape (Nc + 2,)
    Basic variables change between previous and current solutions.

  Returns
  -------
  The step multiplier used to calculate the next value of the specified
  variable.
  """
  if Niter <= 1:
    return 2.
  elif Niter == 2:
    return 1.5
  elif Niter == 3:
    return 1.2
  elif Niter == 4:
    return 0.9
  else:
    return 0.5


class env2pPT(object):
  """Two-phase envelope construction using a PT-based equation of state.

  The approach of the phase envelope calculation is based on the
  algorithms described by M.L. Michelsen with some custom modifications.
  For the math and other details behind the algorithms, see:

  1. M.L. Michelsen. Calculation of phase envelopes and critical points
  for multicomponent mixtures. Fluid Phase Equilibria. 1980. Volume 4.
  Issues 1 - 2. Pages 1 - 10. DOI: 10.1016/0378-3812(80)80001-X.

  2. M.L. Michelsen. A simple method for calculation of approximate phase
  boundaries. 1994. Volume 98. Pages 1 - 11. DOI:
  10.1016/0378-3812(94)80104-5 .

  The algorithm starts with the first saturation pressure calculation
  using the `PsatPT` class. Then a preliminary search using several
  flash calculations by the `flash2pPT` class is implemented to clarify
  an initial guess close for the specified phase mole fraction. After
  that, the system of the phase envelope equations is solved using
  Newton's method described by M.L. Michelsen in the first paper. For
  the second point, the initialization procedure relies on linear
  extrapolation, while cubic extrapolation is used for subsequent
  points. The critical point(s), cricondenbar and cricondentherm, are
  determined through cubic interpolation of the calculated phase
  envelope points. This process considers the relevant conditions that
  define each specific point.

  Parameters
  ----------
  eos: Env2pEosPT
    An initialized instance of a PT-based equation of state. First of
    all, it should contain methods for the first saturation pressure
    calculation:

    - `getPT_kvguess(P: Scalar, T: Scalar,
                     yi: Vector) -> tuple[Vector, ...]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must generate initial
      guesses of k-values as a tuple of `Vector` of shape `(Nc,)`.

    - `getPT_lnphii_Z(P: Scalar,
                      T: Scalar, yi: Vector) -> tuple[Vector, Scalar]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture.

    - `getPT_lnphii_Z_dP(P: Scalar, T: Scalar,
                         yi: Vector) -> tuple[Vector, Scalar, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    If the solution method of the first saturation pressure calculation
    would be one of `'newton'`, `'ss-newton'` or `'qnss-newton'` then it
    also must have:

    - `getPT_lnphii_Z_dnj_dP(P: Scalar, T: Scalar, yi: Vector, n: Scalar,
       ) -> tuple[Vector, Scalar, Matrix, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition,
      this method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`.

    The `flash2pPT` class is used to perform the preliminary search for
    the specified phase mole fraction. If the solution method for flash
    calculations would be one of `'newton'`, `'ss-newton'` or
    `'qnss-newton'` then the instance of an EOS also must have:

    - `getPT_lnphii_Z_dnj(P: Scalar, T: Scalar, yi: Vector,
                          n: Scalar) -> tuple[Vector, Scalar, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`) and phase mole number [mol], this
      method must return a tuple of:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the mixture compressibility factor,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole numbers as a `Matrix` of shape
        `(Nc, Nc)`.

    The solution of the phase envelope equations relies on Newton's
    method, for which the Jacobian is constructed using the partial
    derivatives calculated by the following method.

    - `getPT_lnphii_Z_dP_dT_dyj(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole fractions without taking into
        account the mole fraction constraint as a `Matrix` of shape
        `(Nc, Nc)`.

    For solving equations that describe the approximate phase envelope,
    the following partial derivatives are required from the initialized
    instance of an EOS:

    - `getPT_lnphii_Z_dP_dT(P: Scalar, T: Scalar, yi: Vector
       ) -> tuple[Vector, Scalar, Vector, Vector]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`.

    Also, this instance must have attributes:

    - `mwi: Vector`
      A vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

    - `Nc: int`
      The number of components in the system.

  approx: bool
    A flag indicating whether the approximate phase envelope should be
    calculated. If set to `False`, the standard algorithm of the phase
    envelope construction, which is based on the paper of M.L. Michelsen
    (doi: 10.1016/0378-3812(80)80001-X), will be used. If set to `True`
    the phase envelope construction procedure implements the solution
    of equations of the approximate phase envelope. All points of the
    approximate phase envelope are located inside and close to the
    actual two-phase region boundary. The accuracy of the internal
    phase lines is expected to be poorer than that of the phase
    boundary. Therefore, this approach is most substantial for systems
    with many components. This algorithm is based on the paper of
    M.L. Michelsen (10.1016/0378-3812(94)80104-5). Activating this flag
    currently raises `NotImplementedError`. Default is `False`.

  method: str
    This parameter allows to select a solution algorithm for the phase
    envelope construction. It should be one of:

    - `'newton'` (Newton's method),
    - `'tr'` (the trust region method, currently raises
              `NotImplementedError`).

    Default is `'newton'`.

  Pmin: Scalar
    The minimum pressure [Pa] for phase envelope construction.
    This limit is also used by the saturation point solver. Default is
    `1.0` [Pa].

  Pmax: Scalar
    The maximum pressure [Pa] for phase envelope construction.
    This limit is also used by the saturation point solver. Default is
    `1e8` [Pa].

  Tmin: Scalar
    The minimum temperature [K] for phase envelope construction.
    This limit is also used by the saturation point solver. Default is
    `173.15` [K].

  Tmax: Scalar
    The maximum temperature [K] for phase envelope construction.
    This limit is also used by the saturation point solver. Default is
    `937.15` [K].

  stopunstab: bool
    The flag indicates whether it is necessary to stop the phase
    envelope construction if any of trial and real phase compositions
    along it are found unstable by the stability test. Enabling this
    option may prevent drawing false saturation lines but takes extra
    CPU time to conduct stability test for each point on the phase
    envelope. Default is `False`. This option is not implemented yet.

  stabkwargs: dict
    In order to perform the stability test of trial and real phase
    compositions the stability test is conducted using the `stabilityPT`
    class. This parameter allows to specify settings for this class.

  psatkwargs: dict
    The first saturation point of the phase envelope is calculated
    using `PsatPT` class. This parameter allows to specify settings
    for the first saturation pressure calculation.

  flashkwargs: dict
    To clarify the initial guess of the first phase envelope point,
    a preliminary search is implemented using several flash calculations
    launched by the `flash2pPT` class. This parameter allows to specify
    settings for the flash calculation procedure.

  **kwargs: dict
    Other arguments for the two-phase envelope solver. It may contain
    such arguments as `tolres`, `tolvar`, `maxiter` and `miniter`.

  Methods
  -------
  run(P0: Scalar, T0: Scalar, yi: Vector, Fv: Scalar) -> EnvelopeResult
    This method should be used to run the envelope construction program,
    for which the initial guess of the saturation pressure in [Pa],
    starting temperature in [K], mole fractions of `Nc` components with
    shape `(Nc,)`, and phase mole fraction must be given. It returns the
    phase envelope construction results as an instance of the
    `EnvelopeResult`.
  """
  def __init__(
    self,
    eos: Env2pEosPT,
    approx: bool = False,
    method: str = 'newton',
    Pmin: Scalar = 1.,
    Pmax: Scalar = 1e8,
    Tmin: Scalar = 173.15,
    Tmax: Scalar = 973.15,
    stopunstab: bool = False,
    stabkwargs: dict = {},
    psatkwargs: dict = {},
    flashkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    self.psatsolver = PsatPT(eos, **psatkwargs)
    self.flashsolver = flash2pPT(eos, **flashkwargs)
    lnPmin = np.log(Pmin)
    lnPmax = np.log(Pmax)
    lnTmin = np.log(Tmin)
    lnTmax = np.log(Tmax)
    self.lnTmin = lnTmin
    self.lnPmin = lnPmin
    self.lnTmax = lnTmax
    self.lnPmax = lnPmax
    self.stopunstab = stopunstab
    if approx:
      raise NotImplementedError(
        'The construction of the approximate phase envelope is not '
        'implemented yet.'
      )
    else:
      if method == 'newton':
        self.solver = partial(_env2pPT, eos=eos, lnPmax=lnPmax+0.05,
                              lnTmax=lnTmax+0.05, lnPmin=lnPmin-0.05,
                              lnTmin=lnTmin-0.05, **kwargs)
      elif method == 'tr':
        raise NotImplementedError(
          'The trust region method for phase envelope construction is not '
          'implemented yet.'
        )
      else:
        raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    P0: Scalar,
    T0: Scalar,
    yi: Vector,
    Fv: Scalar,
    sidx0: int | None = None,
    step0: Scalar = 0.01,
    fstep: Callable[[int, Vector], Scalar] = _env2pPT_step,
    maxstep: Scalar = 0.25,
    switchmult: Scalar = 0.5,
    unconvmult: Scalar = 0.75,
    maxrepeats: int = 8,
    maxpoints: int = 200,
    searchkwargs: dict = {},
  ) -> EnvelopeResult:
    """This method should be used to calculate the entire phase envelope.

    Parameters
    ----------
    P0: Scalar
      Initial guess of the saturation pressure [Pa].

    T0: Scalar
      Initial guess of the saturation temperature [K].

    yi: Vector, shape (Nc,)
      Mole fractions of `Nc` components in the mixture.

    Fv: Scalar
      Phase mole fraction for which the envelope is needed.

    sidx0: int | None
      For iteration zero, this parameter indexes the specified variable
      in an array of basic variables. The array of basic variables
      includes:

      - `Nc` natural logarithms of k-values of components,
      - the natural logarithm of pressure,
      - the natural logarithm of temperature.

      The specified variable is considered known and fixed for the
      algorithm of saturation point determination. Therefore, changing
      of this index may improve the algorithm converegence for the
      zeroth and subsequent points. To initiate calculations, a pressure
      specification was recommended by M.L. Michelsen in his paper
      (doi: 10.1016/0378-3812(80)80001-X). The general rule for
      specified variable selection was also given in this paper. It was
      recommended to select the specified variable based on the largest
      rate of change, which refers to the largest derivative of the
      basic variables with respect to the specified variable. When
      `sidx0` is set to `None`, it will desigante the least volatile
      component on the bubble point line and the most volatile component
      on the dew point line. Default is `None`.

    step0: Scalar
      The step size (the difference between two subsequent values of a
      specified variable) for iteration zero. It should be small enough
      to consider the saturation point found at the zeroth iteration as
      a good initial guess for the next saturation point calculation.
      Default is `0.01`.

    fstep: Callable[[int, Vector], Scalar]
      Function for step control in the phase envelope construction
      procedure. It should accept the number of iterations and basic
      variables change between previous and current solutions as a
      `Vector` of shape `(Nc + 2,)` and return the step size
      multiplier. Default is `_env2pPT_step`.

    maxstep: Scalar
      The step size in the phase envelope construction procedure cannot
      be greater than the value specified by the `maxstep` parameter.
      Default is `0.25`.

      Step size control determined by the `fstep` and `maxstep`
      parameters is the most significant aspect of the succeed phase
      envelope construction. If step sizes are too small, the algorithm
      may get stuck near the critical region, requiring larger step
      sizes to jump over it. In the opposite case, if a step size is
      too big, then there is a risk that an initial guess generated by
      cubic or linear extrapolation will not be good enough.

    switchmult: Scalar
      The step size multiplier is implemented if the specified variable
      is changed. Default is `0.5`.

    unconvmult: Scalar
      The step size multiplier is implemented if the solution is not
      found for the specified variable value. Default is `0.75`.

    maxrepeats: int
      The saturation point calculation can be repeated several times
      with a reduced step size if convergence of equations was not
      achieved. This parameter allows to specify the maximum number
      of repeats for any saturation point calculation. If the number
      of repeats exceeds the given bound, then the construction of
      the phase envelope will be stopped. Default is `8`.

    maxpoints: int
      The maximum number of points of the phase envelope. Default is
      `200`.

    searchkwargs: dict
      Advanced parameters for the preliminary search. It can contain
      such keys as `Pmin`, `Pmax and `step`. For the details, see the
      method `run` of the class `PsatPT`. Default is an empty
      dictionary.

    Returns
    -------
    The phase envelope construction results as an instance of the
    `EnvelopeResult`.

    Raises
    ------
    `SolutionNotFoundError` if the solution was not found for the
    zeroth saturation point (for the given initial guesses).
    """
    try:
      psres = self.psatsolver.run(P0, T0, yi, True, **searchkwargs)
    except SolutionNotFoundError:
      raise SolutionNotFoundError(
        'The saturation pressure was not found for the specified\n'
        f'starting temperature {T0 = } K and the initial guess {P0 = } Pa.\n'
        'It may be beneficial to modify the initial guess and/or the\n'
        'starting temperature. Changing the solution method and its\n'
        'numerical settings can also be helpful.'
      )
    kvji0 = psres.yji[0] / psres.yji[1], psres.yji[1] / psres.yji[0]
    P, flash = self.search(psres.P, T0, yi, Fv, kvji0)
    F = np.abs(flash.Fj[0])
    xi = np.log(np.hstack([flash.kvji[0], P, T0]))

    logger.info('Constructing the phase envelope for F = %.3f.', F)
    Nc = self.eos.Nc
    Ncp2 = Nc + 2
    xki = np.zeros(shape=(maxpoints * 2, Ncp2))
    Zkj = np.zeros(shape=(maxpoints * 2, 2))
    M = np.empty(shape=(4, 4))
    M[:, 0] = np.array([1., 0., 1., 0.])
    M[1, 1] = 1.
    M[3, 1] = 1.
    B = np.empty(shape=(4, Ncp2))
    mdgds = np.zeros(shape=(Ncp2,))
    mdgds[-1] = 1.
    crits = []
    crcbars = []
    crctrms = []

    logger.info(
      '%4s%5s%6s%8s%5s%10s' + Nc * '%9s' + '%9s%8s',
      'Npnt', 'Ncut', 'Niter', 'Step', 'Sidx', 'Sval',
      *['lnkv%s' % s for s in range(Nc)], 'lnP', 'lnT',
    )
    tmpl = '%4s%5s%6s%8.4f%5s%10.4f' + Nc * '%9.4f' + '%9.4f%8.4f'

    c = 0
    cmax = maxpoints - 1
    k = cmax
    if sidx0 is None:
      if F > .5:
        s0_idx = np.argmax(flash.kvji[0])
      else:
        s0_idx = np.argmin(flash.kvji[0])
    else:
      s0_idx = sidx0
    s0_val = xi[s0_idx]
    x0, Z0j, J0, nit, flg = self.solver(xi, s0_idx, s0_val, yi, F)
    if flg:
      logger.info(tmpl, c, 0, nit, 0., s0_idx, s0_val, *x0)
      xki[k] = x0
      Zkj[k] = Z0j
      dx0ds = np.linalg.solve(J0, mdgds)
    else:
      raise SolutionNotFoundError(
        '...'
      )

    bounds = []
    for k_dir in [-1, 1]:
      k = cmax
      r = 0
      s_cnt = 1
      lnP = x0[-2]
      lnT = x0[-1]
      xk = x0
      Jk = J0
      sk_idx = s0_idx
      sk_val = s0_val
      dxkds = dx0ds
      B[0] = xk
      B[1] = dxkds
      step = step0
      skp1_idx = np.argmax(np.abs(dxkds))
      skp1_val = sk_val + k_dir * step
      xi = xk + dxkds * (skp1_val - sk_val)
      while (lnT >= self.lnTmin and lnT <= self.lnTmax and
             lnP >= self.lnPmin and lnP <= self.lnPmax and c < cmax):
        if r > maxrepeats:
          logger.warning(
            'The maximum number of step repeats has been reached.'
          )
          break
        xkp1, Zj, Jkp1, nit, flg = self.solver(xi, skp1_idx, skp1_val, yi, F)
        if flg:
          r = 0
          xkm1 = xk
          Jkm1 = Jk
          dxkm1ds = dxkds
          k += k_dir
          xk = xkp1
          Jk = Jkp1
          xki[k] = xk
          Zkj[k] = Zj
          c += 1
          lnP = xk[-2]
          lnT = xk[-1]
          logger.info(tmpl, c, r, nit, step, skp1_idx, skp1_val, *xk)
          dxkds = np.linalg.solve(Jk, mdgds)
          if (xk[:Nc] * xkm1[:Nc] < 0.).all():
            if sk_idx == skp1_idx and skp1_idx < Nc:
              lnk_idx = sk_idx
              dlnPTkds = dxkds[Nc:]
              dlnPTkm1ds = dxkm1ds[Nc:]
            elif skp1_idx < Nc:
              lnk_idx = skp1_idx
              dlnPTkds = dxkds[Nc:]
              Jkm1[-1, sk_idx] = 0.
              Jkm1[-1, lnk_idx] = 1.
              dlnPTkm1ds = np.linalg.solve(Jkm1, mdgds)[Nc:]
            elif sk_idx < Nc:
              lnk_idx = sk_idx
              Jk[-1, skp1_idx] = 0.
              Jk[-1, lnk_idx] = 1.
              dlnPTkds = np.linalg.solve(Jk, mdgds)[Nc:]
              dlnPTkm1ds = dxkm1ds[Nc:]
            else:
              lnk_idx = 0
              Jk[-1, skp1_idx] = 0.
              Jk[-1, lnk_idx] = 1.
              dlnPTkds = np.linalg.solve(Jk, mdgds)[Nc:]
              Jkm1[-1, sk_idx] = 0.
              Jkm1[-1, lnk_idx] = 1.
              dlnPTkm1ds = np.linalg.solve(Jkm1, mdgds)[Nc:]
            lnkk_val = xk[lnk_idx]
            lnkkm1_val = xkm1[lnk_idx]
            self._update_M(lnkk_val, lnkkm1_val, M)
            b = np.vstack([xk[Nc:], dlnPTkds, xkm1[Nc:], dlnPTkm1ds])
            crits.append(np.linalg.solve(M, b)[0])
          sk_idx = skp1_idx
          skp1_idx = np.argmax(np.abs(dxkds))
          if skp1_idx != sk_idx:
            s_cnt = 0
            step *= switchmult
          else:
            s_cnt += 1
            step *= fstep(nit, xk - xkm1)
            if np.abs(step) > maxstep:
              step = np.sign(step) * maxstep
          skm1_val = xkm1[skp1_idx]
          sk_val = xk[skp1_idx]
          self._update_M(sk_val, skm1_val, M)
          self._update_B(xk, dxkds, B)
          skp1_val = sk_val + step * np.sign(sk_val - skm1_val)
          if s_cnt > 1:
            C = np.linalg.solve(M, B)
            skp1_val2 = skp1_val * skp1_val
            skp1_val3 = skp1_val2 * skp1_val
            xi = np.array([1., skp1_val, skp1_val2, skp1_val3]).dot(C)
          else:
            xi = xk + (xk - xkm1) / (sk_val - skm1_val) * (skp1_val - sk_val)
        else:
          step *= unconvmult
          if s_cnt > 1:
            skp1_val = sk_val + step * np.sign(sk_val - skm1_val)
            skp1_val2 = skp1_val * skp1_val
            skp1_val3 = skp1_val2 * skp1_val
            xi = np.array([1., skp1_val, skp1_val2, skp1_val3]).dot(C)
          elif c > 0:
            skp1_val = sk_val + step * np.sign(sk_val - skm1_val)
            xi = xk + (xk - xkm1) / (sk_val - skm1_val) * (skp1_val - sk_val)
          else:
            skp1_val = sk_val + k_dir * step0
            xi = xk + dxkds * (skp1_val - sk_val)
          r += 1
      bounds.append(k)
    bounds[-1] += 1
    slc = slice(*bounds)
    Pk = np.exp(xki[slc, -2])
    Tk = np.exp(xki[slc, -1])
    lnkvki = xki[slc, :Nc]
    if crits:
      crits = np.exp(crits)
      Pc = crits[:, 0]
      Tc = crits[:, 1]
    else:
      Pc = None
      Tc = None
    return EnvelopeResult(Pk=Pk, Tk=Tk, kvki=np.exp(lnkvki), Zkj=Zkj,
                          Pc=Pc, Tc=Tc, succeed=flg)

  def search(
    self,
    Pmax: Scalar,
    T: Scalar,
    yi: Vector,
    Fv: Scalar,
    kvji0: Iterable[Vector] | None = None,
    Pmin: Scalar = 101325.,
    Npoints: int = 100,
  ) -> tuple[Scalar, FlashResult]:
    """The preliminary search is conducted to identify an initial guess
    that is close to the specified phase mole fraction by performing
    flash calculations along the pressure axis.

    Parameters
    ----------
    Pmax: Scalar
      The upper bound of the pressure interval. As a general rule, it
      should be equal to the saturation pressure for a given temperature
      and composition.

    T: Scalar
      Temperature of the mixture.

    yi: Vector, shape (Nc,)
      Mole fractions of components in the mixture.

    Fv: Scalar
      Phase mole fraction close to which the initial guess is needed.

    kvji0: Iterable[Vector] | None
      An iterable object of initial guesses of k-values for flash
      calculations. Default is `None`.

    Pmin: Scalar
      The lower bound of the pressure interval. Default is
      `101325.0` [Pa].

    Npoint: int
      The number of points into which the pressure interval is divided.
      Default is `100`.

    Returns
    -------
    A tuple that contains:
    - pressure [Pa] at which the system is split, resulting in a phase
      mole fraction that closely resembles the specified value,
    - flash calculation results as an instance of `FlashResult`.
    """
    PP = np.linspace(Pmax, Pmin, Npoints, endpoint=True)
    flashs = []
    Fvs = []
    for P in PP:
      flash = self.flashsolver.run(P, T, yi, kvji0)
      flashs.append(flash)
      Fvs.append(flash.Fj[0])
      if np.isclose(flash.Fj, Fv).any():
        return P, flash
    Fvs = np.array(Fvs)
    idx = np.argmin(np.abs(Fvs - Fv))
    return PP[idx], flashs[idx]

  @staticmethod
  def _update_M(sk_val, skm1_val, M):
    skm1_val2 = skm1_val * skm1_val
    sk_val2 = sk_val * sk_val
    M[0, 1] = sk_val
    M[0, 2] = sk_val2
    M[0, 3] = sk_val2 * sk_val
    M[1, 2] = 2. * sk_val
    M[1, 3] = 3. * sk_val2
    M[2, 1] = skm1_val
    M[2, 2] = skm1_val2
    M[2, 3] = skm1_val2 * skm1_val
    M[3, 2] = 2. * skm1_val
    M[3, 3] = 3. * skm1_val2
    pass

  @staticmethod
  def _update_B(xk, dxkds, B):
    B[2:] = B[:2]
    B[0] = xk
    B[1] = dxkds
    pass


def _env2pPT(
  x0: Vector,
  sidx: int,
  sval: Scalar,
  yi: Vector,
  Fv: Scalar,
  eos: Env2pEosPT,
  tolres: Scalar = 1e-12,
  tolvar: Scalar = 1e-14,
  maxiter: int = 5,
  miniter: int = 0,
  lnPmin: Scalar = 0.,
  lnPmax: Scalar = 18.42,
  lnTmin: Scalar = 5.154,
  lnTmax: Scalar = 6.881,
) -> tuple[Vector, Vector, Matrix, int, bool]:
  """Sovles phase envelope equations using Newton's method.

  Parameters
  ----------
  x0: Vector, shape (Nc + 2,)
    Initial guess of basic variables. The array of basic variables
    includes:
      - `Nc` natural logarithms of k-values of components,
      - the natural logarithm of pressure,
      - the natural logarithm of temperature.

  sidx: int
    The specified variable index. The specified variable is considered
    known and fixed for the algorithm of saturation point determination.

  yi: Vector, shape (Nc,)
    Mole fractions of components in the mixture.

  Fv: Scalar
    Phase mole fraction for which the phase envelope should be
    constructed.

  eos: Env2pEosPT
    An inizialized instance of an equation of state. Solution of the
    phase envelope equations relies on Newton's method, for which the
    Jacobian is constructed using the partial derivatives calculated by
    the following method:

    - `getPT_lnphii_Z_dP_dT_dyj(P: Scalar, T: Scalar, yi: Vector,
       ) -> tuple[Vector, Scalar, Vector, Vector, Matrix]`
      For a given pressure [Pa], temperature [K] and mole composition
      (`Vector` of shape `(Nc,)`), this method must return a tuple that
      contains:

      - logarithms of the fugacity coefficients of components as a
        `Vector` of shape `(Nc,)`,
      - the phase compressibility factor of the mixture,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to pressure as a `Vector` of shape
        `(Nc,)`,
      - partial derivatives of logarithms of the fugacity coefficients
        of components with respect to temperature as a `Vector` of
        shape `(Nc,)`,
      - partial derivatives of logarithms of the fugacity coefficients
        with respect to components mole fractions without taking into
        account the mole fraction constraint as a `Matrix` of shape
        `(Nc, Nc)`.

    Also, this instance must have attributes:

    - `Nc: int`
      The number of components in the system.

  tolres: Scalar
    Terminate successfully if the norm of the phase envelope equations
    vector is less than `tolres`. Default is `1e-12`.

  tolvar: Scalar
    Terminate successfully if the sum of squared elements of the
    direction vector is less than `tolvar`. Default is `1e-14`.

  maxiter: int
    The maximum number of iterations. Default is `5`.

  miniter: int
    The minimum number of iterations. Default is `0`.

  lnPmin: Scalar
    Minimum natural logarithm of pressure. Default is `0.0`.

  lnPmax: Scalar
    Maximum natural logarithm of pressure. Default is `18.42`
    (~ 1e8 [Pa]).

  lnTmin: Scalar
    Minimum natural logarithm of temperature. Default is `5.154`
    (~ 173.12 [K]).

  lnTmax: Scalar
    Maximum natural logarithm of temperature. Default is `6.881`
    (~ 937.60 [K]).

  Returns
  -------
  A tuple that contains:
  - the solution of the phase envelope equations as a `Vector` of shape
    `(Nc + 2,)`,
  - compressibility factors of phases as a `Vector` of shape `(2,)`,
  - jacobian (at the solution) as a `Matrix` of shape
    `(Nc + 2, Nc + 2)`,
  - the number of iterations to converge,
  - a boolean flag indicating whether any of the solution conditions
    were satisfied.
  """
  Nc = eos.Nc
  logger.debug(
    'Solving the system of phase boundary equations for: '
    'Fv = %.3f, sidx = %s, sval = %.4f', Fv, sidx, sval,
  )
  logger.debug(
    '%3s' + Nc * '%9s' + '%9s%8s%10s%10s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'lnP', 'lnT', 'gnorm', 'dx2',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%9.4f%8.4f%10.2e%10.2e'
  J = np.zeros(shape=(Nc + 2, Nc + 2))
  J[-1, sidx] = 1.
  g = np.empty(shape=(Nc + 2,))
  I = np.eye(Nc)
  k = 0
  xk = x0.flatten()
  xk[sidx] = sval
  ex = np.exp(xk)
  P = ex[-2]
  T = ex[-1]
  lnkvi = xk[:Nc]
  kvi = ex[:Nc]
  di = 1. + Fv * (kvi - 1.)
  yli = yi / di
  yvi = kvi * yli
  (lnphivi, Zv, dlnphividP,
   dlnphividT, dlnphividyvj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yvi)
  (lnphili, Zl, dlnphilidP,
   dlnphilidT, dlnphilidylj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yli)
  g[:Nc] = lnkvi + lnphivi - lnphili
  g[Nc] = np.sum(yvi - yli)
  g[Nc+1] = xk[sidx] - sval
  gnorm = np.linalg.norm(g)
  dylidlnkvi = -Fv * yvi / di
  dyvidlnkvi = yvi + kvi * dylidlnkvi
  J[:Nc,:Nc] = I + dlnphividyvj * dyvidlnkvi - dlnphilidylj * dylidlnkvi
  J[-2,:Nc] = yvi / di
  J[:Nc,-2] = P * (dlnphividP - dlnphilidP)
  J[:Nc,-1] = T * (dlnphividT - dlnphilidT)
  dx = np.linalg.solve(J, -g)
  dx2 = dx.dot(dx)
  repeat = (np.isfinite(dx2) and
            (dx2 > tolvar and gnorm > tolres or k < miniter))
  logger.debug(tmpl, k, *xk, gnorm, dx2)
  while repeat and k < maxiter:
    k += 1
    xkp1 = xk + dx
    if xkp1[-2] > lnPmax:
      xkp1[-2] = .5 * (xk[-2] + lnPmax)
    elif xkp1[-2] < lnPmin:
      xkp1[-2] = .5 * (xk[-2] + lnPmin)
    if xkp1[-1] > lnTmax:
      xkp1[-1] = .5 * (xk[-1] + lnTmax)
    elif xkp1[-1] < lnTmin:
      xkp1[-1] = .5 * (xk[-1] + lnTmin)
    xk = xkp1
    ex = np.exp(xk)
    P = ex[-2]
    T = ex[-1]
    lnkvi = xk[:Nc]
    kvi = ex[:Nc]
    di = 1. + Fv * (kvi - 1.)
    yli = yi / di
    yvi = kvi * yli
    (lnphivi, Zv, dlnphividP,
     dlnphividT, dlnphividyvj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yvi)
    (lnphili, Zl, dlnphilidP,
     dlnphilidT, dlnphilidylj) = eos.getPT_lnphii_Z_dP_dT_dyj(P, T, yli)
    g[:Nc] = lnkvi + lnphivi - lnphili
    g[Nc] = np.sum(yvi - yli)
    g[Nc+1] = xk[sidx] - sval
    gnorm = np.linalg.norm(g)
    dylidlnkvi = -Fv * yvi / di
    dyvidlnkvi = yvi + kvi * dylidlnkvi
    J[:Nc,:Nc] = I + dlnphividyvj * dyvidlnkvi - dlnphilidylj * dylidlnkvi
    J[-2,:Nc] = yvi / di
    J[:Nc,-2] = P * (dlnphividP - dlnphilidP)
    J[:Nc,-1] = T * (dlnphividT - dlnphilidT)
    dx = np.linalg.solve(J, -g)
    dx2 = dx.dot(dx)
    repeat = (np.isfinite(dx2) and
              (dx2 > tolvar and gnorm > tolres or k < miniter))
    logger.debug(tmpl, k, *xk, gnorm, dx2)
  succeed = (gnorm < tolres or dx2 < tolvar) and np.isfinite(dx2)
  return xk, np.array([Zv, Zl]), J, k, succeed


def _aenv2pPT(
  x0: Vector,
  alpha: Vector,
  lnkpi: Vector,
  Fv: Scalar,
  zi: Vector,
  eos: Env2pEosPT,
  tolres: Scalar = 1e-12,
  tolvar: Scalar = 1e-14,
  maxiter: int = 5,
  miniter: int = 1,
  lnPmin: Scalar = 0.,
  lnPmax: Scalar = 18.42,
  lnTmin: Scalar = 5.154,
  lnTmax: Scalar = 6.881,
) -> tuple[Vector, Vector]:
  """
  """
  k = 0
  logger.debug(
    'Solving the system of equations of the approximate phase boundary for: '
    'Fv = %.3f, alpha = %.3f', Fv, alpha,
  )
  logger.debug('%3s%9s%8s%10s%10s', 'Nit', 'lnP', 'lnT', 'gnorm', 'dx2')
  tmpl = '%3s %8.4f %7.4f %9.2e %9.2e'
  dx = np.empty_like(x0)
  Yi = zi * np.exp(alpha * lnkpi)
  yi = Yi / Yi.sum()
  lnkvi = np.log(yi / zi)
  xk = x0
  P = np.exp(xk[0])
  T = np.exp(xk[1])
  lnphiyi, Zy, dlnphiyidP, dlnphiyidT = eos.getPT_lnphii_Z_dP_dT(P, T, yi)
  lnphizi, Zz, dlnphizidP, dlnphizidT = eos.getPT_lnphii_Z_dP_dT(P, T, zi)
  di = lnkvi + lnphiyi - lnphizi
  g1 = yi.dot(di)
  g2 = zi.dot(di)
  gnorm = np.sqrt(g1 * g1 + g2 * g2)
  ddidP = dlnphiyidP - dlnphizidP
  ddidT = dlnphiyidT - dlnphizidT
  dg1dlnP = P * yi.dot(ddidP)
  dg2dlnP = P * zi.dot(ddidP)
  dg1dlnT = T * yi.dot(ddidT)
  dg2dlnT = T * zi.dot(ddidT)
  D = 1. / (dg1dlnP * dg2dlnT - dg1dlnT * dg2dlnP)
  dx[0] = D * (dg1dlnT * g2 - dg2dlnT * g1)
  dx[1] = D * (dg2dlnP * g1 - dg1dlnP * g2)
  dx2 = dx.dot(dx)
  proceed = (np.isfinite(dx2) and
             (dx2 > tolvar and gnorm > tolres or k < miniter))
  logger.debug(tmpl, k, *xk, gnorm, dx2)
  while dx2 > tol and k < maxiter:
    xkp1 = xk + dx
    if xkp1[0] > lnPmax:
      xk[0] = .5 * (xk[0] + lnPmax)
    elif xkp1[0] < lnPmin:
      xk[0] = .5 * (xk[0] + lnPmin)
    else:
      xk[0] = xkp1[0]
    if xkp1[1] > lnTmax:
      xk[1] = .5 * (xk[1] + lnTmax)
    elif xkp1[1] < lnTmin:
      xk[1] = .5 * (xk[1] + lnTmin)
    else:
      xk[1] = xkp1[1]
    k += 1
    P = np.exp(xk[0])
    T = np.exp(xk[1])
    lnphiyi, Zy, dlnphiyidP, dlnphiyidT = eos.getPT_lnphii_Z_dP_dT(P, T, yi)
    lnphizi, Zz, dlnphizidP, dlnphizidT = eos.getPT_lnphii_Z_dP_dT(P, T, zi)
    di = lnkvi + lnphiyi - lnphizi
    g1 = yi.dot(di)
    g2 = zi.dot(di)
    gnorm = np.sqrt(g1 * g1 + g2 * g2)
    ddidP = dlnphiyidP - dlnphizidP
    ddidT = dlnphiyidT - dlnphizidT
    dg1dlnP = P * yi.dot(ddidP)
    dg2dlnP = P * zi.dot(ddidP)
    dg1dlnT = T * yi.dot(ddidT)
    dg2dlnT = T * zi.dot(ddidT)
    D = 1. / (dg1dlnP * dg2dlnT - dg1dlnT * dg2dlnP)
    dx[0] = D * (dg1dlnT * g2 - dg2dlnT * g1)
    dx[1] = D * (dg2dlnP * g1 - dg1dlnP * g2)
    dx2 = dx.dot(dx)
    proceed = dx2 > tolvar and gnorm > tolres and np.isfinite(dx2)
    logger.debug(tmpl, k, *xk, gnorm, dx2)
  return np.hstack([lnkvi, xk]), np.array([Zy, Zz])
