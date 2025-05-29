import logging

from functools import (
  partial,
)

import numpy as np

from utils import (
  mineig_rayquot,
)

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

  Performs saturation pressure calculation for PT-based equations of
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
      This method should return a tuple of logarithms of the fugacity
      coefficients of components and the phase compressibility factor.

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      This method should return a tuple of logarithms of the fugacity
      coefficients of components, the phase compressibility factor and
      the partial derivatives of the logarithms of the fugacity
      coefficients of components with respect to pressure.

    If the solution method would be one of `'newton'` or `'ss-newton'`
    then it also must have:

    - `getPT_lnphii_Z_dnj(P, T, yi) -> tuple[ndarray, float, ndarray]`
      This method should return a tuple of logarithms of the fugacity
      coefficients, the mixture compressibility factor, and partial
      derivatives of logarithms of the fugacity coefficients with
      respect to components mole numbers which are an ndarray of
      shape `(Nc, Nc)`.

    If the `improve_P0` is set to `True` and the `stabgrid` is set to
    `False` then it also must have:

    - `getPT_lnphii(P, T, yi) -> ndarray`
      This method must return logarithms of the fugacity coefficients
      of components.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      Vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  method: str
    Type of the solver. Should be one of:

    - `'ss'` (Successive Substitution method),
    - `'qnss'` (Quasi-Newton Successive Substitution method),
    - `'bfgs'` (Currently raises `NotImplementedError`),
    - `'newton'` (Currently raises `NotImplementedError`),
    - `'ss-newton'` (Currently raises `NotImplementedError`),
    - `'qnss-newton'` (Currently raises `NotImplementedError`).

    Default is `'ss'`.

  improve_P0: bool
    Flag indicating whether or not to improve the initial guess of the
    saturation pressure. The improvement method depends on the flag
    `stabgrid`. Default is `False`.

  stabgrid: bool
    Flag indicating which method should be used to improve the initial
    guess of the saturation pressure. This flag affects the program
    flow only if the `improve_P0` was set `True`. If the `stabgrid` was
    set `True` then `Npoint` of stability tests will be performed to
    find the pressure, closest to the given and located inside the
    two-phase region. If the `stabgrid` was set `False` then all
    possible roots of the TPD-equation for fixed real and trial
    compositions would be found by splitting the interval from `Pmin` to
    `Pmax` into `(Npoint - 1)` segments. The closest solution of the
    TPD-equation to the given value would be used as an initial guess for
    the saturation pressure calculation. Default is `False`.

  stabkwargs: dict
    Dictionary that used to regulate the stability test procedure.
    Default is an empty dictionary.

  kwargs: dict
    Other arguments for the equilibrium equations solver and the
    TPD-equation solver. It may contain such arguments as `tol`,
    `maxiter`, `tol_tpd`, `maxiter_tpd` or others.

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
    improve_P0: bool = False,
    stabgrid: bool = False,
    stabkwargs: dict = {},
    **kwargs,
  ) -> None:
    self.eos = eos
    self.improve_P0 = improve_P0
    self.stabgrid = stabgrid
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.psatsolver = partial(_PsatPT_ss, eos=eos, **kwargs)
    elif method == 'qnss':
      self.psatsolver = partial(_PsatPT_qnss, eos=eos, **kwargs)
    elif method == 'bfgs':
      raise NotImplementedError(
        'The BFGS-method for the saturation pressure calculation is not '
        'implemented yet.'
      )
    elif method == 'newton':
      raise NotImplementedError(
        "The Newton's method for the saturation pressure calculation is not "
        "implemented yet."
      )
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
    P0: ScalarType = 1e5,
    Pmax: ScalarType = 1e8,
    Pmin: ScalarType = 1.,
    Npoint: int = 10,
  ) -> SatResult:
    """Performs the saturation pressure calculation for given temperature
    and composition.

    Parameters
    ----------
    T: float
      Temperature of a mixture [K].

    yi: ndarray, shape (Nc,)
      Mole fractions of `Nc` components.

    P0: float
      Initial guess of the saturation pressure [Pa]. It should be inside
      the two-phase region unless the `improve_P0` was activated.
      Default is `1e5` [Pa].

    Pmax: float
      Upper bound for the TPD-solver [Pa]. It also used to contruct
      the grid for the algorithm of the initial guess improvement. If it
      was completed successfully then the value of the upper bound will
      be also changed. Default is `1e8` [Pa].

    Pmin: float
      Lower bound for the TPD-solver [Pa]. It also used to contruct
      the grid for the algorithm of the initial guess improvement. If it
      was completed successfully then the value of the lower bound will
      be also changed. Default is `1.0` [Pa].

    Npoint: int
      The number of points for grid construction routines. It affects
      the program flow only if the option of the saturation pressure
      initial guess improvement was activated by `improve_P0`. Default
      is `10`.

    Returns
    -------
    Saturation pressure calculation results as an instance of the
    `SatResult`. Important attributes are: `P` the saturation pressure
    in [Pa], `T` the saturation temperature in [K], `yji` the component
    mole fractions in each phase, `Zj` the compressibility factors of
    each phase, `success` a boolean flag indicating if the calculation
    completed successfully.
    """
    if self.improve_P0:
      if self.stabgrid:
        logger.debug(
          'Improvement of the saturation pressure initial guess by finding\n'
          '\tthe closest pressure to the given in the two-phase region...'
        )
        PP = np.linspace(Pmin, Pmax, Npoint, endpoint=True)
        stabs = []
        where = []
        for P in PP:
          stab = self.stabsolver.run(P, T, yi)
          stabs.append(stab)
          where.append(not stab.stable)
        if np.any(where):
          idx = np.where(where, np.abs(PP - P0), np.inf).argmin()
          P0 = PP[idx]
          stab = stabs[idx]
          if idx > 0:
            Pmin = PP[idx-1]
          if idx < Npoint - 1:
            Pmax = PP[idx+1]
          logger.debug(
            'Improved initial guess:\n\tP0 = %s, Pmin = %s, Pmax = %s',
            P0, Pmin, Pmax,
          )
        else:
          logger.warning(
            'The stability test gridding procedure was unsuccessful.\n'
            '\tTry to increase the number of points.'
          )
          stab = self.stabsolver.run(P0, T, yi)
      else:
        logger.debug(
          'Improvement of the saturation pressure initial guess by\n'
          '\tfinding the roots of the TPD-equation for fixed real\n'
          '\tand trial compositions...'
        )
        stab = self.stabsolver.run(P0, T, yi)
        if stab.stable:
          raise ValueError(
            'The initial guess is outside of the two-phase region.\n'
            'Try to change the value of P0 or stability test parameters\n'
            'using the `stabkwargs` argument. Also this problem can be\n'
            'solved by activating the `improve_P0` and `stabgrid` flags.'
          )
        P0s = []
        Pmins = []
        Pmaxs = []
        PP = np.linspace(Pmin, Pmax, Npoint, endpoint=True)
        for kvi in stab.kvji:
          lnkvi = np.log(kvi)
          yti = kvi * yi
          ftpd = np.vectorize(
            lambda P: yti.dot(lnkvi + self.eos.getPT_lnphii(P, T, yti)
                              - self.eos.getPT_lnphii(P, T, yi))
          )
          tpds = ftpd(PP)
          for i in range(Npoint-1):
            if tpds[i] * tpds[i+1] < 0.:
              P0t, *_, suc = _solveTPDeqPT(.5*(PP[i]+PP[i+1]), T, yi, yti,
                                           self.eos, Pmax=PP[i+1], Pmin=PP[i])
              if suc:
                P0s.append(P0t)
                Pmins.append(PP[i])
                Pmaxs.append(PP[i+1])
        if P0s:
          idx = np.abs(np.asarray(P0s) - P0).argmin()
          P0 = P0s[idx]
          Pmin = Pmins[idx]
          Pmax = Pmaxs[idx]
          logger.debug(
            'Improved initial guess:\n\tP0 = %s, Pmin = %s, Pmax = %s',
            P0, Pmin, Pmax,
          )
          stab = self.stabsolver.run(P0, T, yi)
        else:
          logger.warning(
            'The TPD-equation gridding procedure completed unsuccessfully. '
            'Try to increase the number of points or use another initial '
            'guess improvement method.'
          )
    else:
      stab = self.stabsolver.run(P0, T, yi)
    if stab.stable:
      raise ValueError(
        'The initial guess of the P0 is outside of the two-phase region.\n'
        'Try to change the value of P0 or stability test parameters\n'
        'using the `stabkwargs` argument. Also this problem can be solved\n'
        'by activating the `improve_P0` and `stabgrid` flags.'
      )
    return self.psatsolver(P0, T, yi, stab, Pmin=Pmin, Pmax=Pmax)


def _solveTPDeqPT(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  yti: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 8,
  Pmax: ScalarType = 1e8,
  Pmin: ScalarType = 1.,
) -> tuple[ScalarType, VectorType, VectorType, ScalarType, ScalarType, bool]:
  """Solves the TPD-equation for the PT-based equations of state.
  The TPD-equation is the equation of equality to zero of the
  tangent-plane distance, which determines the phase appearance or
  disappearance. The Newton's method is used to solve the TPD-equation.

  Parameters
  ----------
  P0: float
    Initial guess of the saturation pressure [Pa].

  T: float
    Temperature of a mixture [K].

  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  yti: ndarray, shape (Nc,)
    Mole fractions of `Nc` components in the trial phase.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      This method should return a tuple of logarithms of the fugacity
      coefficients of components, the phase compressibility factor and
      the partial derivatives of the logarithms of the fugacity
      coefficients of components with respect to pressure. The
      arguments are: `P: float` is pressure [Pa], `T: float` is
      temperature [K], and `yi: ndarray`, shape `(Nc,)` is an array of
      components mole fractions.

  maxiter: int
    Maximum number of iterations. Default is `8`.

  tol: float
    Terminate successfully if the absolute value of the equation is less
    than `tol`. Default is `1e-6`.

  Pmax: float
    Upper bound of the saturation pressure [Pa]. If the saturation
    pressure at any iteration is greater than `Pmax`, then the bisection
    update would be used. Default is `1e8` [Pa].

  Pmin: float
    Lower bound of the saturation pressure [Pa]. If the saturation
    pressure at any iteration is less than `Pmin`, then the bisection
    update would be used. Default is `1.0` [Pa].

  Returns
  -------
  A tuple of the saturation pressure, an array of the natural logarithms
  of components in the trial phase and in the mixture, their
  compressibility factors and a flag indicating if the equation was
  solved successfully.
  """
  logger.debug('Solving the TPD equation:')
  k = 0
  Pk = P0
  lnkvi = np.log(yti / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, yti)
  TPD = yti.dot(lnkvi + lnphixi - lnphiyi)
  while (np.abs(TPD) > tol) and (k < maxiter):
    dTPDdP = yti.dot(dlnphixidP - dlnphiyidP)
    # dTPDdlnP = Pk * dTPDdP
    # dlnP = - TPD / dTPDdlnP
    # Pkp1 = Pk * np.exp(dlnP)
    dP = - TPD / dTPDdP
    Pkp1 = Pk + dP
    logger.debug('Iteration #%s:\n\tP = %s\n\tTPD = %s', k, Pk, TPD)
    if Pkp1 > Pmax:
      Pk = .5 * (Pk + Pmax)
    elif Pkp1 < Pmin:
      Pk = .5 * (Pmin + Pk)
    else:
      Pk = Pkp1
    k += 1
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, yti)
    TPD = yti.dot(lnkvi + lnphixi - lnphiyi)
  logger.debug('Iteration #%s:\n\tP = %s\n\tTPD = %s', k, Pk, TPD)
  return Pk, lnphixi, lnphiyi, Zx, Zy, k < maxiter


def _PsatPT_ss(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  stab0: StabResult,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Pmax: ScalarType = 1e8,
  Pmin: ScalarType = 1.,
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

  stab0: StabResult
    An instance of the `StabResult` with results of the stability test
    for the initial guess of saturation pressure.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      This method should return a tuple of logarithms of the fugacity
      coefficients of components, the phase compressibility factor and
      the partial derivatives of the logarithms of the fugacity
      coefficients of components with respect to pressure. The
      arguments are: `P: float` is pressure [Pa], `T: float` is
      temperature [K], and `yi: ndarray`, shape `(Nc,)` is an array of
      components mole fractions.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      Vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    Maximum number of TPD-equation solver iterations. Default is `8`.

  Pmax: float
    Upper bound for the TPD-equation solver. Default is `1e8` [Pa].

  Pmin: float
    Lower bound for the TPD-equation solver. Default is `1.0` [Pa].

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are: `P` the saturation pressure
  in [Pa], `T` the saturation temperature in [K], `yji` the component
  mole fractions in each phase, `Zj` the compressibility factors of
  each phase, `success` a boolean flag indicating if the calculation
  completed successfully.
  """
  logger.debug(
    'Saturation pressure calculation using the SS-method:\n'
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s', P0, T, yi,
  )
  solverTPDeq = partial(_solveTPDeqPT, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Pmax=Pmax, Pmin=Pmin)
  k = 0
  xi = stab0.yti
  kvik = xi / yi
  lnkvik = np.log(kvi)
  Pk = P0
  lnphixi = stab0.lnphiyti
  lnphiyi = stab0.lnphiyi
  Zx = stab0.Zt
  Zy = stab0.Z
  ni = xi
  n = ni.sum()
  TPD = -np.log(n)
  gi = lnkvik + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tPk = %s',
    k, kvik, gnorm, Pk,
  )
  while (gnorm > tol) and (k < maxiter):
    dlnkvi = - gi
    k += 1
    lnkvik += dlnkvi
    kvik = np.exp(lnkvik)
    ni = kvi * yi
    n = ni.sum()
    TPD = -np.log(n)
    xi = ni / n
    Pk, lnphixi, lnphiyi, Zx, Zy, _ = solverTPDeq(Pk, T, yi, xi)
    gi = lnkvik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tPk = %s',
      k, kvik, gnorm, Pk,
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
      "Saturation pressure calculation terminates unsuccessfully. "
      "The solution method was SS, EOS: %s. Parameters:"
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
  stab0: StabResult,
  eos: EOSPTType,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 8,
  Pmax: ScalarType = 1e8,
  Pmin: ScalarType = 1.,
) -> SatResult:
  """QNSS-method for the saturation pressure calculation using a PT-based
  equation of state.

  Performs the Quasi-Newton Successive Substitution (QNSS) method to
  find an equilibrium state by solving a system of non-linear equations.
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

  stab0: StabResult
    An instance of the `StabResult` with results of the stability test
    for the initial guess of saturation pressure.

  eos: EOSPTType
    An initialized instance of a PT-based equation of state. Must have
    the following methods:

    - `getPT_lnphii_Z_dP(P, T, yi) -> tuple[ndarray, float, ndarray]`
      This method should return a tuple of logarithms of the fugacity
      coefficients of components, the phase compressibility factor and
      the partial derivatives of the logarithms of the fugacity
      coefficients of components with respect to pressure. The
      arguments are: `P: float` is pressure [Pa], `T: float` is
      temperature [K], and `yi: ndarray`, shape `(Nc,)` is an array of
      components mole fractions.

    Also, this instance must have attributes:

    - `mwi: ndarray`
      Vector of components molecular weights [kg/mol] of shape
      `(Nc,)`.

    - `name: str`
      The EOS name (for proper logging).

  tol: float
    Terminate equilibrium equation solver successfully if the norm of
    the equation vector is less than `tol`. Default is `1e-5`.

  maxiter: int
    Maximum number of equilibrium equation solver iterations.
    Default is `50`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol`. Default is `1e-6`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance, which determines the second phase
    appearance or disappearance.

  maxiter_tpd: int
    Maximum number of TPD-equation solver iterations. Default is `8`.

  Pmax: float
    Upper bound for the TPD-equation solver. Default is `1e8` [Pa].

  Pmin: float
    Lower bound for the TPD-equation solver. Default is `1.0` [Pa].

  Returns
  -------
  Saturation pressure calculation results as an instance of the
  `SatResult`. Important attributes are: `P` the saturation pressure
  in [Pa], `T` the saturation temperature in [K], `yji` the component
  mole fractions in each phase, `Zj` the compressibility factors of
  each phase, `success` a boolean flag indicating if the calculation
  completed successfully.
  """
  logger.debug(
    'Saturation pressure calculation using the QNSS-method:\n'
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s', P0, T, yi,
  )
  solverTPDeq = partial(_solveTPDeqPT, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd, Pmax=Pmax, Pmin=Pmin)
  k = 0
  xi = stab0.yti
  kvik = xi / yi
  lnkvik = np.log(kvik)
  Pk = P0
  lnphixi = stab0.lnphiyti
  lnphiyi = stab0.lnphiyi
  Zx = stab0.Zt
  Zy = stab0.Z
  ni = xi
  n = ni.sum()
  TPD = -np.log(n)
  gi = np.log(kvik) + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  lmbd = 1.
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tlmbd = %s\n\tPk = %s',
    k, kvik, gnorm, lmbd, Pk,
  )
  while (gnorm > tol) and (k < maxiter):
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
    n = ni.sum()
    TPD = -np.log(n)
    xi = ni / n
    Pk, lnphixi, lnphiyi, Zx, Zy, _ = solverTPDeq(Pk, T, yi, xi)
    gi = lnkvik + lnphixi - lnphiyi
    gnorm = np.linalg.norm(gi)
    lmbd *= np.abs(tkm1 / (dlnkvi.dot(gi) - tkm1))
    if lmbd > 30.:
      lmbd = 30.
    logger.debug(
      'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tlmbd = %s\n\tPk = %s',
      k, kvik, gnorm, lmbd, Pk,
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
      "Saturation pressure calculation terminates unsuccessfully. "
      "The solution method was QNSS, EOS: %s. Parameters:"
      "\n\tP0 = %s Pa, T = %s K\n\tyi = %s\n\tPmin = %s Pa\n\tPmax = %s Pa",
      eos.name, P0, T, yi, Pmin, Pmax,
    )
    return SatResult(P=Pk, T=T, lnphiji=np.vstack([lnphixi, lnphiyi]),
                     Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                     success=False)

