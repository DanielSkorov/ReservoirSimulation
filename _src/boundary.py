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

  Arguments:
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
  k: int = 0
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
  if k < maxiter:
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

  Arguments:
  ----------
  yi: ndarray, shape (Nc,)
    Mole fractions of `Nc` components.

  eos: EOSVTType
    An initialized instance of a VT-based equation of state.

  v0: float
    An initial guess for the critical molar volume of a mixture.
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
  k: int = 0
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
  if k < maxiter:
    return eos.getVT_P(vk, T, yi), T
  logger.warning(
    "The critical point was not found using %s:"
    "\n\tyi = %s\n\tv0 = %s\n\tT0 = %s\n\tkappa0 = %s\n\tmultdV0 = %s"
    "\n\tkrange = %s\n\tkstep = %s\n\ttol = %s\n\tmaxiter = %s",
    eos.name, yi, v0, T0, kappa0, multdV0, krange, kstep, tol, maxiter,
  )
  raise ValueError(
    "The critical point solution procedure completed unsuccessfully. "
    "Try to increase the number of iterations or change the initial guess."
  )


class PsatResult(dict):
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e

  def __repr__(self):
    with np.printoptions(linewidth=np.inf):
      s = (f"Saturation pressure: {self.P} Pa\n"
           f"Phase composition:\n{self.yji}\n"
           f"Phase compressibility factors:\n{self.Zj}\n"
           f"Calculation completed successfully:\n{self.success}")
    return s


class PsatPT(object):
  def __init__(
    self,
    eos: EOSPTType,
    method: str = 'ss',
    tol: ScalarType = 1e-5,
    maxiter: int = 50,
    tol_tpd: ScalarType = 1e-6,
    maxiter_tpd: int = 10,
    improve_P0: bool = True,
    stabkwargs: dict = {},
  ) -> None:
    self.eos = eos
    self.tol = tol
    self.maxiter = maxiter
    self.improve_P0 = improve_P0
    self.stabsolver = stabilityPT(eos, **stabkwargs)
    if method == 'ss':
      self.psatsolver = partial(_PsatPT_ss, eos=eos, improve_P0=improve_P0,
                                tol=tol, maxiter=maxiter, tol_tpd=tol_tpd,
                                maxiter_tpd=maxiter_tpd)
    elif method == 'qnss':
      self.psatsolver = partial(_PsatPT_qnss, eos=eos, improve_P0=improve_P0,
                                tol=tol, maxiter=maxiter, tol_tpd=tol_tpd,
                                maxiter_tpd=maxiter_tpd)
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
    else:
      raise ValueError(f'The unknown method: {method}.')
    pass

  def run(
    self,
    T: ScalarType,
    yi: VectorType,
    P0: ScalarType = 1e5,
  ) -> PsatResult:
    logger.debug(
      'Initial stability test for\n\tP = %s Pa\n\tT = %s K\n\tyi = %s',
      P0, T, yi,
    )
    stab = self.stabsolver.run(P0, T, yi)
    logger.debug(
      'TPD = %s\n\tThe system is stable: %s\n\tComputation succeeded: %s',
      stab.TPD, stab.stable, stab.success,
    )
    if stab.stable:
      raise NotImplementedError(
        'The gridding procedure is not implemented yet. '
        'Try to reduce the value of P0.'
      )
    if not stab.success:
      raise ValueError(
        'The stability test for the initial value of P0 completed '
        'unsuccessfully. Try to change stabmethod or increase level.'
      )
    return self.psatsolver(P0, T, yi, stab)


def _solveTPDeqPT(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  yti: VectorType,
  eos: EOSPTType,
  tol: ScalarType = 1e-6,
  maxiter: int = 10,
) -> tuple[ScalarType, VectorType, VectorType, ScalarType, ScalarType]:
  logger.debug('Solving the TPD equation:')
  k = 0
  Pk = P0.copy()
  lnkvi = np.log(yti / yi)
  lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
  lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, yti)
  TPD = yti.dot(lnkvi + lnphixi - lnphiyi)
  while (np.abs(TPD) > tol) and (k < maxiter):
    lnP = np.log(Pk)
    dTPDdP = yti.dot(dlnphixidP - dlnphiyidP)
    dTPDdlnP = Pk * dTPDdP
    dlnP = - TPD / dTPDdlnP
    logger.debug('Iteration #%s:\n\tP = %s\n\tTPD = %s', k, Pk, TPD)
    lnP += dlnP
    k += 1
    Pk = np.exp(lnP)
    lnphiyi, Zy, dlnphiyidP = eos.getPT_lnphii_Z_dP(Pk, T, yi)
    lnphixi, Zx, dlnphixidP = eos.getPT_lnphii_Z_dP(Pk, T, yti)
    TPD = yti.dot(lnkvi + lnphixi - lnphiyi)
  logger.debug('Iteration #%s:\n\tP = %s\n\tTPD = %s', k, Pk, TPD)
  return Pk, lnphixi, lnphiyi, Zx, Zy


def _PsatPT_ss(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  stab0: StabResult,
  eos: EOSPTType,
  improve_P0: bool = True,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 10,
) -> PsatResult:
  logger.debug(
    'Saturation pressure calculation using the SS-method:\n'
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s', P0, T, yi,
  )
  solverTPDeq = partial(_solveTPDeqPT, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  k = 0
  xi = stab0.yti
  kvik = xi / yi
  lnkvi = np.log(kvik)
  if improve_P0:
    Pk, lnphixi, lnphiyi, Zx, Zy = solverTPDeq(P0, T, yi, xi)
  else:
    Pk = P0.copy()
    lnphixi = stab0.lnphiyti
    lnphiyi = stab0.lnphiyi
    Zx = stab0.Zt
    Zy = stab0.Z
  ni = xi
  n = ni.sum()
  TPD = -np.log(n)
  gi = np.log(kvik) + lnphixi - lnphiyi
  gnorm = np.linalg.norm(gi)
  logger.debug(
    'Iteration #%s:\n\tkvi = %s\n\tgnorm = %s\n\tPk = %s',
    k, kvik, gnorm, Pk,
  )
  while (gnorm > tol) and (k < maxiter):
    dlnkvi = - gi
    k += 1
    kvik *= np.exp(dlnkvi)
    ni = kvik * yi
    n = ni.sum()
    TPD = -np.log(n)
    xi = ni / n
    Pk, lnphixi, lnphiyi, Zx, Zy = solverTPDeq(Pk, T, yi, xi)
    gi = np.log(kvik) + lnphixi - lnphiyi
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
    return PsatResult(P=Pk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation terminates unsuccessfully. "
      "The solution method was SS, EOS: %s. Parameters:"
      "\n\tP0 = %s, T = %s\n\tyi = %s\n\timprove_P0 = %s",
      eos.name, P0, T, yi, improve_P0
    )
    return PsatResult(P=Pk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                      Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                      success=False)


def _PsatPT_qnss(
  P0: ScalarType,
  T: ScalarType,
  yi: VectorType,
  stab0: StabResult,
  eos: EOSPTType,
  improve_P0: bool = True,
  tol: ScalarType = 1e-5,
  maxiter: int = 50,
  tol_tpd: ScalarType = 1e-6,
  maxiter_tpd: int = 10,
) -> PsatResult:
  logger.debug(
    'Saturation pressure calculation using the QNSS-method:\n'
    '\tP0 = %s Pa\n\tT = %s K\n\tyi = %s', P0, T, yi,
  )
  solverTPDeq = partial(_solveTPDeqPT, eos=eos, tol=tol_tpd,
                        maxiter=maxiter_tpd)
  k = 0
  xi = stab0.yti
  kvik = xi / yi
  lnkvi = np.log(kvik)
  if improve_P0:
    Pk, lnphixi, lnphiyi, Zx, Zy = solverTPDeq(P0, T, yi, xi)
  else:
    Pk = P0.copy()
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
    kvik *= np.exp(dlnkvi)
    ni = kvik * yi
    n = ni.sum()
    TPD = -np.log(n)
    xi = ni / n
    Pk, lnphixi, lnphiyi, Zx, Zy = solverTPDeq(Pk, T, yi, xi)
    gi = np.log(kvik) + lnphixi - lnphiyi
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
    return PsatResult(P=Pk, lnphiji=lnphiji, Zj=Zj, yji=yji, success=True)
  else:
    logger.warning(
      "Saturation pressure calculation terminates unsuccessfully. "
      "The solution method was QNSS, EOS: %s. Parameters:"
      "\n\tP0 = %s, T = %s\n\tyi = %s\n\timprove_P0 = %s",
      eos.name, P0, T, yi, improve_P0,
    )
    return PsatResult(P=Pk, lnphiji=np.vstack([lnphixi, lnphiyi]),
                      Zj=np.array([Zx, Zy]), yji=np.vstack([xi, yi]),
                      success=False)

