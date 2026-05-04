from logging import (
  getLogger,
)

from typing import (
  cast,
)

from math import (
  exp,
  isfinite,
  log,
  log10,
)

from numpy import (
  abs as np_abs,
  array as np_array,
  atleast_2d as np_atleast_2d,
  empty as np_empty,
  exp as np_exp,
  eye as np_eye,
  fill_diagonal as np_fill_diagonal,
  linspace as np_linspace,
  log as np_log,
  logspace as np_logspace,
  sqrt as np_sqrt,
  zeros as np_zeros,
  zeros_like as np_zeros_like,
)

from resim.pvt.datatypes import (
  Float,
  MultiPhaseState,
  State,
  Vector,
)

from resim.pvt.constants import (
  R,
)

from resim.pvt.utils import (
  LinearSolver,
  lusolver,
)

from resim.pvt.eos import (
  State2pPTEos,
)

from resim.pvt.stab import (
  StabPTEos,
  StabRoutine,
  stabtest,
)

from resim.pvt.psat.protocols import (
  PsatPTEos,
  PsatSolver,
  PsatSolverPTEos,
)


logger = getLogger('psat')


class PsatConvergenceError(Exception):
  """An exception that will be raised if a solver for saturation
  pressure does not converge.
  """
  def __init__(
    self,
    msg: str = ('The saturation pressure calculation was\nterminated '
                'unsuccessfully. Try to improve the initial guess. '
                'It may\nalso be advisable to change the solver '
                'or increase the number of\niterations.'),
  ) -> None:
    super().__init__(msg)
    pass


def _psatPT_ss(
  eos: PsatSolverPTEos,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  Plow: float,
  Pupp: float,
  upper: bool,
  tol: float = 1e-20,
  maxiter: int = 300,
  tol_tpd: float = 1e-10,
  maxiter_tpd: int = 12,
) -> tuple[float, Vector[Float], Vector[Float]]:
  """Solve the saturation pressure problem formulated for the PT-
  thermodynamics using successive substitution iterations and the
  secant method.

  Parameters
  ----------
  eos: PsatSolverPTEos
    An initialized instance of a PT-based equation of state.

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  Plow: float
    The saturation pressure lower bound [Pa].

  Pupp: float
    The saturation pressure upper bound [Pa].

  upper: bool
    A boolean flag indicating whether the desired value is located at
    the upper saturation curve (`True`) or the lower saturation curve
    (`False`). The cricondentherm serves as the dividing point between
    the upper and lower saturation curves.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `300`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-10`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `12`.

  Returns
  -------
  A tuple containing:
  - saturation pressure [Pa],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.

  Raises
  ------
  PsatConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  The algorithm implemented by this functions uses successive
  substitution iterations to find a non-trivial local minimum
  of the tangent-plane distance (TPD) function. The TPD-equation,
  which is the equation of equality to zero of the TPD-function,
  is solved on pressure in the inner loop by the secant method.
  """
  logger.info('Saturation pressure calculation using the SS-method.')
  Nc = eos.Nc
  logger.info('T = %.2f [K], yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat [Pa]', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%11.2e%11.2e'
  lnyi = np_log(yi)
  if upper:
    P2p = Plow
    P1p = Pupp
  else:
    P2p = Pupp
    P1p = Plow
  k = 0
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  r = 0
  lnkvi = np_log(xi) - lnyi
  Pr = P2p
  lnphiyi = eos.getPT_lnphii(Pr, T, yi)
  lnphixi = eos.getPT_lnphii(Pr, T, xi)
  TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
  repeat = TPDr < -tol_tpd or TPDr > tol_tpd
  if repeat:
    Prm1 = P1p
    lnphiyi = eos.getPT_lnphii(Prm1, T, yi)
    lnphixi = eos.getPT_lnphii(Prm1, T, xi)
    TPDrm1 = xi.dot(lnkvi + lnphixi - lnphiyi)
    if TPDr * TPDrm1 < 0.:
      while repeat and r < maxiter_tpd:
        Prp1 = Pr - TPDr * (Pr - Prm1) / (TPDr - TPDrm1)
        r += 1
        Prm1 = Pr
        TPDrm1 = TPDr
        Pr = Prp1
        lnphiyi = eos.getPT_lnphii(Pr, T, yi)
        lnphixi = eos.getPT_lnphii(Pr, T, xi)
        TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
        repeat = TPDr < -tol_tpd or TPDr > tol_tpd
  gi = lnki + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  repeat = repeat or g2 > tol
  logger.debug(tmpl, k, *lnki, Pr, g2, TPDr)
  while repeat and k < maxiter:
    lnki -= gi
    k += 1
    ki = np_exp(lnki)
    ni = ki * yi
    xi = ni / ni.sum()
    r = 0
    lnkvi = np_log(xi) - lnyi
    lnphiyi = eos.getPT_lnphii(Pr, T, yi)
    lnphixi = eos.getPT_lnphii(Pr, T, xi)
    TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
    repeat = TPDr < -tol_tpd or TPDr > tol_tpd
    if repeat:
      if TPDr < 0.:
        Prm1 = P1p
      else:
        Prm1 = P2p
      lnphiyi = eos.getPT_lnphii(Prm1, T, yi)
      lnphixi = eos.getPT_lnphii(Prm1, T, xi)
      TPDrm1 = xi.dot(lnkvi + lnphixi - lnphiyi)
      if TPDr * TPDrm1 < 0.:
        while repeat and r < maxiter_tpd:
          Prp1 = Pr - TPDr * (Pr - Prm1) / (TPDr - TPDrm1)
          r += 1
          Prm1 = Pr
          TPDrm1 = TPDr
          Pr = Prp1
          lnphiyi = eos.getPT_lnphii(Pr, T, yi)
          lnphixi = eos.getPT_lnphii(Pr, T, xi)
          TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
          repeat = TPDr < -tol_tpd or TPDr > tol_tpd
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    repeat = repeat or g2 > tol
    logger.debug(tmpl, k, *lnki, Pr, g2, TPDr)
  if not repeat and isfinite(g2) and isfinite(Pr):
    return Pr, ki, xi
  logger.warning(
    "The SS-method for saturation pressure calculation does "
    "not converge. EOS: %s.\nParameters:\nT = %s [K]\nyi = %s\n"
    "kvi0 = %s\nPlow = %s [Pa]\nPupp = %s [Pa]",
    eos.name, T, yi.tolist(), kvi0.tolist(), Plow, Pupp,
  )
  raise PsatConvergenceError()


def _psatPT_qnss(
  eos: PsatSolverPTEos,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  Plow: float,
  Pupp: float,
  upper: bool,
  tol: float = 1e-20,
  maxiter: int = 200,
  lmbdmax: float = 30.,
  tol_tpd: float = 1e-10,
  maxiter_tpd: int = 12,
) -> tuple[float, Vector[Float], Vector[Float]]:
  """Solve the saturation pressure problem formulated for the PT-
  thermodynamics using quasi-newton successive substitution iterations
  (QNSS) and the secant method.

  Parameters
  ----------
  eos: PsatSolverPTEos
    An initialized instance of a PT-based equation of state.

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  Plow: float
    The saturation pressure lower bound [Pa].

  Pupp: float
    The saturation pressure upper bound [Pa].

  upper: bool
    A boolean flag indicating whether the desired value is located at
    the upper saturation curve (`True`) or the lower saturation curve
    (`False`). The cricondentherm serves as the dividing point between
    the upper and lower saturation curves.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the vector of equations is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `200`.

  lmbdmax: float
    The maximum step length. Default is `30.0`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-10`.
    The TPD-equation is the equation of equality to zero of the
    tangent-plane distance function.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations.
    Default is `12`.

  Returns
  -------
  A tuple containing:
  - saturation pressure [Pa],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.

  Raises
  ------
  PsatConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  The algorithm implemented by this functions uses QNSS iterations to
  find a non-trivial local minimum of the tangent-plane distance (TPD)
  function. The TPD-equation, which is the equation of equality to zero
  of the TPD-function, is solved on pressure in the inner loop by the
  secant method.

  For the details of the QNSS-method, see the following papers:

  - L.X. Nghiem and Y.-K. Li, 1984 (doi: 10.1016/0378-3812(84)80013-8),
  - L.X. Nghiem [et al], 1985 (doi: 10.1016/0378-3812(85)90059-7).
  """
  logger.info('Saturation pressure calculation using the QNSS-method.')
  Nc = eos.Nc
  logger.info('T = %.2f [K], yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat [Pa]', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%11.2e%11.2e'
  lnyi = np_log(yi)
  if upper:
    P2p = Plow
    P1p = Pupp
  else:
    P2p = Pupp
    P1p = Plow
  k = 0
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  r = 0
  lnkvi = np_log(xi) - lnyi
  Pr = P2p
  lnphiyi = eos.getPT_lnphii(Pr, T, yi)
  lnphixi = eos.getPT_lnphii(Pr, T, xi)
  TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
  repeat = TPDr < -tol_tpd or TPDr > tol_tpd
  if repeat:
    Prm1 = P1p
    lnphiyi = eos.getPT_lnphii(Prm1, T, yi)
    lnphixi = eos.getPT_lnphii(Prm1, T, xi)
    TPDrm1 = xi.dot(lnkvi + lnphixi - lnphiyi)
    if TPDr * TPDrm1 < 0.:
      while repeat and r < maxiter_tpd:
        Prp1 = Pr - TPDr * (Pr - Prm1) / (TPDr - TPDrm1)
        r += 1
        Prm1 = Pr
        TPDrm1 = TPDr
        Pr = Prp1
        lnphiyi = eos.getPT_lnphii(Pr, T, yi)
        lnphixi = eos.getPT_lnphii(Pr, T, xi)
        TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
        repeat = TPDr < -tol_tpd or TPDr > tol_tpd
  gi = lnki + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  lmbd = 1.
  dlnki = -gi
  repeat = repeat or g2 > tol
  logger.debug(tmpl, k, *lnki, Pr, g2, TPDr)
  while repeat and k < maxiter:
    dlnki = -lmbd * gi
    k += 1
    tkm1 = dlnki.dot(gi)
    lnki += dlnki
    ki = np_exp(lnki)
    ni = ki * yi
    xi = ni / ni.sum()
    r = 0
    lnkvi = np_log(xi) - lnyi
    lnphiyi = eos.getPT_lnphii(Pr, T, yi)
    lnphixi = eos.getPT_lnphii(Pr, T, xi)
    TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
    repeat = TPDr < -tol_tpd or TPDr > tol_tpd
    if repeat:
      if TPDr < 0.:
        Prm1 = P1p
      else:
        Prm1 = P2p
      lnphiyi = eos.getPT_lnphii(Prm1, T, yi)
      lnphixi = eos.getPT_lnphii(Prm1, T, xi)
      TPDrm1 = xi.dot(lnkvi + lnphixi - lnphiyi)
      if TPDr * TPDrm1 < 0.:
        while repeat and r < maxiter_tpd:
          Prp1 = Pr - TPDr * (Pr - Prm1) / (TPDr - TPDrm1)
          r += 1
          Prm1 = Pr
          TPDrm1 = TPDr
          Pr = Prp1
          lnphiyi = eos.getPT_lnphii(Pr, T, yi)
          lnphixi = eos.getPT_lnphii(Pr, T, xi)
          TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
          repeat = TPDr < -tol_tpd or TPDr > tol_tpd
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnki, Pr, g2, TPDr)
    repeat = repeat or g2 > tol
    if k % Nc == 0:
      lmbd = 1.
      dlnki = -gi
    else:
      lmbd *= tkm1 / (dlnki.dot(gi) - tkm1)
      if lmbd < 0.:
        lmbd = -lmbd
      if lmbd > lmbdmax:
        lmbd = lmbdmax
      max_dlnki = np_abs(dlnki).max()
      if max_dlnki > 6.:
        relax = 6. / max_dlnki
        lmbd *= relax
        dlnki *= relax
  if not repeat and isfinite(g2) and isfinite(Pr):
    return Pr, ki, xi
  logger.warning(
    "The QNSS-method for saturation pressure calculation does "
    "not converge. EOS: %s.\nParameters:\nT = %s [K]\nyi = %s\n"
    "kvi0 = %s\nPlow = %s [Pa]\nPupp = %s [Pa]",
    eos.name, T, yi.tolist(), kvi0.tolist(), Plow, Pupp,
  )
  raise PsatConvergenceError()


def _psatPT_newtA(
  eos: PsatSolverPTEos,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  Plow: float,
  Pupp: float,
  upper: bool,
  tol: float = 1e-20,
  maxiter: int = 100,
  tol_tpd: float = 1e-10,
  maxiter_tpd: int = 12,
  linsolver: LinearSolver = lusolver,
) -> tuple[float, Vector[Float], Vector[Float]]:
  r"""Solve the saturation pressure problem formulated for the PT-
  thermodynamics using Newton's method.

  Parameters
  ----------
  eos: PsatSolverPTEos
    An initialized instance of a PT-based equation of state.

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    Initial guess of k-values of `Nc` components.

  Plow: float
    The saturation pressure lower bound [Pa].

  Pupp: float
    The saturation pressure upper bound [Pa].

  upper: bool
    A boolean flag indicating whether the desired value is located at
    the upper saturation curve (`True`) or the lower saturation curve
    (`False`). The cricondentherm serves as the dividing point between
    the upper and lower saturation curves. This parameter is only used
    by the algorithm that improves the initial guess of k-values and
    saturation pressure.

  tol: float
    Terminate the solver successfully if the sum of squared elements of
    the vector of equations is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm that improves the initial
    guess of k-values and saturation pressure. Default is `1e-10`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm that improves the initial guess of k-values
    and saturation pressure. Default is `12`.

  linsolver: LinearSolver
    A callable object that accepts an `A: Matrix[Float]` of shape
    `(Nc + 1, Nc + 1)` and `b: Vector[Float]` of shape `(Nc + 1,)` and
    finds `x: Vector[Float]` of shape `(Nc + 1,)`, which is the solution
    to the linear system :math:`\mathbf{A}^\top \mathbf{x}= \mathbf{b}`.
    The matrix `A` is a nonsymmetric matrix. Default is `lusolver`.

  Returns
  -------
  A tuple containing:
  - saturation pressure [Pa],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.

  Raises
  ------
  PsatConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  The system of non-linear equations to be solved consists of:

  - `Nc` equations represented by the equality to zero of the gradient
    of the tangent-plane distance function,
  - the constraint on mole numbers of components in the trial phase.

  The formulation of the system of nonlinear equations is based on the
  following paper:

  - M.L. Michelsen, 1980 (doi: 10.1016/0378-3812(80)80001-X).
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (A-form)."
  )
  Nc = eos.Nc
  logger.info('T = %.2f [K], yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat [Pa]', 'g2',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%11.2e'
  J = np_zeros(shape=(Nc + 1, Nc + 1))
  gi = np_empty(shape=(Nc + 1,))
  Iij = np_eye(Nc)
  k = 0
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  r = 0
  lnkvi = np_log(xi / yi)
  if upper:
    Prm1 = Pupp
    Pr = Plow
  else:
    Prm1 = Plow
    Pr = Pupp
  lnphiyi = eos.getPT_lnphii(Prm1, T, yi)
  lnphixi = eos.getPT_lnphii(Prm1, T, xi)
  TPDrm1 = xi.dot(lnkvi + lnphixi - lnphiyi)
  lnphiyi = eos.getPT_lnphii(Pr, T, yi)
  lnphixi = eos.getPT_lnphii(Pr, T, xi)
  TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
  if TPDr * TPDrm1 < 0.:
    while (TPDr < -tol_tpd or TPDr > tol_tpd) and r < maxiter_tpd:
      Prp1 = Pr - TPDr * (Pr - Prm1) / (TPDr - TPDrm1)
      r += 1
      Prm1 = Pr
      TPDrm1 = TPDr
      Pr = Prp1
      lnphiyi = eos.getPT_lnphii(Pr, T, yi)
      lnphixi = eos.getPT_lnphii(Pr, T, xi)
      TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
  Pk = Pr
  lnphixi, dlnphixidP, dlnphixidnj = eos.getPT_lnphii_dP_dnj(Pk, T, xi, n)
  lnphiyi, dlnphiyidP = eos.getPT_lnphii_dP(Pk, T, yi)
  gi[:Nc] = lnki + lnphixi - lnphiyi
  gi[-1] = n - 1.
  g2 = gi.dot(gi)
  logger.debug(tmpl, k, *lnki, Pk, g2)
  while g2 > tol and k < maxiter:
    J[:Nc, :Nc] = Iij + ni * dlnphixidnj
    J[-1, :Nc] = ni
    J[:Nc, -1] = Pk * (dlnphixidP - dlnphiyidP)
    dlnkilnP = linsolver(J, -gi)
    k += 1
    lnki += dlnkilnP[:-1]
    Pkp1 = Pk * exp(dlnkilnP[-1])
    if Pkp1 > Pupp:
      Pk = .5 * (Pk + Pupp)
    elif Pkp1 < Plow:
      Pk = .5 * (Plow + Pk)
    else:
      Pk = Pkp1
    ki = np_exp(lnki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, dlnphixidP, dlnphixidnj = eos.getPT_lnphii_dP_dnj(Pk, T, xi, n)
    lnphiyi, dlnphiyidP = eos.getPT_lnphii_dP(Pk, T, yi)
    gi[:Nc] = lnki + lnphixi - lnphiyi
    gi[-1] = n - 1.
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnki, Pk, g2)
  if g2 < tol and isfinite(g2) and isfinite(Pk):
    return Pk, ki, xi
  logger.warning(
    "Newton's method (A-form) for saturation pressure calculation "
    "does not converge. EOS: %s.\nParameters:\nT = %s [K]\nyi = %s\n"
    "kvi0 = %s\nPlow = %s [Pa]\nPupp = %s [Pa]",
    eos.name, T, yi.tolist(), kvi0.tolist(), Plow, Pupp,
  )
  raise PsatConvergenceError()


def _psatPT_newtB(
  eos: PsatSolverPTEos,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  Plow: float,
  Pupp: float,
  upper: bool,
  tol: float = 1e-20,
  maxiter: int = 100,
  tol_tpd: float = 1e-10,
  maxiter_tpd: int = 12,
  linsolver: LinearSolver = lusolver,
) -> tuple[float, Vector[Float], Vector[Float]]:
  r"""Solve the saturation pressure problem formulated for the PT-
  thermodynamics using Newton's method.

  Parameters
  ----------
  eos: PsatSolverPTEos
    An initialized instance of a PT-based equation of state.

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    Initial guess of k-values of `Nc` components.

  Plow: float
    The saturation pressure lower bound [Pa].

  Pupp: float
    The saturation pressure upper bound [Pa].

  upper: bool
    A boolean flag indicating whether the desired value is located at
    the upper saturation curve (`True`) or the lower saturation curve
    (`False`). The cricondentherm serves as the dividing point between
    the upper and lower saturation curves. This parameter is also used
    by the algorithm that improves the initial guess of k-values and
    saturation pressure.

  tol: float
    Terminate the solver successfully if the sum of squared elements of
    the vector of equations is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. The TPD-equation is
    the equation of equality to zero of the tangent-plane distance,
    which determines the second phase appearance or disappearance.
    This parameter is used by the algorithm that improves the initial
    guess of k-values and saturation pressure. Default is `1e-10`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. This parameter
    is used by the algorithm that improves the initial guess of k-values
    and saturation pressure. Default is `12`.

  linsolver: LinearSolver
    A callable object that accepts an `A: Matrix[Float]` of shape
    `(Nc + 1, Nc + 1)` and `b: Vector[Float]` of shape `(Nc + 1,)` and
    finds `x: Vector[Float]` of shape `(Nc + 1,)`, which is the solution
    to the linear system :math:`\mathbf{A}^\top \mathbf{x}= \mathbf{b}`.
    The matrix `A` is a nonsymmetric matrix. Default is `lusolver`.

  Returns
  -------
  A tuple containing:
  - saturation pressure [Pa],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.

  Raises
  ------
  PsatConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  The system of non-linear equations to be solved consists of:

  - `Nc` equations represented by the equality to zero of the gradient
    of the tangent-plane distance function,
  - the equality of the TPD-function to zero.

  The formulation of the system of nonlinear equations is based on the
  following paper:

  - L.X. Nghiem [et al], 1985 (doi: 10.1016/0378-3812(85)90059-7).
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (B-form)."
  )
  Nc = eos.Nc
  logger.info('T = %.2f [K], yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)], 'Psat [Pa]', 'g2',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%11.2e'
  J = np_empty(shape=(Nc + 1, Nc + 1))
  gi = np_empty(shape=(Nc + 1,))
  Iij = np_eye(Nc)
  k = 0
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  r = 0
  lnkvi = np_log(xi / yi)
  if upper:
    Prm1 = Pupp
    Pr = Plow
  else:
    Prm1 = Plow
    Pr = Pupp
  lnphiyi = eos.getPT_lnphii(Prm1, T, yi)
  lnphixi = eos.getPT_lnphii(Prm1, T, xi)
  TPDrm1 = xi.dot(lnkvi + lnphixi - lnphiyi)
  lnphiyi = eos.getPT_lnphii(Pr, T, yi)
  lnphixi = eos.getPT_lnphii(Pr, T, xi)
  TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
  if TPDr * TPDrm1 < 0.:
    while (TPDr < -tol_tpd or TPDr > tol_tpd) and r < maxiter_tpd:
      Prp1 = Pr - TPDr * (Pr - Prm1) / (TPDr - TPDrm1)
      r += 1
      Prm1 = Pr
      TPDrm1 = TPDr
      Pr = Prp1
      lnphiyi = eos.getPT_lnphii(Pr, T, yi)
      lnphixi = eos.getPT_lnphii(Pr, T, xi)
      TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
  Pk = Pr
  lnphixi, dlnphixidP, dlnphixidnj = eos.getPT_lnphii_dP_dnj(Pk, T, xi, n)
  lnphiyi, dlnphiyidP = eos.getPT_lnphii_dP(Pk, T, yi)
  gi[:Nc] = lnki + lnphixi - lnphiyi
  hi = gi[:Nc] - log(n)
  gi[-1] = xi.dot(hi)
  g2 = gi.dot(gi)
  logger.debug(tmpl, k, *lnki, Pk, g2)
  while g2 > tol and k < maxiter:
    J[:Nc, :Nc] = Iij + ni * dlnphixidnj
    J[-1, :Nc] = xi * (hi - gi[-1])
    J[:Nc, -1] = Pk * (dlnphixidP - dlnphiyidP)
    J[-1, -1] = xi.dot(J[:Nc,-1])
    dlnkilnP = linsolver(J, -gi)
    k += 1
    lnki += dlnkilnP[:-1]
    Pkp1 = Pk * exp(dlnkilnP[-1])
    if Pkp1 > Pupp:
      Pk = .5 * (Pk + Pupp)
    elif Pkp1 < Plow:
      Pk = .5 * (Plow + Pk)
    else:
      Pk = Pkp1
    ki = np_exp(lnki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    lnphixi, dlnphixidP, dlnphixidnj = eos.getPT_lnphii_dP_dnj(Pk, T, xi, n)
    lnphiyi, dlnphiyidP = eos.getPT_lnphii_dP(Pk, T, yi)
    gi[:Nc] = lnki + lnphixi - lnphiyi
    hi = gi[:Nc] - log(n)
    gi[-1] = xi.dot(hi)
    g2 = gi.dot(gi)
    logger.debug(tmpl, k, *lnki, Pk, g2)
  if g2 < tol and isfinite(g2) and isfinite(Pk):
    return Pk, ki, xi
  logger.warning(
    "Newton's method (B-form) for saturation pressure calculation "
    "does not converge. EOS: %s.\nParameters:\nT = %s [K]\nyi = %s\n"
    "kvi0 = %s\nPlow = %s [Pa]\nPupp = %s [Pa]",
    eos.name, T, yi.tolist(), kvi0.tolist(), Plow, Pupp,
  )
  raise PsatConvergenceError()


def _psatPT_newtC(
  eos: PsatSolverPTEos,
  T: float,
  yi: Vector[Float],
  kvi0: Vector[Float],
  Plow: float,
  Pupp: float,
  upper: bool,
  tol: float = 1e-20,
  maxiter: int = 100,
  tol_tpd: float = 1e-10,
  maxiter_tpd: int = 12,
  linsolver: LinearSolver = lusolver,
) -> tuple[float, Vector[Float], Vector[Float]]:
  r"""Solve the saturation pressure problem formulated for the PT-
  thermodynamics using Newton's method and the secant method.

  Parameters
  ----------
  eos: PsatSolverPTEos
    An initialized instance of a PT-based equation of state.

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    Initial guess of k-values of `Nc` components.

  Plow: float
    The saturation pressure lower bound [Pa].

  Pupp: float
    The saturation pressure upper bound [Pa].

  upper: bool
    A boolean flag indicating whether the desired value is located at
    the upper saturation curve (`True`) or the lower saturation curve
    (`False`). The cricondentherm serves as the dividing point between
    the upper and lower saturation curves.

  tol: float
    Terminate the solver successfully if the sum of squared elements of
    the gradient is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  tol_tpd: float
    Terminate the TPD-equation solver successfully if the absolute
    value of the equation is less than `tol_tpd`. Default is `1e-10`.

  maxiter_tpd: int
    The maximum number of TPD-equation solver iterations. Default is
    `12`.

  linsolver: LinearSolver
    A callable object that takes an `A: Matrix[Float]` of shape
    `(Nc, Nc)` and a `b: Vector[Float]` of shape `(Nc,)` and finds
    `x: Vector[Float]` of shape `(Nc,)`, which is the solution to the
    linear system :math:`\mathbf{A}^\top \mathbf{x} = \mathbf{b}`. The
    matrix `A` is a symmetric matrix. Default is `lusolver`.

  Returns
  -------
  A tuple containing:
  - saturation pressure [Pa],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.

  Raises
  ------
  PsatConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  The system of non-linear equations to be solved consists of `Nc`
  equations represented by the equality to zero of the gradient
  of the Michelsen's modified tangent-plane distance (TPD) function.
  The TPD-equation, which is the equation of equality to zero of the
  TPD-function, is solved on pressure in the inner loop by the secant
  method.
  """
  logger.info(
    "Saturation pressure calculation using Newton's method (C-form)."
  )
  Nc = eos.Nc
  logger.info('T = %.2f [K], yi =' + Nc * '%7.4f', T, *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%12s%11s%11s',
    'Nit', *['alph%s' % s for s in range(Nc)], 'Psat [Pa]', 'g2', 'TPD',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%12.1f%11.2e%11.2e'
  lnyi = np_log(yi)
  if upper:
    P2p = Plow
    P1p = Pupp
  else:
    P2p = Pupp
    P1p = Plow
  k = 0
  ki = kvi0
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  r = 0
  lnkvi = np_log(xi) - lnyi
  Pr = P2p
  lnphiyi = eos.getPT_lnphii(Pr, T, yi)
  lnphixi = eos.getPT_lnphii(Pr, T, xi)
  TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
  repeat = TPDr < -tol_tpd or TPDr > tol_tpd
  if repeat:
    Prm1 = P1p
    lnphiyi = eos.getPT_lnphii(Prm1, T, yi)
    lnphixi = eos.getPT_lnphii(Prm1, T, xi)
    TPDrm1 = xi.dot(lnkvi + lnphixi - lnphiyi)
    if TPDr * TPDrm1 < 0.:
      while repeat and r < maxiter_tpd:
        Prp1 = Pr - TPDr * (Pr - Prm1) / (TPDr - TPDrm1)
        r += 1
        Prm1 = Pr
        TPDrm1 = TPDr
        Pr = Prp1
        lnphiyi = eos.getPT_lnphii(Pr, T, yi)
        lnphixi = eos.getPT_lnphii(Pr, T, xi)
        TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
        repeat = TPDr < -tol_tpd or TPDr > tol_tpd
  gi = np_log(ki) + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  sqrtni = np_sqrt(ni)
  alphai = 2. * sqrtni
  repeat = repeat or g2 > tol
  logger.debug(tmpl, k, *alphai, Pr, g2, TPDr)
  while repeat and k < maxiter:
    H = sqrtni[:,None] * sqrtni * eos.getPT_lnphii_dnj(Pr, T, xi, n)[1]
    np_fill_diagonal(H, H.diagonal() + .5 * gi + 1.)
    # TODO: Replace the LU-solver with the custom implementation
    #       of the modified Cholesky decomposition solver.
    dalphai = linsolver(H, -sqrtni * gi)
    k += 1
    alphai += dalphai
    sqrtni = alphai * .5
    ni = sqrtni * sqrtni
    n = ni.sum()
    xi = ni / n
    r = 0
    lnkvi = np_log(xi) - lnyi
    lnphiyi = eos.getPT_lnphii(Pr, T, yi)
    lnphixi = eos.getPT_lnphii(Pr, T, xi)
    TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
    repeat = TPDr < -tol_tpd or TPDr > tol_tpd
    if repeat:
      if TPDr < 0.:
        Prm1 = P1p
      else:
        Prm1 = P2p
      lnphiyi = eos.getPT_lnphii(Prm1, T, yi)
      lnphixi = eos.getPT_lnphii(Prm1, T, xi)
      TPDrm1 = xi.dot(lnkvi + lnphixi - lnphiyi)
      if TPDr * TPDrm1 < 0.:
        while repeat and r < maxiter_tpd:
          Prp1 = Pr - TPDr * (Pr - Prm1) / (TPDr - TPDrm1)
          r += 1
          Prm1 = Pr
          TPDrm1 = TPDr
          Pr = Prp1
          lnphiyi = eos.getPT_lnphii(Pr, T, yi)
          lnphixi = eos.getPT_lnphii(Pr, T, xi)
          TPDr = xi.dot(lnkvi + lnphixi - lnphiyi)
          repeat = TPDr < -tol_tpd or TPDr > tol_tpd
    gi = np_log(ni) + lnphixi - lnphiyi - lnyi
    g2 = gi.dot(gi)
    repeat = repeat or g2 > tol
    logger.debug(tmpl, k, *alphai, Pr, g2, TPDr)
  if not repeat and isfinite(g2) and isfinite(Pr):
    return Pr, xi / yi, xi
  logger.warning(
    "Newton's method (C-form) for saturation pressure calculation "
    "does not converge. EOS: %s.\nParameters:\nT = %s [K]\nyi = %s\n"
    "kvi0 = %s\nPlow = %s [Pa]\nPupp = %s [Pa]",
    eos.name, T, yi.tolist(), kvi0.tolist(), Plow, Pupp,
  )
  raise PsatConvergenceError()


class psat(object):
  def __init__(
    self,
    state: State | None = None,
  ) -> None:
    """Specify settings for the saturation pressure calculation.

    Parameters
    ----------
    state: State | None
      A thermodynamic state of a mixture, k-values an pressure of which
      can be used as an initial guess for the saturation pressure
      calculation procedure. If it is `None`, the option to use
      previously calculated results is not available until such a state
      is passed directly to the callable instance of this class. Default
      is `None`.

    Notes
    -----
    Above settings influence the calculation only if the corresponding
    parameters of the `__call__` method of this class are set to `None`.
    """
    self.state = state
    pass

  def __call__(
    self,
    eos: PsatPTEos,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    upper: bool = True,
    init: tuple[Vector[Float], float, float] | State | None = None,
    solver: PsatSolver | str = 'default',
    **kwargs,
  ) -> MultiPhaseState:
    """Find a two-phase saturation state for a given temperature and
    mole composition of a mixture.

    Parameters
    ----------
    eos: PsatPTEos
      An initialized instance of an equation of state.

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture [mol]. Default is `1.0` [mol].

    upper: bool
      A boolean flag indicating whether the desired value is located
      at the upper saturation curve (`True`) or the lower saturation
      curve (`False`). The cricondentherm serves as the dividing point
      between the upper and lower saturation curves. Default is `True`.

    init: tuple[Vector[Float], float, float] | State | None
      This parameter is used to initialize the saturation pressure
      calculation procedure. The detailed explanation of the logic
      behind different types of `init` is given in the following table:

      +------------------+---------------------------------------------+
      | Type             | Description                                 |
      +==================+=============================================+
      | tuple[           | A tuple, containing:                        |
      |   Vector[Float], | - initial guess of k-values as a vector of  |
      |   float,         |   shape `(Nc,)`,                            |
      |   float,         | - a lower bound of pressure [Pa],           |
      | ]                | - an upper bound of pressure [Pa].          |
      +------------------+---------------------------------------------+
      | State            | A state of a mixture, k-values and pressure |
      |                  | of which can be used to initialize the      |
      |                  | procedure. The one-dimensional search along |
      |                  | the pressure axis is used to find a change  |
      |                  | in stability of a one-phase state starting  |
      |                  | from a given initial state.                 |
      +------------------+---------------------------------------------+
      | None             | Perform several stability tests along the   |
      |                  | pressure axis to find a change in stability |
      |                  | of a one-phase state of a mixture starting  |
      |                  | from the upper or lower boundary of the     |
      |                  | possible saturation pressure range.         |
      +------------------+---------------------------------------------+

      Default is `None`.

    solver: PsatSolver | str
      A callable object that can solve the saturation pressure problem.
      It also can be a string defining the name of an internal solver.

      For the PT-thermodynamics, the following internal solvers are
      available:

      +-----------------+----------------------------------------------+
      | Internal solver | Description                                  |
      +=================+==============================================+
      | `'ss'`          | Uses successive substitution iterations to   |
      |                 | solve the system of equilibrium equations    |
      |                 | and the secant method for the TPD-equation.  |
      +-----------------+----------------------------------------------+
      | `'qnss'`        | Uses quasi-newton successive substitution    |
      |                 | iterations (QNSS) to solve the system of     |
      |                 | equilibrium equations and the secant method  |
      |                 | for the TPD-equation.                        |
      +-----------------+----------------------------------------------+
      | `'newton'`      | Uses Newton's method to find a local minimum |
      |                 | of the modified Michelsen's TPD-function and |
      |                 | the secant method for the TPD-equation.      |
      +-----------------+----------------------------------------------+
      | `'newton-a'`    | Uses Newton's method to solve a system of    |
      |                 | nonlinear equations consisting of the mole   |
      |                 | number constraint and equilibrium equations. |
      +-----------------+----------------------------------------------+
      | `'newton-b'`    | Uses Newton's method to solve a system of    |
      |                 | nonlinear equations consisting of the TPD-   |
      |                 | equation and equilibrium equations.          |
      +-----------------+----------------------------------------------+

      The following table lists default internal solvers for each
      formulation of the saturation pressure problem:

      +-----------------+----------------------------------------------+
      | Basic variables | Default internal solvers (`'default'`)       |
      +=================+==============================================+
      | P, T            | Newton (`'newton'`).                         |
      +-----------------+----------------------------------------------+

      Default is `'default'`.

    **kwargs
      Other parameters for an internal initialization procedure.

    Notes
    -----
    Selection between different formulations of the saturation pressure
    calculation is based on the `form` attribute of the `eos`.
    """
    if init is None:
      init = self.state
    if eos.form == 'PT':
      eos = cast(PsatPTEos, eos)
      solverPT: PsatSolver[PsatSolverPTEos]
      if callable(solver):
        solverPT = cast(PsatSolver[PsatSolverPTEos], solver)
      elif solver == 'default' or solver == 'newton':
        solverPT = _psatPT_newtC
      elif solver == 'ss':
        solverPT = _psatPT_ss
      elif solver == 'qnss':
        solverPT = _psatPT_qnss
      elif solver == 'newton-a':
        solverPT = _psatPT_newtA
      elif solver == 'newton-b':
        solverPT = _psatPT_newtB
      else:
        raise ValueError(
          'Unknown saturation pressure solver for a PT-based EOS: '
          f'"{solver}".'
        )
      return self.runPT(eos, T, yi, n, upper, init, solverPT, **kwargs)
    else:
      raise NotImplementedError(
        f'The {eos.form}-formulation of the saturation pressure calculation '
        'routine is not implemented yet.'
      )

  @classmethod
  def runPT(
    cls,
    eos: PsatPTEos,
    T: float,
    yi: Vector[Float],
    n: float = 1.,
    upper: bool = True,
    init: tuple[Vector[Float], float, float] | State | None = None,
    solver: PsatSolver[PsatSolverPTEos] = _psatPT_newtC,
    **kwargs,
  ) -> MultiPhaseState:
    """Find a two-phase saturation state for a given temperature and
    mole composition of a mixture using a PT-based equation of state.

    Parameters
    ----------
    eos: PsatPTEos
      An initialized instance of a PT-based equation of state.

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture [mol]. Default is `1.0` [mol].

    init: tuple[Vector[Float], float, float] | State | None
      This parameter is used to initialize the saturation pressure
      calculation procedure. The detailed explanation of the logic
      behind different types of `init` is given in the following table:

      +------------------+---------------------------------------------+
      | Type             | Description                                 |
      +==================+=============================================+
      | tuple[           | A tuple, containing:                        |
      |   Vector[Float], | - initial guess of k-values as a vector of  |
      |   float,         |   shape `(Nc,)`,                            |
      |   float,         | - a lower bound of pressure [Pa],           |
      | ]                | - an upper bound of pressure [Pa].          |
      +------------------+---------------------------------------------+
      | State            | A state of a mixture, k-values and pressure |
      |                  | of which can be used to initialize the      |
      |                  | procedure. The one-dimensional search along |
      |                  | the pressure axis is used to find a change  |
      |                  | in stability of a one-phase state starting  |
      |                  | from a given initial state.                 |
      +------------------+---------------------------------------------+
      | None             | Perform several stability tests along the   |
      |                  | pressure axis to find a change in stability |
      |                  | of a one-phase state of a mixture starting  |
      |                  | from the upper or lower boundary of the     |
      |                  | possible saturation pressure range.         |
      +------------------+---------------------------------------------+

      Default is `None`.

    upper: bool
      A boolean flag indicating whether the desired value is located
      at the upper saturation curve (`True`) or the lower saturation
      curve (`False`). The cricondentherm serves as the dividing point
      between the upper and lower saturation curves. Default is `True`.

    solver: PsatSolver[PsatSolverPTEos]
      A callable object that can solve the saturation pressure problem
      formulated for the PT-thermodynamics. Default is `_psatPT_newtC`.

    **kwargs
      Other parameters for an internal initialization procedure.

    Returns
    -------
    A saturation state of a mixture.

    Raises
    ------
    PsatConvergenceError
      This exception is raised if a saturation pressure solver does
      not converge for all initial guesses of k-values.
    """
    if init is None:
      kvi0, Plow, Pupp = cls.gridding(eos, T, yi, upper, **kwargs)
    elif isinstance(init, State):
      kvi0, Plow, Pupp = cls.search(eos, T, yi, upper, init, **kwargs)
    else:
      kvi0, Plow, Pupp = init
    P, kvi, xi = solver(eos, T, yi, kvi0, Plow, Pupp, upper)
    return cls.outputPT(eos, P, T, yi, n, kvi, xi)

  @staticmethod
  def gridding(
    eos: StabPTEos,
    T: float,
    yi: Vector[Float],
    upper: bool,
    stabroutine: StabRoutine[StabPTEos] = stabtest.runPT,
    Pmin: float = 1.,
    Pmax: float = 1e8,
    Nnodes: int = 20,
    logspace: bool = False,
  ) -> tuple[Vector[Float], float, float]:
    """Perform the gridding procedure to obtain the initial guess
    of k-values and the confidence interval for saturation pressure
    calculation. Instead of evaluating all segments, the gridding
    procedure would terminate as soon as an interval exhibiting a
    change in stability of a one-phase state is identified.

    Parameters
    ----------
    eos: StabPTEos
      An initialized instance of a PT-based equation of state.

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    upper: bool
      A boolean flag indicating whether the desired value is located
      at the upper saturation curve (`True`) or the lower saturation
      curve (`False`). The cricondentherm serves as the dividing point
      between the upper and lower saturation curves. This flag also
      controls the start point: if it is set to `True`, then the start
      point would be `Pmax`; otherwise it would be `Pmin`.

    stabroutine: StabRoutine[StabPTEos]
      A callable object that can be used to perform the one-phase
      stability test. Default is `stabtest.runPT`.

    Pmin: float
      The minimum pressure for the gridding procedure. Default is `1.0`
      [Pa].

    Pmax: float
      The maximum pressure for the gridding procedure. Default is `1e8`
      [Pa].

    Nnodes: int
      The number of points to construct a grid. Default is `20`.

    logspace: bool
      A boolean flag indicating whether the log10 grid should be used
      instead of linear. It is advisable to use this option only if
      saturation pressure is expected to be very small. Default is
      `False`.

    Returns
    -------
    A tuple containing:
    - the initial guess for k-values of `Nc` components as
      a `Vector[Float]` of shape `(Nc,)`,
    - the lower bound of saturation pressure [Pa],
    - the upper bound of saturation pressure [Pa].

    Raises
    ------
    ValueError
      This exception will be raised if a change in stability of a one-
      phase state of a mixture is not found.
    """
    if upper:
      if logspace:
        PP = np_logspace(log10(Pmax), log10(Pmin), Nnodes, endpoint=True)
      else:
        PP = np_linspace(Pmax, Pmin, Nnodes, endpoint=True)
    else:
      if logspace:
        PP = np_logspace(log10(Pmin), log10(Pmax), Nnodes, endpoint=True)
      else:
        PP = np_linspace(Pmin, Pmax, Nnodes, endpoint=True)
    prevP = PP[0]
    prevstate = stabroutine(eos, prevP, T, yi, 1., None)
    for nextP in PP[1:]:
      state = stabroutine(eos, nextP, T, yi, 1., None)
      if state.kvji is not None and prevstate.kvji is None:
        if upper:
          return state.kvji.ravel(), nextP, prevP
        else:
          return state.kvji.ravel(), prevP, nextP
      prevstate = state
      prevP = nextP
    raise ValueError(
      'A boundary of the two-phase region was not found. It could be '
      'because of its narrowness or absence in the given range of '
      'pressures. If you are confident in it, try to change the number '
      'of points for gridding or stability test settings.'
    )

  @staticmethod
  def search(
    eos: StabPTEos,
    T: float,
    yi: Vector[Float],
    upper: bool,
    state: State,
    stabroutine: StabRoutine[StabPTEos] = stabtest.runPT,
    Pmin: float = 1.,
    Pmax: float = 1e8,
    step: float = 0.1,
  ) -> tuple[Vector[Float], float, float]:
    """Perform the preliminary search to obtain the initial guess
    of k-values and the confidence interval for saturation pressure
    calculation.

    Parameters
    ----------
    eos: StabPTEos
      An initialized instance of a PT-based equation of state.

    T: float
      Temperature [K].

    yi: Vector[Double], shape (Nc,)
      Mole fractions of `Nc` components.

    upper: bool
      A boolean flag indicating whether the desired value is located
      at the upper saturation curve (`True`) or the lower saturation
      curve (`False`). The cricondentherm serves as the dividing point
      between the upper and lower saturation curves. This flag also
      controls the direction of the search. Default is `True`.

    state: State
      A state of a mixture, k-values and pressure of which can be used
      as a starting point for the preliminary search.

    stabroutine: StabRoutine[StabPTEos]
      A callable object that can be used to perform the one-phase
      stability test. Default is `stabtest.runPT`.

    Pmin: float
      During the preliminary search, the lower bound of the confidence
      interval can not drop below `Pmin`. Otherwise, the `ValueError`
      will be rised. Default is `1.` [Pa].

    Pmax: float
      During the preliminary search, the upper bound of the confidence
      interval can not exceed `Pmax`. Otherwise, the `ValueError` will
      be rised. Default is `1e8` [Pa].

    step: float
      To specify the confidence interval for the saturation pressure
      calculation, the preliminary search is performed. This parameter
      regulates the step of this search in fraction units. For example,
      if it is necessary to find the upper bound of the confidence
      interval, then the next value of pressure will be calculated from
      the previous one using the formula: `Pnext = Pprev * (1. + step)`.
      Default is `0.1`.

    Returns
    -------
    A tuple containing:
    - the initial guess for k-values of `Nc` components as
      a `Vector[Float]` of shape `(Nc,)`,
    - the lower bound of saturation pressure [Pa],
    - the upper bound of saturation pressure [Pa].

    Raises
    ------
    ValueError
      This exception will be raised if a change in stability of a one-
      phase state of a mixture is not found.
    """
    P = state.P
    state = stabroutine(eos, P, T, yi, 1., state)
    # TODO: Implement an extrapolation procedure to improve the initial
    #       guess of k-values and pressure. For the details, see the
    #       paper L.X. Nghiem and Y.K. Li, 1990 (doi: 10.2118/13517-PA).
    if isinstance(state, MultiPhaseState):
      if upper:
        logger.debug('Finding one-phase region for the upper-bound curve.')
        Plow = P
        statemp = state
        c = 1. + step
        Pupp = P
        while Pupp < Pmax:
          Pupp *= c
          state = stabroutine(eos, Pupp, T, yi, 1., None)
          if isinstance(state, MultiPhaseState):
            Plow = Pupp
            statemp = state
          else:
            return statemp.kvji.ravel(), Plow, Pupp
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for pressure and/or `Pmax` parameter.'
        )
      else:
        logger.debug('Finding one-phase region for the lower-bound curve.')
        Pupp = P
        statemp = state
        c = 1. - step
        Plow = P
        while Plow > Pmin:
          Plow *= c
          state = stabroutine(eos, Plow, T, yi, 1., None)
          if isinstance(state, MultiPhaseState):
            statemp = state
            Pupp = Plow
          else:
            return statemp.kvji.ravel(), Plow, Pupp
        raise ValueError(
          'The one-phase region was not identified. Try to change the\n'
          'initial guess for pressure and/or `Pmin` parameter.'
        )
    else:
      if upper:
        logger.debug('Finding two-phase region for the upper-bound curve.')
        Pupp = P
        c = 1. - step
        Plow = P
        while Plow > Pmin:
          Plow *= c
          state = stabroutine(eos, Plow, T, yi, 1., None)
          if state.kvji is not None:
            return state.kvji.ravel(), Plow, Pupp
          else:
            Pupp = Plow
        raise ValueError(
          'The two-phase region was not identified. It could be because\n'
          'of its narrowness or absence. Try to change the initial guess\n'
          'for pressure or stability test parameters. It also might be\n'
          'helpful to reduce the value of the `step`.'
        )
      else:
        logger.debug('Finding two-phase region for the lower-bound curve.')
        Plow = P
        c = 1. + step
        Pupp = P
        while Pupp < Pmax:
          Pupp *= c
          state = stabroutine(eos, Pupp, T, yi, 1., None)
          if state.kvji is not None:
            return state.kvji.ravel(), Plow, Pupp
          else:
            Plow = Pupp
        raise ValueError(
          'The two-phase region was not identified. It could be because\n'
          'of its narrowness or absence. Try to change the initial guess\n'
          'for pressure or stability test parameters. It also might be\n'
          'helpful to reduce the value of the `step`.'
        )

  @staticmethod
  def outputPT(
    eos: State2pPTEos,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float,
    kvi: Vector[Float],
    xi: Vector[Float],
  ) -> MultiPhaseState:
    Zy = eos.getPT_Z(P, T, yi)
    Zx = eos.getPT_Z(P, T, xi)
    pidy = eos.getPT_PID(P, T, yi)
    pidx = eos.getPT_PID(P, T, xi)
    RTP = R * T / P
    vy = Zy * RTP
    vx = Zx * RTP
    V = n * vy
    dy = yi.dot(eos.mwi) / vy
    dx = xi.dot(eos.mwi) / vx
    ni = n * yi
    fj = np_array([0., 1])
    yji = np_array([xi, yi])
    nj = np_array([0., n])
    nji = np_zeros_like(yji)
    nji[1] = ni
    Zj = np_array([Zx, Zy])
    vj = np_array([vx, vy])
    Vj = np_array([0., V])
    sj = np_array([0., 1])
    dj = np_array([dx, dy])
    pidj = np_array([pidx, pidy])
    kvji = np_atleast_2d(kvi)
    return MultiPhaseState(
      eos.Nc, 2, P, T, V, n, ni, nj, nji, fj, yji, Zj, vj, Vj, sj, dj, pidj,
      kvji,
    )
