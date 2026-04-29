from logging import (
  getLogger,
)

from typing import (
  cast,
)

from math import (
  isfinite,
  log,
)

from numpy import (
  abs as np_abs,
  array as np_array,
  atleast_2d as np_atleast_2d,
  exp as np_exp,
  fill_diagonal as np_fill_diagonal,
  linspace as np_linspace,
  log as np_log,
  sqrt as np_sqrt,
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

from resim.pvt.tmax.protocols import (
  TmaxPTEos,
  TmaxSolver,
  TmaxSolverPTEos,
)


logger = getLogger('tmax')


class TmaxConvergenceError(Exception):
  """An exception that will be raised if a cricondentherm solver does
  not converge.
  """
  def __init__(
    self,
    msg: str = ('The cricondentherm point calculation was\nterminated '
                'unsuccessfully. Try to improve the initial guess. '
                'It may\nalso be advisable to change the solver '
                'or increase the number of\niterations.'),
  ) -> None:
    super().__init__(msg)
    pass


def _tmaxPT_ss(
  eos: TmaxSolverPTEos,
  yi: Vector[Float],
  P0: float,
  T0: float,
  kvi0: Vector[Float],
  tol: float = 1e-20,
  maxiter: int = 300,
  tol_tpd: float = 1e-10,
  maxiter_tpd: int = 12,
) -> tuple[float, float, Vector[Float], Vector[Float]]:
  """The successive substitution (SS) method for the cricondentherm
  calculation using a PT-based equation of state.

  Parameters
  ----------
  eos: TmaxSolverPTEos
    An initialized instance of a PT-based equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  P0: float
    An initial guess of the cricondentherm pressure [Pa].

  T0: float
    An initial guess of the cricondentherm temperature [K].

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the gradient is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `300`.

  tol_tpd: float
    Terminate solvers for the TPD- and cricondentherm equations
    successfully if the absolute value of each equation is less
    than `tol_tpd`. Default is `1e-10`.

  maxiter_tpd: int
    The maximum number of iterations for solvers of the TPD- and
    cricondentherm equations. Default is `12`.

  Returns
  -------
  A tuple containing:
  - pressure of the cricondentherm point [Pa],
  - temperature of the cricondentherm point [K],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.

  Raises
  ------
  TmaxConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  To determine the cricondentherm, the algorithm implements successive
  substitution iterations to find a non-trivial local minimum of the
  tangent-plane distance (TPD) function. In the inner loops, the TPD-
  equation, which is the equation of equality to zero of the TPD-
  function, and the cricondentherm equation, which is the equation of
  equality to zero of the partial derivative of the TPD-function with
  respect to pressure, are solved using Newton's method.

  For the details of the algorithm, see the paper of L.X. Nghiem
  [et al], 1985 (doi: 10.1016/0378-3812(85)90059-7).
  """
  logger.info('Cricondentherm calculation using the SS-method.')
  Nc = eos.Nc
  logger.info('yi =' + Nc * '%7.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%13s%9s%11s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P [Pa]', 'T [K]', 'g2', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%13.1f%9.2f%11.2e%11.2e%11.2e'
  lnyi = np_log(yi)
  k = 0
  Pk = P0
  Tk = T0
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  r = 0
  lnkvi = np_log(xi) - lnyi
  lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
  lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  while (TPD < -tol_tpd or TPD > tol_tpd) and r < maxiter_tpd:
    dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    r += 1
    Tk -= TPD / dTPDdT
    lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
    lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  r = 0
  lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
  lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
  dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
  while (dTPDdP < -tol_tpd or dTPDdP > tol_tpd) and r < maxiter_tpd:
    d2TPDdP2 = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
    r += 1
    Pk -= dTPDdP / d2TPDdP2
    lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
    lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
  gi = lnki + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  TPD = xi.dot(gi - log(n))
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
  logger.debug(tmpl, k, *lnki, Pk, Tk, g2, TPD, dTPDdP)
  while repeat and k < maxiter:
    lnki -= gi
    k += 1
    ki = np_exp(lnki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    r = 0
    lnkvi = np_log(xi) - lnyi
    lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
    lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    while (TPD < -tol_tpd or TPD > tol_tpd) and r < maxiter_tpd:
      dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
      r += 1
      Tk -= TPD / dTPDdT
      lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
      lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
      TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    r = 0
    lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
    lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    while (dTPDdP < -tol_tpd or dTPDdP > tol_tpd) and r < maxiter_tpd:
      d2TPDdP2 = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
      r += 1
      Pk -= dTPDdP / d2TPDdP2
      lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
      lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
      dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    TPD = xi.dot(gi - log(n))
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
    logger.debug(tmpl, k, *lnki, Pk, Tk, g2, TPD, dTPDdP)
  if not repeat and isfinite(g2) and isfinite(Pk) and isfinite(Tk):
    return Pk, Tk, ki, xi
  logger.warning(
    "The SS-method for cricondentherm calculation does not converge.\n"
    "EOS: %s.\nParameters:\nyi = %s\nP0 = %s [Pa]\nT0 = %s [K]\nkvi0 = %s",
    eos.name, yi.tolist(), P0, T0, kvi0.tolist(),
  )
  raise TmaxConvergenceError()


def _tmaxPT_qnss(
  eos: TmaxSolverPTEos,
  yi: Vector[Float],
  P0: float,
  T0: float,
  kvi0: Vector[Float],
  lmbdmax: float = 30.,
  tol: float = 1e-20,
  maxiter: int = 200,
  tol_tpd: float = 1e-10,
  maxiter_tpd: int = 12,
) -> tuple[float, float, Vector[Float], Vector[Float]]:
  """The quasi-newton successive substitution (QNSS) method for
  the cricondentherm calculation using a PT-based equation of state.

  Parameters
  ----------
  eos: TmaxSolverPTEos
    An initialized instance of a PT-based equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  P0: float
    An initial guess of the cricondentherm pressure [Pa].

  T0: float
    An initial guess of the cricondentherm temperature [K].

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  lmbdmax: float
    The maximum step length. Default is `30.0`.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the gradient is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `200`.

  tol_tpd: float
    Terminate solvers for the TPD- and cricondentherm equations
    successfully if the absolute value of each equation is less
    than `tol_tpd`. Default is `1e-10`.

  maxiter_tpd: int
    The maximum number of iterations for solvers of the TPD- and
    cricondentherm equations. Default is `12`.

  Returns
  -------
  A tuple containing:
  - pressure of the cricondentherm point [Pa],
  - temperature of the cricondentherm point [K],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.

  Raises
  ------
  TmaxConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  To determine the cricondentherm, the algorithm implements quasi-newton
  successive substitution iterations to find a non-trivial local minimum
  of the tangent-plane distance (TPD) function. In the inner loops, the
  TPD-equation, which is the equation of equality to zero of the TPD-
  function, and the cricondentherm equation, which is the equation of
  equality to zero of the partial derivative of the TPD-function with
  respect to pressure, are solved using Newton's method.

  For the details of the algorithm, see the following papers:
  - L.X. Nghiem [et al], 1985 (doi: 10.1016/0378-3812(85)90059-7),
  - L.X. Nghiem and Y.-K. Li, 1984 (doi: 10.1016/0378-3812(84)80013-8).
  """
  logger.info('Cricondentherm calculation using the QNSS-method.')
  Nc = eos.Nc
  logger.info('yi =' + Nc * '%7.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%13s%9s%11s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P [Pa]', 'T [K]', 'g2', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%13.1f%9.2f%11.2e%11.2e%11.2e'
  lnyi = np_log(yi)
  k = 0
  Pk = P0
  Tk = T0
  ki = kvi0
  lnki = np_log(ki)
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  r = 0
  lnkvi = np_log(xi) - lnyi
  lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
  lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  while (TPD < -tol_tpd or TPD > tol_tpd) and r < maxiter_tpd:
    dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    r += 1
    Tk -= TPD / dTPDdT
    lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
    lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  r = 0
  lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
  lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
  dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
  while (dTPDdP < -tol_tpd or dTPDdP > tol_tpd) and r < maxiter_tpd:
    d2TPDdP2 = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
    r += 1
    Pk -= dTPDdP / d2TPDdP2
    lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
    lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
  gi = lnki + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  TPD = xi.dot(gi - log(n))
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
  logger.debug(tmpl, k, *lnki, Pk, Tk, g2, TPD, dTPDdP)
  lmbd = 1.
  dlnki = -gi
  while repeat and k < maxiter:
    k += 1
    tkm1 = dlnki.dot(gi)
    lnki += dlnki
    ki = np_exp(lnki)
    ni = ki * yi
    n = ni.sum()
    xi = ni / n
    r = 0
    lnkvi = np_log(xi) - lnyi
    lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
    lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    while (TPD < -tol_tpd or TPD > tol_tpd) and r < maxiter_tpd:
      dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
      r += 1
      Tk -= TPD / dTPDdT
      lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
      lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
      TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    r = 0
    lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
    lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    while (dTPDdP < -tol_tpd or dTPDdP > tol_tpd) and r < maxiter_tpd:
      d2TPDdP2 = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
      r += 1
      Pk -= dTPDdP / d2TPDdP2
      lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
      lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
      dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    gi = lnki + lnphixi - lnphiyi
    g2 = gi.dot(gi)
    TPD = xi.dot(gi - log(n))
    logger.debug(tmpl, k, *lnki, Pk, Tk, g2, TPD, dTPDdP)
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
    if k % Nc == 0:
      lmbd = 1.
      dlnki = -gi
    else:
      lmbd *= tkm1 / (dlnki.dot(gi) - tkm1)
      if lmbd < 0.:
        lmbd = -lmbd
      if lmbd > lmbdmax:
        lmbd = lmbdmax
      dlnki = -lmbd * gi
      max_dlnki = np_abs(dlnki).max()
      if max_dlnki > 6.:
        relax = 6. / max_dlnki
        lmbd *= relax
        dlnki *= relax
  if not repeat and isfinite(g2) and isfinite(Pk) and isfinite(Tk):
    return Pk, Tk, ki, xi
  logger.warning(
    "The QNSS-method for cricondentherm calculation does not converge.\n"
    "EOS: %s.\nParameters:\nyi = %s\nP0 = %s [Pa]\nT0 = %s [K]\nkvi0 = %s",
    eos.name, yi.tolist(), P0, T0, kvi0.tolist(),
  )
  raise TmaxConvergenceError()


def _tmaxPT_newt(
  eos: TmaxSolverPTEos,
  yi: Vector[Float],
  P0: float,
  T0: float,
  kvi0: Vector[Float],
  tol: float = 1e-20,
  maxiter: int = 100,
  tol_tpd: float = 1e-10,
  maxiter_tpd: int = 12,
  linsolver: LinearSolver = lusolver,
) -> tuple[float, float, Vector[Float], Vector[Float]]:
  r"""Newton's method for the cricondentherm calculation using a PT-
  based equation of state.

  Parameters
  ----------
  eos: TmaxSolverPTEos
    An initialized instance of a PT-based equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  P0: float
    An initial guess of the cricondentherm pressure [Pa].

  T0: float
    An initial guess of the cricondentherm temperature [K].

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  tol: float
    Terminate the solver successfully if the sum of squared elements
    of the gradient is less than `tol`. Default is `1e-20`.

  maxiter: int
    The maximum number of solver iterations. Default is `100`.

  tol_tpd: float
    Terminate solvers for the TPD- and cricondentherm equations
    successfully if the absolute value of each equation is less
    than `tol_tpd`. Default is `1e-10`.

  maxiter_tpd: int
    The maximum number of iterations for solvers of the TPD- and
    cricondentherm equations. Default is `12`.

  linsolver: LinearSolver
    A callable object that takes an `A: Matrix[Float]` of shape
    `(Nc, Nc)` and a `b: Vector[Float]` of shape `(Nc,)` and finds
    `x: Vector[Float]` of shape `(Nc,)`, which is the solution to the
    linear system :math:`\mathbf{A}^\top \mathbf{x} = \mathbf{b}`. The
    matrix `A` is a symmetric matrix. Default is `lusolver`.

  Returns
  -------
  A tuple containing:
  - pressure of the cricondentherm point [Pa],
  - temperature of the cricondentherm point [K],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.

  Raises
  ------
  TmaxConvergenceError
    This exception is raised if the solver does not converge.

  Notes
  -----
  To determine the cricondentherm, the algorithm finds a non-trivial
  local minimum of Michelsen's modified tangent-plane distance (TPD)
  function using Newton's method. In the inner loops, the TPD-equation,
  which is the equation of equality to zero of the TPD-function, and
  the cricondentherm equation, which is the equation of equality to
  zero of the partial derivative of the TPD- function with respect to
  pressure, are solved using Newton's method.

  For the details of the algorithm, see the paper of L.X. Nghiem
  [et al], 1985 (doi: 10.1016/0378-3812(85)90059-7).
  """
  logger.info("Cricondentherm calculation using Newton's method (C-form).")
  Nc = eos.Nc
  logger.info('yi =' + Nc * '%7.4f', *yi)
  logger.debug(
    '%3s' + Nc * '%9s' + '%13s%9s%11s%11s%11s',
    'Nit', *['lnkv%s' % s for s in range(Nc)],
    'P [Pa]', 'T [K]', 'g2', 'TPD', 'dTPDdP',
  )
  tmpl = '%3s' + Nc * '%9.4f' + '%13.1f%9.2f%11.2e%11.2e%11.2e'
  lnyi = np_log(yi)
  k = 0
  Pk = P0
  Tk = T0
  ki = kvi0
  ni = ki * yi
  n = ni.sum()
  xi = ni / n
  r = 0
  lnkvi = np_log(xi) - lnyi
  lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
  lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
  TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  while (TPD < -tol_tpd or TPD > tol_tpd) and r < maxiter_tpd:
    dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
    r += 1
    Tk -= TPD / dTPDdT
    lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
    lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
  r = 0
  lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
  lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
  dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
  while (dTPDdP < -tol_tpd or dTPDdP > tol_tpd) and r < maxiter_tpd:
    d2TPDdP2 = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
    r += 1
    Pk -= dTPDdP / d2TPDdP2
    lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
    lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
  gi = np_log(ki) + lnphixi - lnphiyi
  g2 = gi.dot(gi)
  TPD = xi.dot(gi - log(n))
  sqrtni = np_sqrt(ni)
  alphai = 2. * sqrtni
  repeat = (g2 > tol or
            TPD < -tol_tpd or TPD > tol_tpd or
            dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
  logger.debug(tmpl, k, *alphai, Pk, Tk, g2, TPD, dTPDdP)
  while repeat and k < maxiter:
    H = sqrtni[:,None] * sqrtni * eos.getPT_lnphii_dnj(Pk, Tk, xi, n)[1]
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
    lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
    lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
    TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    while (TPD < -tol_tpd or TPD > tol_tpd) and r < maxiter_tpd:
      dTPDdT = xi.dot(dlnphixidT - dlnphiyidT)
      r += 1
      Tk -= TPD / dTPDdT
      lnphiyi, dlnphiyidT = eos.getPT_lnphii_dT(Pk, Tk, yi)
      lnphixi, dlnphixidT = eos.getPT_lnphii_dT(Pk, Tk, xi)
      TPD = xi.dot(lnkvi + lnphixi - lnphiyi)
    r = 0
    lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
    lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
    dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    while (dTPDdP < -tol_tpd or dTPDdP > tol_tpd) and r < maxiter_tpd:
      d2TPDdP2 = xi.dot(d2lnphixidP2 - d2lnphiyidP2)
      r += 1
      Pk -= dTPDdP / d2TPDdP2
      lnphiyi, dlnphiyidP, d2lnphiyidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, yi)
      lnphixi, dlnphixidP, d2lnphixidP2 = eos.getPT_lnphii_dP_dP2(Pk, Tk, xi)
      dTPDdP = xi.dot(dlnphixidP - dlnphiyidP)
    gi = np_log(ni) + lnphixi - lnphiyi - lnyi
    g2 = gi.dot(gi)
    TPD = xi.dot(gi - log(n))
    repeat = (g2 > tol or
              TPD < -tol_tpd or TPD > tol_tpd or
              dTPDdP < -tol_tpd or dTPDdP > tol_tpd)
    logger.debug(tmpl, k, *alphai, Pk, Tk, g2, TPD, dTPDdP)
  if not repeat and isfinite(g2) and isfinite(Pk) and isfinite(Tk):
    return Pk, Tk, xi / yi, xi
  logger.warning(
    "Newton's method for cricondentherm calculation does not converge.\n"
    "EOS: %s.\nParameters:\nyi = %s\nP0 = %s [Pa]\nT0 = %s [K]\nkvi0 = %s",
    eos.name, yi.tolist(), P0, T0, kvi0.tolist(),
  )
  raise TmaxConvergenceError()


class tmax(object):
  def __init__(
    self,
    state: State | None = None,
  ) -> None:
    """Specify settings for the cricondentherm calculation.

    Parameters
    ----------
    state: State | None
      A thermodynamic state of a mixture, pressure, temperature, and
      k-values of which can be used to initialize the cricondentherm
      calculation procedure. If it is `None`, the option to use
      previously calculated results is not available until such a state
      is passed directly to the callable instance of this class. Default
      is `None`.

    Notes
    -----
    These settings influence the calculation only if the corresponding
    parameters of the `__call__` method of this class are set to `None`.
    """
    self.state = state
    pass

  def __call__(
    self,
    eos: TmaxPTEos,
    yi: Vector[Float],
    n: float = 1.,
    init: tuple[float, float, Vector[Float]] | State | None = None,
    solver: TmaxSolver | str = 'default',
    **kwargs,
  ) -> MultiPhaseState:
    """Determine the cricondentherm point (state) of a mixture.

    Parameters
    ----------
    eos: TmaxPTEos
      An initialized instance of an equation of state.

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture [mol]. Default is `1.0` [mol].

    init: tuple[float, float, Vector[Float]] | State | None
      This parameter is used to initialize the cricondentherm state
      calculation procedure. The detailed explanation of the logic
      behind different types of `init` is given in the following table:

      +------------------+---------------------------------------------+
      | Type             | Description                                 |
      +==================+=============================================+
      | tuple[           | A tuple, containing:                        |
      |   float,         | - pressure [Pa],                            |
      |   float,         | - temperature [K].                          |
      |   Vector[Float], | - k-values as a vector of shape `(Nc,)`     |
      | ]                |                                             |
      +------------------+---------------------------------------------+
      | State            | A state of a mixture, pressure,             |
      |                  | temperature, and k-values of which can be   |
      |                  | used to initialize the procedure. The       |
      |                  | extrapolation routine is used to clarify    |
      |                  | this initial guess.                         |
      +------------------+---------------------------------------------+
      | None             | Perform the 2D-gridding routine exploiting  |
      |                  | the stability test procedure to find a      |
      |                  | state close to the cricondentherm point.    |
      |                  | Specifying `Pmin`, `Pmax`, `Tmin`, and      |
      |                  | `Tmax` for the gridding procedure is highly |
      |                  | recommended.                                |
      +------------------+---------------------------------------------+

      Default is `None`.

    solver: TmaxSolver | str
      A callable object that can solve the cricondentherm problem.
      It also can be a string defining the name of an internal solver.

      For the PT-thermodynamics, the following internal solvers are
      available:

      +-----------------+----------------------------------------------+
      | Internal solver | Description                                  |
      +=================+==============================================+
      | `'ss'`          | Uses successive substitution iterations to   |
      |                 | solve the system of equilibrium equations    |
      |                 | and Newton's method for the TPD-equation and |
      |                 | the cricondentherm equation.                 |
      +-----------------+----------------------------------------------+
      | `'qnss'`        | Uses quasi-newton successive substitution    |
      |                 | iterations (QNSS) to solve the system of     |
      |                 | equilibrium equations and Newton's method    |
      |                 | for the TPD-equation and the cricondentherm  |
      |                 | equation.                                    |
      +-----------------+----------------------------------------------+
      | `'newton'`      | Uses Newton's method to find a local minimum |
      |                 | of the modified Michelsen's TPD-function and |
      |                 | for the TPD-equation and the cricondentherm  |
      |                 | equation.                                    |
      +-----------------+----------------------------------------------+

      The following table lists default internal solvers for each
      formulation of the cricondentherm problem:

      +-----------------+----------------------------------------------+
      | Basic variables | Default internal solvers (`'default'`)       |
      +=================+==============================================+
      | P, T            | Newton (`'newton'`).                         |
      +-----------------+----------------------------------------------+

      Default is `'default'`.

    **kwargs
      Other parameters for an internal initialization procedure.

    Returns
    -------
    The cricondentherm point (state) of a mixture.

    Raises
    ------
    TmaxConvergenceError
      This exception is raised if a cricondentherm solver does not
      converge.
    """
    if init is None:
      init = self.state
    if eos.form == 'PT':
      eos = cast(TmaxPTEos, eos)
      solverPT: TmaxSolver[TmaxSolverPTEos]
      if callable(solver):
        solverPT = cast(TmaxSolver[TmaxSolverPTEos], solver)
      elif solver == 'default' or solver == 'newton':
        solverPT = _tmaxPT_newt
      elif solver == 'ss':
        solverPT = _tmaxPT_ss
      elif solver == 'qnss':
        solverPT = _tmaxPT_qnss
      else:
        raise ValueError(
          f'Unknown cricondentherm solver for a PT-based EOS: "{solver}".'
        )
      return self.runPT(eos, yi, n, init, solverPT, **kwargs)
    else:
      raise NotImplementedError(
        f'The {eos.form}-formulation of the cricondentherm calculation '
        'routine is not implemented yet.'
      )

  @classmethod
  def runPT(
    cls,
    eos: TmaxPTEos,
    yi: Vector[Float],
    n: float = 1.,
    init: tuple[float, float, Vector[Float]] | State | None = None,
    solver: TmaxSolver[TmaxSolverPTEos] = _tmaxPT_newt,
    **kwargs,
  ) -> MultiPhaseState:
    """Determine the cricondentherm point (state) of a mixture using a
    PT-based equation of state.

    Parameters
    ----------
    eos: TmaxPTEos
      An initialized instance of a PT-based equation of state.

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      The mole number of a mixture [mol]. Default is `1.0` [mol].

    init: tuple[float, float, Vector[Float]] | State | None
      This parameter is used to initialize the cricondentherm state
      calculation procedure. The detailed explanation of the logic
      behind different types of `init` is given in the following table:

      +------------------+---------------------------------------------+
      | Type             | Description                                 |
      +==================+=============================================+
      | tuple[           | A tuple, containing:                        |
      |   float,         | - pressure [Pa],                            |
      |   float,         | - temperature [K].                          |
      |   Vector[Float], | - k-values as a vector of shape `(Nc,)`     |
      | ]                |                                             |
      +------------------+---------------------------------------------+
      | State            | A state of a mixture, pressure,             |
      |                  | temperature, and k-values of which can be   |
      |                  | used to initialize the procedure. The       |
      |                  | extrapolation routine is used to clarify    |
      |                  | this initial guess.                         |
      +------------------+---------------------------------------------+
      | None             | Perform the 2D-gridding routine exploiting  |
      |                  | the stability test procedure to find a      |
      |                  | state close to the cricondentherm point.    |
      |                  | Specifying `Pmin`, `Pmax`, `Tmin`, and      |
      |                  | `Tmax` for the gridding procedure is highly |
      |                  | recommended.                                |
      +------------------+---------------------------------------------+

      Default is `None`.

    solver: TmaxSolver[TmaxSolverPTEos]
      A callable object that can solve the cricondentherm problem
      formulated for the PT-thermodynamics. Default is `_tmaxPT_newt`.

    **kwargs
      Other parameters for an internal initialization procedure.

    Returns
    -------
    The cricondetherm point (state) of a mixture.

    Raises
    ------
    TmaxConvergenceError
      This exception is raised if the cricondentherm solver does not
      converge.
    """
    if isinstance(init, tuple):
      P0, T0, kvi0 = init
    else:
      if init is None:
        P0, T0, kvi0 = cls.gridding(eos, yi, **kwargs)
      else:
        kvjim1 = init.kvji
        if kvjim1 is None:
          logger.info(
            'K-values obtained from the given state cannot be used '
            'to initialize the cricondentherm routine because they '
            'are `None`.'
          )
          P0, T0, kvi0 = cls.gridding(eos, yi, **kwargs)
        elif init.Np > 2:
          logger.info(
            'K-values obtained from the given state cannot be used '
            'to initialize the cricondentherm routine because the '
            'number of phases is greater than two.'
          )
          P0, T0, kvi0 = cls.gridding(eos, yi, **kwargs)
        else:
          P0 = init.P
          T0 = init.T
          kvi0 = kvjim1.ravel()
      # TODO: Apply the extrapolation procedure to clarify the initial
      #       guess of pressure, temperature, and k-values. For details,
      #       see L.X. Nghiem and Y.K. Li, 1990 (doi: 10.2118/13517-PA).
    P, T, kvi, xi = solver(eos, yi, P0, T0, kvi0)
    return cls.outputPT(eos, P, T, yi, n, kvi, xi)

  @staticmethod
  def gridding(
    eos: StabPTEos,
    yi: Vector[Float],
    stabroutine: StabRoutine[StabPTEos] = stabtest.runPT,
    Pmin: float = 101325.,
    Pmax: float = 1e8,
    Tmin: float = 173.15,
    Tmax: float = 973.15,
    Pnodes: int = 10,
    Tnodes: int = 20,
  ) -> tuple[float, float, Vector[Float]]:
    """Perform the 2D-gridding procedure to obtain the initial guess
    of pressure, temperature, and k-values for the cricondentherm state
    of a mixture.

    Parameters
    ----------
    eos: StabPTEos
      An initialized instance of a PT-based equation of state.

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    stabroutine: StabRoutine[StabPTEos]
      A callable object that can be used to perform the one-phase
      stability test. Default is `stabtest.runPT`.

    Pmin: float
      The minimum pressure for the gridding procedure. Default is
      `101325.0` [Pa].

    Pmax: float
      The maximum pressure for the gridding procedure. Default is `1e8`
      [Pa].

    Tmin: float
      The minimum temperature for the gridding procedure. Default is
      `173.15` [K].

    Tmax: float
      The maximum temperature for the gridding procedure. Default is
      `973.15` [K].

    Pnodes: int
      The number of nodes to construct a grid along the pressure axis.
      Default is `10`.

    Tnodes: int
      The number of nodes to construct a grid along the temperature
      axis. Default is `20`.

    Returns
    -------
    A tuple containing:
    - pressure [Pa],
    - temperature [K],
    - k-values of `Nc` components as a `Vector[Float]` of shape `(Nc,)`.

    Raises
    ------
    ValueError
      This exception will be raised if a change in stability of a one-
      phase state is not found for all temperatures and pressures.

    Notes
    -----
    The 2D-gridding procedure performs the following steps:

    1. The algorithm generates linear grids along the pressure and
    temperature axes.

    2. For all pressures starting from `Pmin` to `Pmax` the one-phase
    stability of a mixture at `Tmax` is checked. If the one-phase state
    is stable, the algorithm searches for a switch in stability along
    the temperature axis between `Tmax` and `Tmin`. Otherwise, it
    continues to the next pressure.

    3. Once the switch in stability is found, the algorithm checks
    whether the current temperature is greater than a previously found
    stability switch temperature. If so, the algorithm updates the
    result and continues to the next pressure. Otherwise, the algorithm
    checks whether there is an increasing trend in the stability switch
    temperature. If so, the algorithm returns the result assuming that
    the current pressure is greater than the cricondentherm pressure.
    """
    PP = np_linspace(Pmin, Pmax, Pnodes, endpoint=True)
    TT = np_linspace(Tmax, Tmin, Tnodes, endpoint=True)
    res: tuple[float, float, Vector[Float]] | None = None
    increasing = False
    for P in PP:
      state = stabroutine(eos, P, Tmax, yi, 1., None)
      if state.kvji is None:
        for T in TT[1:]:
          state = stabroutine(eos, P, T, yi, 1., None)
          kvji = state.kvji
          if kvji is not None:
            if res is None:
              res = P, T, kvji.ravel()
            elif T >= res[1]:
              increasing = True
              res = P, T, kvji.ravel()
            elif increasing:
              return res
            break
        else:
          if res is not None:
            return res
    if res is None:
      raise ValueError(
        'A change in stability of the one-phase state of a mixture is '
        'not found for all temperatures and pressures.'
      )
    return res

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
