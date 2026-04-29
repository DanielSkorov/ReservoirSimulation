from typing import (
  cast,
)

from math import (
  sqrt,
  inf,
)

from numpy import (
  fill_diagonal as np_fill_diagonal,
  ones as np_ones,
)

from numpy.linalg import (
  solve as np_lusolver,
)

from resim.pvt.datatypes import (
  Float,
  Matrix,
  State,
  Vector,
)

from resim.pvt.utils.protocols import (
  LinearSolver,
)


def findex(state: State, pid: int) -> int:
  """Find the index of the first phase which ID is equal to a given
  value.

  Parameters
  ----------
  state: State
    A thermodynamic state of a mixture.

  pid: int
    The desired phase ID (`0` = vapour, `1` = liquid, etc.).

  Returns
  -------
  An index of the first match.

  Raises
  ------
  IndexError
    This exception is raised if there is no phase with ID `pid` in the
    `state` of a mixture.
  """
  i = 0
  for s in state.pidj:
    if s == pid:
      return i
    i += 1
  else:
    raise IndexError(f'There is no phase with ID: "{pid}" in the state.')


def denmin(state: State, pid: int) -> int:
  """Find the index of a phase with the lowest density which ID is
  equal to a given value.

  Parameters
  ----------
  state: State
    A thermodynamic state of a mixture.

  pid: int
    The desired phase ID (`0` = vapour, `1` = liquid, etc.).

  Returns
  -------
  The index of the phase with the lowest density which ID is equal to a
  given value.

  Raises
  ------
  IndexError
    This exception is raised if there is no phase with ID `pid` in the
    `state` of a mixture.
  """
  idx: int | None = None
  den = inf
  for i, (s, d) in enumerate(zip(state.pidj, state.dj)):
    if s == pid and d < den:
      den = d
      idx = i
  if idx is None:
    raise IndexError(f'There is no phase with ID: "{pid}" in the state.')
  else:
    return idx


def denmax(state: State, pid: int) -> int:
  """Find the index of a phase with the largest density which ID is
  equal to a given value.

  Parameters
  ----------
  state: State
    A thermodynamic state of a mixture.

  pid: int
    The desired phase ID (`0` = vapour, `1` = liquid, etc.).

  Returns
  -------
  The index of the phase with the largest density which ID is equal to
  a given value.

  Raises
  ------
  IndexError
    This exception is raised if there is no phase with ID `pid` in the
    `state` of a mixture.
  """
  idx: int | None = None
  den = -1.
  for i, (s, d) in enumerate(zip(state.pidj, state.dj)):
    if s == pid and d > den:
      den = d
      idx = i
  if idx is None:
    raise IndexError(f'There is no phase with ID: "{pid}" in the state.')
  else:
    return idx


def lusolver(A: Matrix[Float], b: Vector[Float]) -> Vector[Float]:
  r"""Solve a linear matrix equation. This function is a cover for
  `numpy.linalg.solve` with custom type narrowing.

  Parameters
  ----------
  A: Matrix[Float]
    Coefficient matrix.

  b: Vector[Float]
    Ordinate or "dependent variable" values.

  Returns
  -------
  A `Vector[Float]`, which is the solution to the system of linear
  equations :math:`\mathbf{A}^\top \mathbf{x} = \mathbf{b}`.
  """
  return cast(Vector[Float], np_lusolver(A, b))


def rqi(
  Q: Matrix[Float],
  x0: Vector[Float] | None = None,
  lmbd0: float | None = None,
  tol: float = 1e-14,
  maxiter: int = 20,
  linsolver: LinearSolver = lusolver,
) -> tuple[Vector[Float], float]:
  """
  """
  if x0 is None:
    xk = np_ones(shape=(Q.shape[0],))
  else:
    xk = x0
  if lmbd0 is None:
    lmbdk = xk.dot(Q).dot(xk)
  else:
    lmbdk = lmbd0
  diagQ = Q.diagonal()
  M = Q.copy()
  np_fill_diagonal(M, diagQ - lmbdk)
  xkp1 = linsolver(M, xk)
  xkp1 /= sqrt(xkp1.dot(xkp1))
  lmbdkp1 = xkp1.dot(Q).dot(xkp1)
  k = 1
  while abs((lmbdkp1 - lmbdk) / lmbdk) > tol and k < maxiter:
    xk = xkp1
    lmbdk = lmbdkp1
    np_fill_diagonal(M, diagQ - lmbdk)
    xkp1 = linsolver(M, xk)
    xkp1 /= sqrt(xkp1.dot(xkp1))
    lmbdkp1 = xkp1.dot(Q).dot(xkp1)
    k += 1
  return xkp1, lmbdkp1
