from typing import (
  Protocol,
)

from resim.pvt.datatypes import (
  Float,
  Matrix,
  State,
  Vector,
)


class PhaseFinder(Protocol):
  """A protocol for callable objects that can be used to return an index
  of a phase for a given state of a mixture and phase designation index.

  Parameters
  ----------
  state: State
    A state of a mixture as an instance of `State`.

  pid: int
    A phase designation index.

  Returns
  -------
  Index of a phase.
  """
  def __call__(self, state: State, pid: int, /) -> int:
    pass


class LinearSolver(Protocol):
  r"""A protocol for callable objects that can be used to solve linear
  systems.

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
  def __call__(self, A: Matrix[Float], b: Vector[Float], /) -> Vector[Float]:
    pass


class EigenSolver(Protocol):
  """A protocol for callable objects that can be used to find an
  eigenvector and corresponding eigenvalue of a matrix.

  Parameters
  ----------
  A: Matrix[Float], shape (N, N)
    A `Matrix[Float]` of shape `(N, N)` for which an eigenvector and
    corresponding eigenvalue must be found.

  x0: Vector[Float], shape (N,) | None
    An initial guess for an eigenvector as a `Vector[Float]` of shape
    `(N,)`. If it is `None`, an internal procedure should be used to
    estimate the initial guess.

  lmbd0: float | None
    An initial guess for an eigenvalue. If it is `None`, an internal
    procedure should be used to estimate the initial guess.

  Returns
  -------
  A tuple containing:
  - an eigenvector as a `Vector[Float]` of shape `(N,)`,
  - an eigenvalue.
  """
  def __call__(
    self,
    A: Matrix[Float],
    x0: Vector[Float] | None,
    lmbd0: float | None,
    /,
  ) -> tuple[Vector[Float], float]:
    pass
