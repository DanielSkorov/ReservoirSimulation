from typing import (
  Iterable,
  Literal,
  Protocol,
)

from resim.pvt.datatypes import (
  Float,
  Vector,
  Matrix,
  State,
)

from resim.pvt.eos import (
  Eos,
  State2pPTEos,
)


class StabSolverPTEos(Eos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to solve one-phase stability problems.
  It must have the following attributes:

  +-----------+------+-------------------------------------------------+
  | Attribute | Type | Description                                     |
  +===========+======+=================================================+
  | name      | str  | The name of an EOS (for logging).               |
  +-----------+------+-------------------------------------------------+
  | Nc        | int  | The number of components in a system.           |
  +-----------+------+-------------------------------------------------+

  Any class that implements this protocol must also have methods:

  +------------------+-------------------------------------------------+
  | Method           | Result                                          |
  +==================+=================================================+
  | getPT_lnphii     | A vector of shape `(Nc,)` of natural logarithms |
  |                  | of fugacity coefficients of components.         |
  +------------------+-------------------------------------------------+
  | getPT_lnphii_dnj | Previous + a matrix of shape `(Nc, Nc)` of      |
  |                  | their partial derivatives with respect to mole  |
  |                  | numbers of components.                          |
  +------------------+-------------------------------------------------+
  """
  def getPT_lnphii(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> Vector[Float]:
    """Compute a vector of logarithms of fugacity coefficients of
    components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    A `Vector[Float]` of shape `(Nc,)` of natural logarithms of fugacity
    coefficients of components.
    """
    pass

  def getPT_lnphii_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float,
  ) -> tuple[Vector[Float], Matrix[Float]]:
    """Compute a vector of logarithms of fugacity coefficients of
    components and a matrix of their partial derivatives with respect
    to mole numbers of components.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    n: float
      Mole number of a mixture [mol].

    Returns
    -------
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients with respect to
      mole numbers of components.
    """
    pass


class StabPTEos(StabSolverPTEos, State2pPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to perform the stability test routine.
  It must have the following attributes:

  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | form      | Literal['PT'] | The formalism of an EOS.               |
  +-----------+---------------+----------------------------------------+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components in a system.  |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molecular weights of components        |
  |           |               | [kg/mol] as a vector of shape `(Nc,)`. |
  +-----------+---------------+----------------------------------------+

  Any class that implements this protocol must also have methods to
  solve one-phase stability problems:

  +------------------+-------------------------------------------------+
  | Method           | Result                                          |
  +==================+=================================================+
  | getPT_kvguess    | A sequence of initial guesses of k-values.      |
  +------------------+-------------------------------------------------+
  | getPT_lnphii     | A vector of shape `(Nc,)` of natural logarithms |
  |                  | of fugacity coefficients of components.         |
  +------------------+-------------------------------------------------+
  | getPT_lnphii_dnj | Previous + a matrix of shape `(Nc, Nc)` of      |
  |                  | their partial derivatives with respect to mole  |
  |                  | numbers of components.                          |
  +------------------+-------------------------------------------------+
  | getPT_PID        | The phase designation index of a mixture.       |
  +------------------+-------------------------------------------------+
  | getPT_Z          | The compressibility factor of a mixture.        |
  +------------------+-------------------------------------------------+
  """
  form: Literal['PT'] = 'PT'

  def getPT_kvguess(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> Iterable[Vector[Float]]:
    """Create an iterable object of initial guesses of k-values.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yi: Vector[Float], shape (Nc,)
      Mole fractions of `Nc` components.

    Returns
    -------
    An iterable object of vectors (with the shape `(Nc,)`) of initial
    guesses of k-values.
    """
    pass


class StabSolver[Eos](Protocol):
  """A protocol for callable objects that can be used to solve one-phase
  stability problems.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  P1: float
    The first thermodynamic parameter in SI units.

  P2: float
    The second thermodynamic parameter in SI units.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  Returns
  -------
  A tuple containing:
  - a local minimum of the TPD-function,
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components
    corresponding to the found local minimum of the TPD-function.
  """
  def __call__(
    self,
    eos: Eos,
    P1: float,
    P2: float,
    yi: Vector[Float],
    kvi0: Vector[Float],
    /,
  ) -> tuple[float, Vector[Float]]:
    pass


class StabRoutine[Eos](Protocol):
  """A protocol for callable objects that can perform the one-phase
  stability test of a mixture for a given mole composition and two
  thermodynamic parameters.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  P1: float
    The first thermodynamic parameter is SI units.

  P2: float
    The second thermodynamic parameter is SI units.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    The mole number of a mixture [mol].

  init: Iterable[Vector[Float]] | State | None
    This parameter is used to initialize the stability test. The
    detailed explanation of the logic behind possible types of `init`
    is given in the following table:

    +-------------------------+----------------------------------------+
    | Type                    | Description                            |
    +=========================+========================================+
    | Iterable[Vector[Float]] | An iterable object of initial guesses  |
    |                         | of k-values (arrays of shape `(Nc,)`), |
    |                         | which are directly used by a routine.  |
    +-------------------------+----------------------------------------+
    | State                   | A state of a mixture, k-values from    |
    |                         | which can be used to initialize the    |
    |                         | stability test. In addition to these   |
    |                         | k-values, initial guesses are also     |
    |                         | obtained from the instance of an EOS.  |
    +-------------------------+----------------------------------------+
    | None                    | The `eos` is used to prepare initial   |
    |                         | guesses of k-values.                   |
    +-------------------------+----------------------------------------+

  Returns
  -------
  Stability test results as an instance of `State`.
  """
  def __call__(
    self,
    eos: Eos,
    P1: float,
    P2: float,
    yi: Vector[Float],
    n: float,
    init: Iterable[Vector[Float]] | State | None,
    /,
  ) -> State:
    pass
