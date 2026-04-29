from typing import (
  Iterable,
  Literal,
  Protocol,
)

from resim.pvt.datatypes import (
  Float,
  Matrix,
  MultiPhaseState,
  State,
  Vector,
)

from resim.pvt.eos import (
  Eos,
  State2pPTEos,
)


class PmaxSolverPTEos(Eos, Protocol):
  """A protocol for an initialized instance of a PT-based equation
  of state (PTEos) that can be used to solve cricondenbar problems.
  It must have the following attributes:
  +-----------+------+-------------------------------------------------+
  | Attribute | Type | Description                                     |
  +===========+======+=================================================+
  | name      | str  | The name of an EOS (for logging).               |
  +-----------+------+-------------------------------------------------+
  | Nc        | int  | The number of components in a system.           |
  +-----------+------+-------------------------------------------------+

  Any class that implements this protocol must also have methods:
  +---------------------+----------------------------------------------+
  | Method              | Result                                       |
  +=====================+==============================================+
  | getPT_lnphii_dnj    | Previous + a matrix of shape `(Nc, Nc)` of   |
  |                     | their partial derivatives with respect to    |
  |                     | mole numbers of components.                  |
  +---------------------+----------------------------------------------+
  | getPT_lnphii_dP     | A vector of shape `(Nc,)` of logarithms      |
  |                     | of fugacity coefficients of components and   |
  |                     | a vector of shape `(Nc,)` of their partial   |
  |                     | derivatives with respect to pressure.        |
  +---------------------+----------------------------------------------+
  | getPT_lnphii_dT_dT2 | A vector of shape `(Nc,)` of logarithms      |
  |                     | of fugacity coefficients of components and   |
  |                     | vectors of the same shape of their first and |
  |                     | second partial derivatives with respect to   |
  |                     | temperature.                                 |
  +---------------------+----------------------------------------------+
  """
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

  def getPT_lnphii_dP(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[Vector[Float], Vector[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to
    pressure.

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
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure.
    """
    pass

  def getPT_lnphii_dT_dT2(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[Vector[Float], Vector[Float], Vector[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their first and second partial derivatives with
    respect to temperature.

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
    A tuple containing:
    - a `Vector[Float]` of shape `(Nc,)` of natural logarithms of
      fugacity coefficients of components,
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to temperature,
    - a `Vector[Float]` of shape `(Nc,)` of second partial derivatives
      of natural logarithms of fugacity coefficients of components with
      respect to temperature.
    """
    pass


class PmaxPTEos(PmaxSolverPTEos, State2pPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to determine the cricondenbar point
  (state) of a mixture. It must have the following attributes:
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

  Any class that implements this protocol must also have methods:
  +---------------------+----------------------------------------------+
  | Method              | Result                                       |
  +=====================+==============================================+
  | getPT_kvguess       | A sequence of initial guesses of k-values.   |
  +---------------------+----------------------------------------------+
  | getPT_lnphii        | A vector of shape `(Nc,)` of logarithms      |
  |                     | of fugacity coefficients of components.      |
  +---------------------+----------------------------------------------+
  | getPT_lnphii_dnj    | Previous + a matrix of shape `(Nc, Nc)` of   |
  |                     | their partial derivatives with respect to    |
  |                     | mole numbers of components.                  |
  +---------------------+----------------------------------------------+
  | getPT_lnphii_dP     | A vector of shape `(Nc,)` of logarithms      |
  |                     | of fugacity coefficients of components and   |
  |                     | a vector of shape `(Nc,)` of their partial   |
  |                     | derivatives with respect to pressure.        |
  +---------------------+----------------------------------------------+
  | getPT_lnphii_dT_dT2 | A vector of shape `(Nc,)` of logarithms      |
  |                     | of fugacity coefficients of components and   |
  |                     | vectors of the same shape of their first and |
  |                     | second partial derivatives with respect to   |
  |                     | temperature.                                 |
  +---------------------+----------------------------------------------+
  | getPT_PID           | The phase designation index of a mixture.    |
  +---------------------+----------------------------------------------+
  | getPT_Z             | The compressibility factor of a mixture.     |
  +---------------------+----------------------------------------------+
  """
  form: Literal['PT']

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


class PmaxSolver[Eos](Protocol):
  """A protocol for callable objects that can be used to find two
  thermodynamic parameters that characterize the cricondenbar point
  (state) of a mixture.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  P10: float
    An initial guess of the first thermodynamic parameter in SI units.

  P20: float
    An initial guess of the second thermodynamic parameter in SI units.

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  Returns
  -------
  A tuple containing:
  - the first thermodynamic parameter in SI units corresponding to the
    cricondenbar point,
  - the second thermodynamic parameter in SI units corresponding to the
    cricondenbar point,
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.
  """
  def __call__(
    self,
    eos: Eos,
    yi: Vector[Float],
    P10: float,
    P20: float,
    kvi0: Vector[Float],
    /,
  ) -> tuple[float, float, Vector[Float], Vector[Float]]:
    pass


class PmaxRoutine[Eos](Protocol):
  """A protocol for callable objects that can be used to find the
  cricondenbar point (state) of a mixture.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    The mole number of a mixture [mol].

  init: tuple[float, float, Vector[Float]] | State | None
    This parameter is used to initialize a cricondenbar state
    calculation procedure. The detailed explanation of the logic
    behind different types of `init` is given in the following table:

    +------------------+-----------------------------------------------+
    | Type             | Description                                   |
    +==================+===============================================+
    | tuple[           | A tuple, containing:                          |
    |   float,         | - pressure [Pa],                              |
    |   float,         | - temperature [K].                            |
    |   Vector[Float], | - k-values as a vector of shape `(Nc,)`       |
    | ]                |                                               |
    +------------------+-----------------------------------------------+
    | State            | A state of a mixture, pressure, temperature,  |
    |                  | and k-values of which can be used to          |
    |                  | initialize a procedure.                       |
    +------------------+-----------------------------------------------+
    | None             | An internal procedure should be used to       |
    |                  | obtain initial pressure, temperature, and     |
    |                  | k-values.                                     |
    +------------------+-----------------------------------------------+

  **kwargs
    Other parameters for an internal initialization procedure.

  Returns
  -------
  The cricondenbar point (state) of a mixture.
  """
  def __call__(
    self,
    eos: Eos,
    yi: Vector[Float],
    n: float,
    init: tuple[float, float, Vector[Float]] | State | None,
    **kwargs,
  ) -> MultiPhaseState:
    pass
