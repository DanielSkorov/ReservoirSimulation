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


class PsatSolverPTEos(Eos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to solve a saturation pressure problem.
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
  | getPT_lnphii_dP_dnj | Previous + a matrix of shape `(Nc, Nc)` of   |
  |                     | their partial derivatives with respect to    |
  |                     | mole numbers of components.                  |
  +---------------------+----------------------------------------------+
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

  def getPT_lnphii_dP_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float,
  ) -> tuple[Vector[Float], Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to
    pressure and mole numbers of components.

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
    - a `Vector[Float]` of shape `(Nc,)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to pressure,
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to mole numbers of components.
    """
    pass


class PsatPTEos(PsatSolverPTEos, State2pPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to determine saturation pressure of a
  mixture. It must have the following attributes:
  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | form      | Literal['PT'] | A formalism of the EOS.                |
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
  | getPT_lnphii_dP_dnj | Previous + a matrix of shape `(Nc, Nc)` of   |
  |                     | their partial derivatives with respect to    |
  |                     | mole numbers of components.                  |
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


class PsatSolver[Eos](Protocol):
  """A protocol for callable objects that can be used to find saturation
  pressure for a given temperature and mole composition of a mixture.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  T: float
    Another thermodynamic parameter in SI units.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  Plow: float
    The saturation pressure lower bound [Pa].

  Pupp: float
    The saturation pressure upper bound [Pa].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation curve or the lower saturation curve.
    The cricondentherm serves as the dividing point between upper and
    lower phase boundaries.

  Returns
  -------
  A tuple containing:
  - saturation pressure [Pa],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.
  """
  def __call__(
    self,
    eos: Eos,
    T: float,
    yi: Vector[Float],
    kvi0: Vector[Float],
    Plow: float,
    Pupp: float,
    upper: bool,
    /,
  ) -> tuple[float, Vector[Float], Vector[Float]]:
    pass


class PsatRoutine[Eos](Protocol):
  """A protocol for callable objects that can be used to find a
  saturation state of a mixture for a given temperature and mole
  composition.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  T: float
    Temperature [K].

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    The mole number of a mixture [mol].

  upper: bool
    A boolean flag indicatin whether the desired value is located
    at the upper saturation curve or the lower saturation curve.

  init: tuple[Vector[Float], float, float] | State | None
    This parameter is used to initialize a saturation pressure
    calculation procedure. The detailed explanation of the logic
    behind different types of `init` is given in the following table:

    +------------------+-----------------------------------------------+
    | Type             | Description                                   |
    +==================+===============================================+
    | tuple[           | A tuple, containing:                          |
    |   Vector[Float], | - initial guess of k-values as a vector of    |
    |   float,         |   shape `(Nc,)`,                              |
    |   float,         | - a lower bound of pressure [Pa],             |
    | ]                | - an upper bound of pressure [Pa].            |
    +------------------+-----------------------------------------------+
    | State            | A state of a mixture, k-values and pressure   |
    |                  | of which can be used to initialize a routine. |
    +------------------+-----------------------------------------------+
    | None             | An internal procedure should be used to       |
    |                  | obtain initial pressure and k-values.         |
    +------------------+-----------------------------------------------+

  Pmin: float
    This parameter can be used by an internal initialization procedure.
    It defines the lower bound of the range of possible solutions.
    Default is `1.` [Pa].

  Pmax: float
    This parameter can be used by an internal initialization procedure.
    It defines the upper bound of the range of possible solutions.
    Default is `1e8` [Pa].

  **kwargs
    Other parameters for an internal initialization procedure.

  Returns
  -------
  A saturation state of a mixture.
  """
  def __call__(
    self,
    eos: Eos,
    T: float,
    yi: Vector[Float],
    n: float,
    upper: bool,
    init: tuple[Vector[Float], float, float] | State | None,
    **kwargs,
  ) -> MultiPhaseState:
    pass
