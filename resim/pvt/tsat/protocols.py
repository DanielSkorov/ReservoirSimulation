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


class TsatSolverPTEos(Eos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to solve a saturation temperature
  problem. It must have the following attributes:
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
  | getPT_lnphii_dT     | A vector of shape `(Nc,)` of logarithms      |
  |                     | of fugacity coefficients of components and   |
  |                     | a vector of shape `(Nc,)` of their partial   |
  |                     | derivatives with respect to temperature.     |
  +---------------------+----------------------------------------------+
  | getPT_lnphii_dT_dnj | Previous + a matrix of shape `(Nc, Nc)` of   |
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

  def getPT_lnphii_dT(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
  ) -> tuple[Vector[Float], Vector[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to
    temperature.

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
      respect to temperature.
    """
    pass

  def getPT_lnphii_dT_dnj(
    self,
    P: float,
    T: float,
    yi: Vector[Float],
    n: float,
  ) -> tuple[Vector[Float], Vector[Float], Matrix[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to
    temperature and mole numbers of components.

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
      respect to temperature,
    - a `Matrix[Float]` of shape `(Nc, Nc)` of partial derivatives of
      natural logarithms of fugacity coefficients of components with
      respect to mole numbers of components.
    """
    pass


class TsatPTEos(TsatSolverPTEos, State2pPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to determine saturation temperature
  of a mixture. It must have the following attributes:
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
  | getPT_lnphii_dT     | A vector of shape `(Nc,)` of logarithms      |
  |                     | of fugacity coefficients of components and   |
  |                     | a vector of shape `(Nc,)` of their partial   |
  |                     | derivatives with respect to temperature.     |
  +---------------------+----------------------------------------------+
  | getPT_lnphii_dT_dnj | Previous + a matrix of shape `(Nc, Nc)` of   |
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


class TsatSolver[Eos](Protocol):
  """A protocol for callable objects that can be used to find
  saturation temperature of a mixture for a given value of a
  thermodynamic parameter (e.g., pressure or volume depending
  on the used equation of state) and mole composition.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  P: float
    A thermodynamic parameter in SI units.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  kvi0: Vector[Float], shape (Nc,)
    An initial guess of k-values of `Nc` components.

  Tlow: float
    The saturation temperature lower bound [K].

  Tupp: float
    The saturation temperature upper bound [K].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation curve or the lower saturation curve.
    The cricondenbar serves as the dividing point between upper and
    lower phase boundaries.

  Returns
  -------
  A tuple containing:
  - saturation temperature [K],
  - a `Vector[Float]` of shape `(Nc,)` of k-values of `Nc` components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the trial phase.
  """
  def __call__(
    self,
    eos: Eos,
    P: float,
    yi: Vector[Float],
    kvi0: Vector[Float],
    Tlow: float,
    Tupp: float,
    upper: bool,
    /,
  ) -> tuple[float, Vector[Float], Vector[Float]]:
    pass


class TsatRoutine[Eos](Protocol):
  """A protocol for callable objects that can be used to find a
  saturation state of a mixture for a given thermodynamics parameter
  (e.g., pressure or volume depending on the used equation of state)
  and mole composition.

  Parameters
  ----------
  eos: Eos
    An initialized instance of an equation of state.

  P: float
    A thermodynamic parameter in SI units.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  n: float
    The mole number of a mixture [mol].

  upper: bool
    A boolean flag that indicates whether the desired value is located
    at the upper saturation curve or the lower saturation curve.

  init: tuple[Vector[Float], float, float] | State | None
    This parameter is used to initialize a saturation temperature
    calculation procedure. The detailed explanation of the logic
    behind different types of `init` is given in the following table:

    +------------------+-----------------------------------------------+
    | Type             | Description                                   |
    +==================+===============================================+
    | tuple[           | A tuple, containing:                          |
    |   Vector[Float], | - initial guess of k-values as a vector of    |
    |   float,         |   shape `(Nc,)`,                              |
    |   float,         | - a lower bound of temperature [K],           |
    | ]                | - an upper bound of temperature [K].          |
    +------------------+-----------------------------------------------+
    | State            | A state of a mixture, k-values and            |
    |                  | temperature of which can be used to           |
    |                  | initialize a routine.                         |
    +------------------+-----------------------------------------------+
    | None             | An internal procedure should be used to       |
    |                  | obtain initial temperature and k-values.      |
    +------------------+-----------------------------------------------+

  **kwargs
    Other parameters for an internal initialization procedure.

  Returns
  -------
  A saturation state of a mixture.
  """
  def __call__(
    self,
    eos: Eos,
    P: float,
    yi: Vector[Float],
    n: float,
    upper: bool,
    init: tuple[Vector[Float], float, float] | State | None,
    **kwargs,
  ) -> MultiPhaseState:
    pass
