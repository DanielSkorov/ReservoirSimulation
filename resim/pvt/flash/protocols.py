from typing import (
  Iterable,
  Literal,
  Protocol,
)

from resim.pvt.datatypes import (
  Float,
  Matrix,
  State,
  Tensor,
  Vector,
)

from resim.pvt.eos import (
  Eos,
  State2pPTEos,
  StateNpPTEos,
)


class RR2pSolver(Protocol):
  """A protocol for callable objects that can be used to solve the
  Rachford-Rice equation.

  Parameters
  ----------
  kvi: Vector[Float], shape (Nc,)
    K-values of `Nc` components. At least one of them should be greater
    than one, and at least one should be lower than one.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  f0: float | None
    An initial guess for the mole fraction of the non-reference phase.
    If it is `None`, an internal formula should be used to obtain the
    initial guess.

  Returns
  -------
  The solution of the Rachford-Rice equation corresponding to the
  negative-flash window (mole fraction of the non-reference phase).
  """
  def __call__(
    self,
    kvi: Vector[Float],
    yi: Vector[Float],
    f0: float | None,
  ) -> float:
    pass


class RRNpSolver(Protocol):
  """A protocol for callable objects that can be used to solve a system
  Rachford-Rice equations.

  Parameters
  ----------
  Kji: Matrix[Float], shape (Np - 1, Nc)
    K-values of `Nc` components in `Np - 1` phases.

  yi: Vector[Float], shape (Nc,)
    Mole fractions of `Nc` components.

  fj0: Vector[Float], shape (Np - 1,) | None
    An initial guess for phase mole fractions. If it is `None`, an
    internal formula should be used to obtain the initial guess.

  Returns
  -------
  A `Vector[Float]` of shape `(Np - 1,)`, which is the solution of the
  system of Rachford-Rice equations corresponding to the negative-flash
  window (mole fractions of non-reference phases).
  """
  def __call__(
    self,
    Kji: Matrix[Float],
    yi: Vector[Float],
    fj0: Vector[Float] | None,
  ) -> Vector[Float]:
    pass


class Flash2pSolverPTEos(Eos, Protocol):
  """A protocol fpr an initialized instance of a PT-based equation of
  state (PTEos) that can be used to solve two-phase flash problems.
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


class FlashNpSolverPTEos(Flash2pSolverPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to solve multiphase flash problems.
  It must have the following attributes:
  +-----------+------+-------------------------------------------------+
  | Attribute | Type | Description                                     |
  +===========+======+=================================================+
  | name      | str  | The name of an EOS (for logging).               |
  +-----------+------+-------------------------------------------------+
  | Nc        | int  | The number of components in a system.           |
  +-----------+------+-------------------------------------------------+

  Any class that implements this protocol must also have methods:
  +-------------------+------------------------------------------------+
  | Method            | Result                                         |
  +===================+================================================+
  | getPT_lnphii      | Natural logarithms of fugacity coefficients of |
  |                   | components as a vector of shape `(Nc,)`.       |
  +-------------------+------------------------------------------------+
  | getPT_lnphii_dnj  | Previous + a matrix of shape `(Nc, Nc)` of     |
  |                   | their partial derivatives with respect to mole |
  |                   | numbers of components.                         |
  +-------------------+------------------------------------------------+
  | getPT_lnphiji     | Natural logarithms of fugacity coefficients of |
  |                   | components for each mixture as a matrix of     |
  |                   | shape `(Np, Nc)`.                              |
  +-------------------+------------------------------------------------+
  | getPT_lnphiji_dnj | Previous + a tensor of shape `(Np, Nc, Nc)` of |
  |                   | their partial derivatives with respect to mole |
  |                   | numbers of components.                         |
  +-------------------+------------------------------------------------+
  """
  def getPT_lnphiji(
    self,
    P: float,
    T: float,
    yji: Matrix[Float],
  ) -> Matrix[Float]:
    """Compute natural logarithms of fugacity coefficients of components
    for each mixture.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yji: Matrix[Float], shape (Np, Nc)
      A `Matrix[Float]` of shape `(Np, Nc)` of mole fractions of `Nc`
      components in each of `Np` mixtures.

    Returns
    -------
    A `Matrix[Float]` of shape `(Np, Nc)` of natural logarithms of
    fugacity coefficients of components in mixtures.
    """
    pass

  def getPT_lnphiji_dnk(
    self,
    P: float,
    T: float,
    yji: Matrix[Float],
    nj: Vector[Float],
  ) -> tuple[Matrix[Float], Tensor[Float]]:
    """Compute natural logarithms of fugacity coefficients of
    components and their partial derivatives with respect to
    mole numbers of components for each mixture.

    Parameters
    ----------
    P: float
      Pressure [Pa].

    T: float
      Temperature [K].

    yji: Matrix[Float], shape (Np, Nc)
      A `Matrix[Float]` of shape `(Np, Nc)` of mole fractions of `Nc`
      components in each of `Np` mixtures.

    Returns
    -------
    A tuple containing:
    - a `Matrix[Float]` of shape `(Np, Nc)` of natural logarithms of
      fugacity coefficients of components in mixtures,
    - a `Tensor[Float]` of shape `(Np, Nc, Nc)` of partial derivatives
      of natural logarithms of fugacity coefficients of components with
      respect to mole numbers of components in mixtures.
    """
    pass


class Flash2pPTEos(Flash2pSolverPTEos, State2pPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation
  of state (PTEos) that can be used to perform two-phase flash
  calculations. It must have the following attributes:
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


class FlashNpPTEos(FlashNpSolverPTEos, StateNpPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation
  of state (PTEos) that can be used to perform multiphase flash
  calculations. It must have the following attributes:
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
  perform two-phase flash calculations:
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

  And multiphase flash calculations:
  +-------------------+------------------------------------------------+
  | Method            | Result                                         |
  +===================+================================================+
  | getPT_lnphiji     | Natural logarithms of fugacity coefficients of |
  |                   | components for each mixture as a matrix of     |
  |                   | shape `(Np, Nc)`.                              |
  +-------------------+------------------------------------------------+
  | getPT_lnphiji_dnj | Previous + a tensor of shape `(Np, Nc, Nc)` of |
  |                   | their partial derivatives with respect to mole |
  |                   | numbers of components.                         |
  +-------------------+------------------------------------------------+
  | getPT_PIDj        | A vector of shape `(Np,)` of phase designation |
  |                   | indices. `Np` is the number of phases.         |
  +-------------------+------------------------------------------------+
  | getPT_Zj          | A vector of shape `(Np,)` of compressibility   |
  |                   | factors.                                       |
  +-------------------+------------------------------------------------+
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


class Flash2pSolver[Eos](Protocol):
  """A protocol for callable objects that can solve two-phase flash
  problems.

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
  - mole fraction of the non-reference phase,
  - a `Vector[Float]` of shape `(Nc,)` of k-values of components,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the non-reference phase,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the reference phase.
  """
  def __call__(
    self,
    eos: Eos,
    P1: float,
    P2: float,
    yi: Vector[Float],
    kvi0: Vector[Float],
    /,
  ) -> tuple[float, Vector[Float], Vector[Float], Vector[Float]]:
    pass


class FlashNpSolver[Eos](Protocol):
  """A protocol for callable objects that can solve multiphase flash
  problems.

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

  fj: Vector[Float], shape (Np - 1,)
    An initial guess of mole fractions of non-reference phases.

  kvji: Matrix[Float], shape (Np - 1, Nc)
    An initial guess of k-values of `Nc` components in non-reference
    phases.

  Returns
  -------
  A tuple containing:
  - a `Vector[Float]` of shape `(Np - 1,)` of mole fractions of non-
    reference phases,
  - a `Matrix[Float]` of shape `(Np - 1, Nc)` of k-values of components
    in non-reference phases,
  - a `Matrix[Float]` of shape `(Np - 1, Nc)` of mole fractions of
    components in non-reference phases,
  - a `Vector[Float]` of shape `(Nc,)` of mole fractions of components
    in the reference phase.
  """
  def __call__(
    self,
    eos: Eos,
    P1: float,
    P2: float,
    yi: Vector[Float],
    fj: Vector[Float],
    kvji: Matrix[Float],
    /,
  ) -> tuple[Vector[Float], Matrix[Float], Matrix[Float], Vector[Float]]:
    pass


class FlashRoutine[Eos](Protocol):
  """A protocol for callable objects that can be used to perform a flash
  routine.

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

  n: float
    The mole number of a mixture [mol].

  init: Iterable[Vector[Float]] | State | None
    This parameter is used to initialize the flash calculations. The
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
    |                         | which can be used to initialize flash  |
    |                         | calculations. In addition to these     |
    |                         | k-values, initial guesses are also     |
    |                         | obtained from the instance of an EOS.  |
    +-------------------------+----------------------------------------+
    | None                    | The `eos` is used to prepare initial   |
    |                         | guesses of k-values.                   |
    +-------------------------+----------------------------------------+

  Returns
  -------
  Flash calculation results as an instance of `State`.
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
