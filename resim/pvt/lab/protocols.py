from typing import (
  Protocol,
)

from resim.pvt.flash import (
  FlashNpPTEos,
)

from resim.pvt.psat import (
  PsatPTEos,
)


class CvdPTEos(PsatPTEos, FlashNpPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to perform the multiphase constant
  volume depletion experiment. It must have the following attributes:
  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | form      | Literal['PT'] | A formalism of the EOS.                |
  +-----------+---------------+----------------------------------------+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components.              |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molar weights [kg/mol] of components   |
  |           |               | as a vector of shape `(Nc,)`.          |
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
  | getPT_lnphiji       | Natural logarithms of fugacity coefficients  |
  |                     | of components for each mixture as a matrix   |
  |                     | of shape `(Np, Nc)`.                         |
  +---------------------+----------------------------------------------+
  | getPT_lnphiji_dnj   | Previous + a tensor of shape `(Np, Nc, Nc)`  |
  |                     | of their partial derivatives with respect to |
  |                     | mole numbers of components.                  |
  +---------------------+----------------------------------------------+
  | getPT_PIDj          | A vector of shape `(Np,)` of phase           |
  |                     | designation indices. `Np` is the number of   |
  |                     | phases.                                      |
  +---------------------+----------------------------------------------+
  | getPT_Zj            | A vector of shape `(Np,)` of compressibility |
  |                     | factors.                                     |
  +---------------------+----------------------------------------------+
  """
  pass


class CcePTEos(FlashNpPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to perform the multiphase constant
  composition expansion experiment. It must have the following
  attributes:
  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | form      | Literal['PT'] | A formalism of the EOS.                |
  +-----------+---------------+----------------------------------------+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components.              |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molar weights [kg/mol] of components   |
  |           |               | as a vector of shape `(Nc,)`.          |
  +-----------+---------------+----------------------------------------+

  Any class that implements this protocol must also have methods:
  +-------------------+------------------------------------------------+
  | Method            | Result                                         |
  +===================+================================================+
  | getPT_kvguess     | A sequence of initial guesses of k-values.     |
  +-------------------+------------------------------------------------+
  | getPT_lnphii      | A vector of shape `(Nc,)` of logarithms of     |
  |                   | fugacity coefficients of components.           |
  +-------------------+------------------------------------------------+
  | getPT_lnphii_dnj  | Previous + a matrix of shape `(Nc, Nc)` of     |
  |                   | their partial derivatives with respect to mole |
  |                   | numbers of components.                         |
  +-------------------+------------------------------------------------+
  | getPT_PID         | The phase designation index of a mixture.      |
  +-------------------+------------------------------------------------+
  | getPT_Z           | The compressibility factor of a mixture.       |
  +-------------------+------------------------------------------------+
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
  pass


class DlePTEos(CvdPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to perform the multiphase differential
  liberation experiment. It must have the following attributes:
  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | form      | Literal['PT'] | A formalism of the EOS.                |
  +-----------+---------------+----------------------------------------+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components.              |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molar weights [kg/mol] of components   |
  |           |               | as a vector of shape `(Nc,)`.          |
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
  | getPT_lnphiji       | Natural logarithms of fugacity coefficients  |
  |                     | of components for each mixture as a matrix   |
  |                     | of shape `(Np, Nc)`.                         |
  +---------------------+----------------------------------------------+
  | getPT_lnphiji_dnj   | Previous + a tensor of shape `(Np, Nc, Nc)`  |
  |                     | of their partial derivatives with respect to |
  |                     | mole numbers of components.                  |
  +---------------------+----------------------------------------------+
  | getPT_PIDj          | A vector of shape `(Np,)` of phase           |
  |                     | designation indices. `Np` is the number of   |
  |                     | phases.                                      |
  +---------------------+----------------------------------------------+
  | getPT_Zj            | A vector of shape `(Np,)` of compressibility |
  |                     | factors.                                     |
  +---------------------+----------------------------------------------+
  """
  pass


class SwlPTEos(PsatPTEos, Protocol):
  """A protocol for an initialized instance of a PT-based equation of
  state (PTEos) that can be used to perform the two-phase swelling
  experiment. It must have the following attributes:
  +-----------+---------------+----------------------------------------+
  | Attribute | Type          | Description                            |
  +===========+===============+========================================+
  | form      | Literal['PT'] | A formalism of the EOS.                |
  +-----------+---------------+----------------------------------------+
  | name      | str           | The name of an EOS (for logging).      |
  +-----------+---------------+----------------------------------------+
  | Nc        | int           | The number of components.              |
  +-----------+---------------+----------------------------------------+
  | mwi       | Vector[Float] | Molar weights [kg/mol] of components   |
  |           |               | as a vector of shape `(Nc,)`.          |
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
  pass
