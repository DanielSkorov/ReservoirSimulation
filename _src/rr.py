import numpy as np
import numpy.typing as npt

from functools import (
  partial,
)

from typing import (
  Callable,
)


def fD(
  a: np.float64,
  yi: npt.NDArray[np.float64],
  di: npt.NDArray[np.float64],
  yidi: npt.NDArray[np.float64],
  y0: np.float64,
  yN: np.float64,
) -> tuple[np.float64, np.float64]:
  denom = 1. / (di * (a + 1.) + a)
  return (
    y0 + a * yi.dot(denom) - yN * a,
    yidi.dot(denom * denom) - yN,
  )


def fG(
  a: np.float64,
  yi: npt.NDArray[np.float64],
  di: npt.NDArray[np.float64],
  y0: np.float64,
  yN: np.float64,
) -> tuple[np.float64, np.float64]:
  denom = 1. / (di * (a + 1.) + a)
  return (
    (a + 1.) * (y0 / a + yi.dot(denom) - yN),
    -y0 / (a * a) - yi.dot(denom * denom) - yN,
  )


def fH(
  a: np.float64,
  yi: npt.NDArray[np.float64],
  di: npt.NDArray[np.float64],
  y0: np.float64,
  yN: np.float64,
) -> tuple[np.float64, np.float64]:
  denom = 1. / (di * (a + 1.) + a)
  G = (1. + a) * (y0 / a + yi.dot(denom) - yN)
  dGda = -y0 / (a * a) - yi.dot(denom * denom) - yN
  return -a * G, -G - a * dGda


def solve2p_FGH(
  kvi: npt.NDArray[np.float64],
  yi: npt.NDArray[np.float64],
  tol: np.float64 = np.float64(1e-10),
  Niter: int = 50,
) -> np.float64:
  """
  Solves the Rachford-rice equation for two-phase systems using
  the FGH-method. For the details see 10.1016/j.fluid.2017.08.020.

  Arguments:
    kvi: k-values of components. Must be a numpy.ndarray[numpy.float64]
      with the shape `(Nc,)`.
    yi: mole fractions of components. Must be a numpy.ndarray[numpy.float64]
      with the shape `(Nc,)`.
    tol: tolerance. Must be a scalar of numpy.float64.
    Niter: maximum number of iterations. Must be a scalar of integer.

  Returns:
    Mole fraction of the non-reference phase in a system.
  """
  idx = kvi.argsort()[::-1]
  yi = yi[idx]
  kvi = kvi[idx]
  ci = 1. / (1. - kvi)
  di = (ci[0] - ci[1:-1]) / (ci[-1] - ci[0])
  y0 = yi[0]
  yN = yi[-1]
  yi = yi[1:-1]
  pD = partial(fD, yi=yi, di=di, yidi=yi*di, y0=y0, yN=yN)
  a = y0 / yN
  D, dDda = pD(a)
  print(f'0: {a = }, {D = }')
  pcondit = partial(_solve2p_FGH_condit, tol=tol, Niter=Niter)
  pupdate = partial(_solve2p_FGH_update, pD=pD)
  carry = (1, a, D / dDda, D)
  while pcondit(carry):
    carry = pupdate(carry)
  a = carry[1]
  return (ci[0] + a * ci[-1]) / (1. + a)


def _solve2p_FGH_condit(
  carry: tuple[int, np.float64, np.float64, np.float64],
  tol: np.float64,
  Niter: int,
) -> bool:
  i, a, _, D = carry
  return (i < Niter) & (np.abs(D) > tol)


def _solve2p_FGH_update(
  carry: tuple[int, np.float64, np.float64, np.float64],
  pD: Callable[[np.float64], tuple[np.float64, np.float64]],
) -> tuple[int, np.float64, np.float64, np.float64]:
  i, a_, h_, D_ = carry
  a = a_ - h_
  print(f'{i}: {a = }')
  if a < 0.:
    if D_ > 0.:
      a += h_ * h_ / (h_ - a_ * (a_ + 1.))
    else:
      a += h_ * h_ / (h_ + a_ + 1.)
  D, dDda = pD(a)
  print(f'{i}: {a = }, {D = }')
  return i + 1, a, D / dDda, D



def solve2p_GH(
  kvi: npt.NDArray[np.float64],
  yi: npt.NDArray[np.float64],
  tol: np.float64 = 1e-10,
  Niter: int = 50,
) -> np.float64:
  """
  Solves the Rachford-rice equation for two-phase systems using
  the GH-method. For the details see 10.1016/j.fluid.2017.08.020.

  Arguments:
    kvi: k-values of components. Must be a numpy.ndarray[numpy.float64]
      with the shape `(Nc,)`.
    yi: mole fractions of components. Must be a numpy.ndarray[numpy.float64]
      with the shape `(Nc,)`.
    tol: tolerance. Must be a scalar of numpy.float64.
    Niter: maximum number of iterations. Must be a scalar of integer.

  Returns:
    Mole fraction of the non-reference phase in a system.
  """
  idx = kvi.argsort()[::-1]
  yi = yi[idx]
  kvi = kvi[idx]
  ci = 1. / (1. - kvi)
  di = (ci[0] - ci[1:-1]) / (ci[-1] - ci[0])
  y0 = yi[0]
  yN = yi[-1]
  yi = yi[1:-1]
  a = y0 / yN
  denom = 1. / (di * (a + 1.) + a)
  G = (a + 1.) * (y0 / a + yi.dot(denom) - yN)
  dGda = -y0 / (a * a) - yi.dot(denom * denom) - yN
  if G > 0.:
    pG = partial(fG, yi=yi, di=di, y0=y0, yN=yN)
    da = -G / dGda
    carry = (1, a, da, G)
    pupdate = partial(_solve2p_GH_update, pF=pG)
  else:
    pH = partial(fH, yi=yi, di=di, y0=y0, yN=yN)
    H = -a * G
    dHda = -G - a * dGda
    da = -H / dHda
    carry = (1, a, da, H)
    pupdate = partial(_solve2p_GH_update, pF=pH)
  pcondit = partial(_solve2p_FGH_condit, tol=tol, Niter=Niter)
  while pcondit(carry):
    carry = pupdate(carry)
  a = carry[1]
  return (ci[0] + a * ci[-1]) / (1. + a)

def _solve2p_GH_update(
  carry: tuple[int, np.float64, np.float64, np.float64],
  pF: Callable[[np.float64], tuple[np.float64, np.float64]],
) -> tuple[int, np.float64, np.float64, np.float64]:
  i, a_, da_, _ = carry
  a = a_ + da_
  eq, grad = pF(a)
  da = -eq / grad
  return i + 1, a, da, eq

