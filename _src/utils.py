import numpy as np

from customtypes import (
  Double,
  Vector,
  Matrix,
  Linsolver,
)


def pyrqi(
  Q: Matrix[Double],
  x0: Vector[Double] | None = None,
  lmbd0: float | None = None,
  tol: float = 1e-14,
  maxiter: int = 20,
  linsolver: Linsolver = np.linalg.solve,
) -> tuple[Vector[Double], float]:
  """
  """
  if x0 is not None:
    xk = x0
  else:
    xk = np.ones(shape=(Q.shape[0],))
  if lmbd0 is None:
    lmbdk = xk.dot(Q).dot(xk)
  else:
    lmbdk = lmbd0
  M: Matrix[Double] = Q - np.diagflat(np.full_like(xk, lmbdk))
  xkp1 = linsolver(M, xk)
  xkp1 /= np.linalg.norm(xkp1)
  lmbdkp1 = xkp1.dot(Q).dot(xkp1)
  k = 1
  while (np.abs((lmbdkp1 - lmbdk) / lmbdk) > tol) & (k < maxiter):
    xk = xkp1
    lmbdk = lmbdkp1
    M = Q - np.diagflat(np.full_like(xk, lmbdk))
    xkp1 = linsolver(M, xk)
    xkp1 /= np.linalg.norm(xkp1)
    lmbdkp1 = xkp1.dot(Q).dot(xkp1)
    k += 1
  return xkp1, lmbdkp1
