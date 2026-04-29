! python -m numpy.f2py -c -m futils futils.f90 only: lusolver : --quiet --build-dir "./build/" --backend=meson

module types
  implicit none
  integer, parameter :: dp = kind(0.d0)
  ! integer, parameter :: dp = selected_real_kind(15, 307)
end module types


module constants
  use types, only: dp

  implicit none

  real(dp), parameter :: RGAS = 8.31446261815324_dp
end module constants


module math
  use types, only: dp

  implicit none

  contains

  function cbrt(x) result(r)
    ! ------------------------------------------------------------------------ !
    ! Calculate the cubic root of a double precision number.
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | x          | real(8) | in     | A double precision number.           |
    ! +------------+---------+--------+--------------------------------------+
    !
    ! Based on the benchmarks, this function is a little bit faster than the
    ! use of `**` operator.
    ! ------------------------------------------------------------------------ !
    real(dp), intent(in) :: x
    real(dp) :: r
    if (x > 0._dp) then
      r = exp(log(x) / 3._dp)
    else if (x < 0._dp) then
      r = -exp(log(-x) / 3._dp)
    else
      r = 0._dp
    end if
  end function cbrt
end module math


module linalg
  use types, only: dp

  implicit none

  contains

  subroutine gem(n, A, b, x, singular)
    ! ------------------------------------------------------------------------ !
    ! Solve a linear system using the naive implementation of the Gaussian
    ! Elimination Method with partial pivoting.
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | n          | integer | in     | The size of input arrays.            |
    ! +------------+---------+--------+--------------------------------------+
    ! | A(n, n)    | real(8) | inout  | A coefficient matrix.                |
    ! +------------+---------+--------+--------------------------------------+
    ! | b(n)       | real(8) | inout  | Ordinate or "dependent variable"     |
    ! |            |         |        | values.                              |
    ! +------------+---------+--------+--------------------------------------+
    ! | x(n)       | real(8) | out    | The solution of a linear system.     |
    ! +------------+---------+--------+--------------------------------------+
    ! | singular   | logical | out    | A boolean flag indicating if the     |
    ! |            |         |        | matrix A is singular.                |
    ! +------------+---------+--------+--------------------------------------+
    !
    ! No copies of input arrays are made, meaning that the matrix A and vector
    ! b will be modified in place.
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: n
    ! f2py real(8), intent(in, out) :: A(n, n), b(n)
    ! f2py real(8), intent(out) :: x(n)
    ! f2py logical, intent(out) :: singular
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: n
    real(dp), intent(inout) :: A(n, n), b(n)
    real(dp), intent(out) :: x(n)
    logical, intent(out) :: singular
    integer :: j, k, prow
    real(dp) :: buf
    real(dp), parameter :: eps = 1.e-13_dp
    singular = .false.
    do k = 1, n - 1
      prow = maxloc(abs(A(k:n, k)), 1) + k - 1
      if (abs(A(prow, k)) <= eps) then
        singular = .true.
        return
      end if
      if (prow /= k) then
        do j = k, n
          buf = A(prow, j)
          A(prow, j) = A(k, j)
          A(k, j) = buf
        end do
        buf = b(prow)
        b(prow) = b(k)
        b(k) = buf
      end if
      A(k+1:n, k) = A(k+1:n, k) / A(k, k)
      do j = k + 1, n
        A(k+1:n, j) = A(k+1:n, j) - A(k, j) * A(k+1:n, k)
      end do
      b(k+1:n) = b(k+1:n) - A(k+1:n, k) * b(k)
    end do
    do k = n, 1, -1
      x(k) = (b(k) - dot_product(A(k, k+1:n), x(k+1:n))) / A(k, k)
    end do
  end subroutine gem

  subroutine rqi(n, Q, x, lmbd, singular, maxiter, tol)
    ! ------------------------------------------------------------------------ !
    ! Find an eigenvalue and corresponding eigenvector using the Rayleigh
    ! quotient iteration algorithm.
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | n          | integer | in     | Size of input arrays.                |
    ! +------------+---------+--------+--------------------------------------+
    ! | Q(n, n)    | real(8) | in     | A symmetric matrix.                  |
    ! +------------+---------+--------+--------------------------------------+
    ! | x(n)       | real(8) | inout  | A vector representing an initial     |
    ! |            |         |        | guess of an eigenvector and used to  |
    ! |            |         |        | store a found solution.              |
    ! +------------+---------+--------+--------------------------------------+
    ! | lmbd       | real(8) | inout  | An initial guess of an eigenvalue.   |
    ! +------------+---------+--------+--------------------------------------+
    ! | singular   | logical | out    | A boolean flag indicating if the     |
    ! |            |         |        | matrix Q is singular.                |
    ! +------------+---------+--------+--------------------------------------+
    ! | maxiter    | integer | in     | The maximum number of iterations.    |
    ! +------------+---------+--------+--------------------------------------+
    ! | tol        | real(8) | in     | Terminate the algorithm successfully |
    ! |            |         |        | if an absolute relative change in    |
    ! |            |         |        | eigenvalue is less than tol.         |
    ! +------------+---------+--------+--------------------------------------+
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: n
    ! f2py real(8), intent(in) :: Q(n, n), tol
    ! f2py integer, intent(in) :: maxiter
    ! f2py real(8), intent(in, out) :: x(n), lmbd
    ! f2py logical, intent(out) :: singular
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: n, maxiter
    real(dp), intent(in) :: tol, Q(n, n)
    real(dp), intent(inout) :: x(n)
    real(dp), intent(inout) :: lmbd
    logical, intent(out) :: singular
    integer :: i, k
    real(dp) :: lmbdkp1, dlmbd, M(n, n), xkp1(n)
    k = 0
    xkp1 = matmul(Q, x) - lmbd * x
    if (sqrt(dot_product(xkp1, xkp1)) < tol) then
      return
    end if
    singular = .false.
    M = Q
    do i = 1, n
      M(i, i) = Q(i, i) - lmbd
    end do
    call gem(n, M, x, xkp1, singular)
    if (singular) then
      return
    end if
    xkp1 = xkp1 / sqrt(dot_product(xkp1, xkp1))
    lmbdkp1 = dot_product(matmul(Q, xkp1), xkp1)
    dlmbd = abs((lmbdkp1 - lmbd) / lmbd)
    do while ((dlmbd > tol) .and. (k < maxiter))
      x = xkp1
      lmbd = lmbdkp1
      M = Q
      do i = 1, n
        M(i, i) = Q(i, i) - lmbd
      end do
      call gem(n, M, x, xkp1, singular)
      if (singular) then
        return
      end if
      xkp1 = xkp1 / sqrt(dot_product(xkp1, xkp1))
      lmbdkp1 = dot_product(matmul(Q, xkp1), xkp1)
      dlmbd = abs((lmbdkp1 - lmbd) / lmbd)
      k = k + 1
    end do
    lmbd = lmbdkp1
    x = xkp1
  end subroutine rqi

  subroutine plu(n, A, p, singular)
    ! ------------------------------------------------------------------------ !
    ! Perform the naive LU-decomposition with partial pivoting.
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | n          | integer | in     | The size of input arrays.            |
    ! +------------+---------+--------+--------------------------------------+
    ! | A(n, n)    | real(8) | inout  | A coefficient matrix.                |
    ! +------------+---------+--------+--------------------------------------+
    ! | p(n)       | integer | out    | Permuted rows of the input matrix.   |
    ! +------------+---------+--------+--------------------------------------+
    ! | singular   | logical | out    | A boolean flag indicating if the     |
    ! |            |         |        | matrix A is singular.                |
    ! +------------+---------+--------+--------------------------------------+
    !
    ! No copies of the input matrix are made, meaning that it will be
    ! overwritten by the matrices L and U.
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: n
    ! f2py real(8), intent(in, out) :: A(n, n)
    ! f2py integer, intent(out) :: p(n)
    ! f2py logical, intent(out) :: singular
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: n
    real(dp), intent(inout) :: A(n, n)
    integer, intent(out) :: p(n)
    logical, intent(out) :: singular
    integer :: j, k, prow
    real(dp) :: buf
    real(dp), parameter :: eps = 1.e-13_dp
    singular = .false.
    do k = 1, n
      p(k) = k
    end do
    do k = 1, n
      prow = maxloc(abs(A(k:n, k)), 1) + k - 1
      if (abs(A(prow, k)) <= eps) then
        singular = .true.
        return
      end if
      if (prow /= k) then
        do j = 1, n
          buf = A(prow, j)
          A(prow, j) = A(k, j)
          A(k, j) = buf
        end do
        j = p(k)
        p(k) = p(prow)
        p(prow) = j
      end if
      A(k+1:n, k) = A(k+1:n, k) / A(k, k)
      do j = k + 1, n
        A(k+1:n, j) = A(k+1:n, j) - A(k+1:n, k) * A(k, j)
      end do
    end do
  end subroutine plu

  subroutine plubksb(n, A, b, p, x)
    ! ------------------------------------------------------------------------ !
    ! Solve a triangular linear system using back substitution.
    ! Implements back substitution to solve Ax = b 
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | n          | integer | in     | The size of input arrays.            |
    ! +------------+---------+--------+--------------------------------------+
    ! | A(n, n)    | real(8) | in     | A coefficient matrix.                |
    ! +------------+---------+--------+--------------------------------------+
    ! | b(n)       | real(8) | inout  | Ordinate or "dependent variable"     |
    ! |            |         |        | values.                              |
    ! +------------+---------+--------+--------------------------------------+
    ! | p(n)       | integer | in     | Permuted rows of the input matrix.   |
    ! +------------+---------+--------+--------------------------------------+
    ! | x(n)       | real(8) | out    | The solution of a linear system.     |
    ! +------------+---------+--------+--------------------------------------+
    !
    ! The algorithm uses the results of the LU decomposition that can be
    ! obtained from the plu subroutine.
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: n
    ! f2py integer, intent(in) :: p(n)
    ! f2py real(8), intent(in) :: A(n, n)
    ! f2py real(8), intent(in, out) :: b(n)
    ! f2py real(8), intent(out) :: x(n)
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: n
    integer, intent(in) :: p(n)
    real(dp), intent(in) :: A(n, n)
    real(dp), intent(inout) :: b(n)
    real(dp), intent(out) :: x(n)
    integer :: i
    b = b(p)
    do i = 1, n
      x(i) = b(i) - dot_product(A(i, 1:i-1), x(1:i-1))
    end do
    do i = n, 1, -1
      x(i) = (x(i) - dot_product(A(i, i+1:n), x(i+1:n))) / A(i, i)
    end do
  end subroutine plubksb

  subroutine lusolver(n, A, b, x, singular)
    ! ------------------------------------------------------------------------ !
    ! Solve a linear matrix equation.
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | n          | integer | in     | The size of input arrays.            |
    ! +------------+---------+--------+--------------------------------------+
    ! | A(n, n)    | real(8) | inout  | A coefficient matrix.                |
    ! +------------+---------+--------+--------------------------------------+
    ! | b(n)       | real(8) | inout  | Ordinate or "dependent variable"     |
    ! |            |         |        | values.                              |
    ! +------------+---------+--------+--------------------------------------+
    ! | x(n)       | real(8) | out    | The solution of a linear system.     |
    ! +------------+---------+--------+--------------------------------------+
    ! | singular   | logical | out    | A boolean flag indicating if the     |
    ! |            |         |        | matrix A is singular.                |
    ! +------------+---------+--------+--------------------------------------+
    !
    ! Uses the LU-decomposition to solver a linear matrix equation. No copies
    ! of input arrays are made, meaning that they will be updated.
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: n
    ! f2py real(8), intent(in, out) :: A(n, n)
    ! f2py real(8), intent(in, out) :: b(n)
    ! f2py real(8), intent(out) :: x(n)
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: n
    real(dp), intent(inout) :: A(n, n)
    real(dp), intent(inout) :: b(n)
    real(dp), intent(out) :: x(n)
    logical, intent(out) :: singular
    integer :: p(n)
    call plu(n, A, p, singular)
    if (.not. singular) then
      call plubksb(n, A, b, p, x)
    end if
  end subroutine lusolver

  ! subroutine chol()

  ! end subroutine chol

  ! subroutine mchol()

  ! end subroutine mchol
end module linalg


module abstract_mixing_rule
  use types, only: dp

  implicit none

  type, abstract :: typemr
    real(dp), dimension(:), allocatable :: kp, rsqrtc, sqra, b, vsb
    real(dp), dimension(:, :), allocatable :: d
    contains
    procedure(protocol_init), deferred :: mr_init
    procedure(protocol_z), deferred :: mr_z
    procedure(protocol_z_dt), deferred :: mr_z_dt
    procedure(protocol_z_dt_dt2), deferred :: mr_z_dt_dt2
    procedure(protocol_lnphi), deferred :: mr_lnphi
    procedure(protocol_lnphi_dt), deferred :: mr_lnphi_dT
    procedure(protocol_lnphi_dy), deferred :: mr_lnphi_dy
  end type

  abstract interface
    subroutine protocol_init(self, nc0, kp0, rsqrtc0, sqra0, b0, vsb0, d0)
      import dp, typemr
      class(typemr), intent(inout) :: self
      integer, intent(in) :: nc0
      real(dp), intent(in) :: kp0(nc0), rsqrtc0(nc0), sqra0(nc0), b0(nc0), &
                              vsb0(nc0), d0(nc0, nc0)
    end subroutine protocol_init

    subroutine protocol_z(self, nc, tem, y, am, bm, vs)
      import dp, typemr
      class(typemr), intent(in) :: self
      integer, intent(in) :: nc
      real(dp), intent(in) :: tem, y(nc)
      real(dp), intent(out) :: am, bm, vs
    end subroutine protocol_z

    subroutine protocol_z_dt(self, nc, tem, y, am, bm, vs, damdt)
      import dp, typemr
      class(typemr), intent(in) :: self
      integer, intent(in) :: nc
      real(dp), intent(in) :: tem, y(nc)
      real(dp), intent(out) :: am, bm, vs, damdt
    end subroutine protocol_z_dt

    subroutine protocol_z_dt_dt2(self, nc, tem, y, am, bm, vs, damdt, d2amdt2)
      import dp, typemr
      class(typemr), intent(in) :: self
      integer, intent(in) :: nc
      real(dp), intent(in) :: tem, y(nc)
      real(dp), intent(out) :: am, bm, vs, damdt, d2amdt2
    end subroutine protocol_z_dt_dt2

    subroutine protocol_lnphi(self, nc, tem, y, am, bm, lnphis, damdy, dbmdy, &
                              s1, s2)
      import dp, typemr
      class(typemr), intent(in) :: self
      integer, intent(in) :: nc
      real(dp), intent(in) :: tem, y(nc)
      real(dp), intent(out) :: am, bm, lnphis(nc), damdy(nc), dbmdy(nc), &
                               s1, s2
    end subroutine protocol_lnphi

    subroutine protocol_lnphi_dt(self, nc, tem, y, am, bm, lnphis, damdy, &
                                 dbmdy, s1, s2, damdt, d2amdydt, ds2dt)
      import dp, typemr
      class(typemr), intent(in) :: self
      integer, intent(in) :: nc
      real(dp), intent(in) :: tem, y(nc)
      real(dp), intent(out) :: am, bm, lnphis(nc), damdy(nc), dbmdy(nc), &
                               s1, s2, damdt, d2amdydt(nc), ds2dt
    end subroutine protocol_lnphi_dt

    subroutine protocol_lnphi_dy(self, nc, tem, y, am, bm, lnphis, damdy, &
                                 dbmdy, s1, s2, d2amdy2, d2bmdy2, ds1dy, ds2dy)
      import dp, typemr
      class(typemr), intent(in) :: self
      integer, intent(in) :: nc
      real(dp), intent(in) :: tem, y(nc)
      real(dp), intent(out) :: am, bm, lnphis(nc), damdy(nc), dbmdy(nc), &
                               s1, s2, d2amdy2(nc, nc), d2bmdy2(nc, nc), &
                               ds1dy(nc), ds2dy(nc)
    end subroutine protocol_lnphi_dy
  end interface
end module abstract_mixing_rule


module vdw_mixing_rule
  use types, only: dp
  use abstract_mixing_rule, only: typemr

  implicit none

  type, extends(typemr) :: vdwmr
    contains
    procedure :: mr_init => vdwmr_init
    procedure :: mr_z => vdwmr_z
    procedure :: mr_z_dt => vdwmr_z_dt
    procedure :: mr_z_dt_dt2 => vdwmr_z_dt_dt2
    procedure :: mr_lnphi => vdwmr_lnphi
    procedure :: mr_lnphi_dt => vdwmr_lnphi_dt
    procedure :: mr_lnphi_dy => vdwmr_lnphi_dy
  end type

  contains

  subroutine vdwmr_init(self, nc0, kp0, rsqrtc0, sqra0, b0, vsb0, d0)
    class(vdwmr), intent(inout) :: self
    integer, intent(in) :: nc0
    real(dp), intent(in) :: kp0(nc0), rsqrtc0(nc0), sqra0(nc0), b0(nc0), &
                            vsb0(nc0), d0(nc0, nc0)
    if (allocated(self%kp)) deallocate(self%kp)
    if (allocated(self%rsqrtc)) deallocate(self%rsqrtc)
    if (allocated(self%sqra)) deallocate(self%sqra)
    if (allocated(self%b)) deallocate(self%b)
    if (allocated(self%vsb)) deallocate(self%vsb)
    if (allocated(self%d)) deallocate(self%d)
    allocate(self%kp(1:nc0))
    allocate(self%rsqrtc(1:nc0))
    allocate(self%sqra(1:nc0))
    allocate(self%b(1:nc0))
    allocate(self%vsb(1:nc0))
    allocate(self%d(1:nc0, 1:nc0))
    self%kp = kp0
    self%rsqrtc = rsqrtc0
    self%sqra = sqra0
    self%b = b0
    self%vsb = vsb0
    self%d = d0
  end subroutine vdwmr_init

  subroutine vdwmr_z(self, nc, tem, y, am, bm, vs)
    class(vdwmr), intent(in) :: self
    integer, intent(in) :: nc
    real(dp), intent(in) :: tem, y(nc)
    real(dp), intent(out) :: am, bm, vs
    integer :: i, j
    real(dp) :: sqrtem, u(nc), uj
    sqrtem = sqrt(tem)
    do i = 1, nc
      u(i) = y(i) * self%sqra(i) * ((1._dp - sqrtem * self%rsqrtc(i)) &
                                    * self%kp(i) + 1._dp)
    end do
    am = 0._dp
    do j = 1, nc
      uj = u(j)
      am = am + uj * uj
      do i = j + 1, nc
        am = am + 2._dp * uj * self%d(i, j) * u(i)
      end do
    end do
    bm = dot_product(y, self%b)
    vs = dot_product(y, self%vsb)
  end subroutine vdwmr_z

  subroutine vdwmr_z_dt(self, nc, tem, y, am, bm, vs, damdt)
    class(vdwmr), intent(in) :: self
    integer, intent(in) :: nc
    real(dp), intent(in) :: tem, y(nc)
    real(dp), intent(out) :: am, bm, vs, damdt
    integer :: i, j
    real(dp) :: sqrtem, invsqrtem, u(nc), dudt(nc), uj, dudtj, ui, dudti, dij
    sqrtem = sqrt(tem)
    invsqrtem = -.5_dp / sqrtem
    do i = 1, nc
      u(i) = y(i) * self%sqra(i) * ((1._dp - sqrtem * self%rsqrtc(i)) &
                                    * self%kp(i) + 1._dp)
      dudt(i) = invsqrtem * y(i) * self%sqra(i) * self%kp(i) * self%rsqrtc(i)
    end do
    am = 0._dp
    do j = 1, nc
      uj = u(j)
      dudtj = dudt(j)
      am = am + uj * uj
      damdt = damdt + 2._dp * uj * dudtj
      do i = j + 1, nc
        ui = u(i)
        dudti = dudt(i)
        dij = self%d(i, j)
        am = am + 2._dp * ui * uj * dij
        damdt = damdt + 2._dp * dij * (ui * dudtj + uj * dudti)
      end do
    end do
    bm = dot_product(y, self%b)
    vs = dot_product(y, self%vsb)
  end subroutine vdwmr_z_dt

  subroutine vdwmr_z_dt_dt2(self, nc, tem, y, am, bm, vs, damdt, d2amdt2)
    class(vdwmr), intent(in) :: self
    integer, intent(in) :: nc
    real(dp), intent(in) :: tem, y(nc)
    real(dp), intent(out) :: am, bm, vs, damdt, d2amdt2
    integer :: i, j
    real(dp) :: sqrtem, invtem, invsqrtem, u(nc), dudt(nc), &
                uj, dudtj, d2udt2j, ui, dudti, d2udt2i, dij
    sqrtem = sqrt(tem)
    invtem = -.5_dp / tem
    invsqrtem = -.5_dp / sqrtem
    do i = 1, nc
      u(i) = y(i) * self%sqra(i) * ((1._dp - sqrtem * self%rsqrtc(i)) &
                                    * self%kp(i) + 1._dp)
      dudt(i) = invsqrtem * y(i) * self%sqra(i) * self%kp(i) * self%rsqrtc(i)
    end do
    am = 0._dp
    do j = 1, nc
      uj = u(j)
      dudtj = dudt(j)
      d2udt2j = invtem * dudtj
      am = am + uj * uj
      damdt = damdt + 2._dp * uj * dudtj
      d2amdt2 = d2amdt2 + 2._dp * (dudtj * dudtj + uj * d2udt2j)
      do i = j + 1, nc
        ui = u(i)
        dudti = dudt(i)
        d2udt2i = invtem * dudti
        dij = self%d(i, j)
        am = am + 2._dp * uj * dij * ui
        damdt = damdt + 2._dp * dij * (ui * dudtj + uj * dudti)
        d2amdt2 = d2amdt2 + 2._dp * dij * (2._dp * dudti * dudtj &
                                           + ui * d2udt2j + uj * d2udt2i)
      end do
    end do
    bm = dot_product(y, self%b)
    vs = dot_product(y, self%vsb)
  end subroutine vdwmr_z_dt_dt2

  subroutine vdwmr_lnphi(self, nc, tem, y, am, bm, lnphis, damdy, dbmdy, s1, s2)
    class(vdwmr), intent(in) :: self
    integer, intent(in) :: nc
    real(dp), intent(in) :: tem, y(nc)
    real(dp), intent(out) :: am, bm, lnphis(nc), damdy(nc), dbmdy(nc), s1, s2
    integer :: i, j
    real(dp) :: sqrtem, u(nc), yu(nc), yuj, s(nc), sj, dij
    sqrtem = sqrt(tem)
    do i = 1, nc
      u(i) = self%sqra(i) * (self%kp(i) * (1._dp - sqrtem * self%rsqrtc(i)) &
                             + 1._dp)
      yu(i) = y(i) * u(i)
    end do
    s = 0._dp
    do j = 1, nc
      sj = s(j)
      yuj = yu(j)
      do i = j + 1, nc
        dij = self%d(i, j)
        sj = sj + dij * yu(i)
        s(i) = s(i) + dij * yuj
      end do
      s(j) = (sj + yuj) * u(j)
    end do
    am = dot_product(y, s)
    bm = dot_product(y, self%b)
    lnphis = self%vsb
    damdy = 2._dp * s
    dbmdy = self%b
    s1 = 0._dp
    s2 = 0._dp
  end subroutine vdwmr_lnphi

  subroutine vdwmr_lnphi_dt(self, nc, tem, y, am, bm, lnphis, damdy, dbmdy, &
                            s1, s2, damdt, d2amdydt, ds2dt)
    class(vdwmr), intent(in) :: self
    integer, intent(in) :: nc
    real(dp), intent(in) :: tem, y(nc)
    real(dp), intent(out) :: am, bm, lnphis(nc), damdy(nc), dbmdy(nc), &
                             s1, s2, damdt, d2amdydt(nc), ds2dt
    integer :: i, j
    real(dp) :: sqrtem, invsqrtem, u(nc), uj, yu(nc), yuj, dudt(nc), &
                ydudt(nc), ydudtj, dij, s(nc), dsdt(nc), sj, dsdtj
    sqrtem = sqrt(tem)
    invsqrtem = -.5_dp / sqrtem
    do i = 1, nc
      u(i) = self%sqra(i) * (self%kp(i) * (1._dp - sqrtem * self%rsqrtc(i)) &
                             + 1._dp)
      yu(i) = y(i) * u(i)
      dudt(i) = invsqrtem * self%sqra(i) * self%kp(i) * self%rsqrtc(i)
      ydudt(i) = y(i) * dudt(i)
    end do
    s = 0._dp
    dsdt = 0._dp
    do j = 1, nc
      sj = s(j)
      uj = u(j)
      yuj = yu(j)
      ydudtj = ydudt(j)
      dsdtj = dsdt(j)
      do i = j + 1, nc
        dij = self%d(i, j)
        sj = sj + dij * yu(i)
        s(i) = s(i) + dij * yuj
        dsdtj = dsdtj + dij * ydudt(i)
        dsdt(i) = dsdt(i) + dij * ydudtj
      end do
      s(j) = (sj + yuj) * uj
      dsdt(j) = (sj + yuj) * dudt(j) + (dsdtj + ydudtj) * uj
    end do
    am = dot_product(y, s)
    bm = dot_product(y, self%b)
    lnphis = self%vsb
    damdy = 2._dp * s
    dbmdy = self%b
    s1 = 0._dp
    s2 = 0._dp
    damdt = dot_product(y, dsdt)
    d2amdydt = 2._dp * dsdt
    ds2dt = 0._dp
  end subroutine vdwmr_lnphi_dt

  subroutine vdwmr_lnphi_dy(self, nc, tem, y, am, bm, lnphis, damdy, dbmdy, &
                            s1, s2, d2amdy2, d2bmdy2, ds1dy, ds2dy)
    class(vdwmr), intent(in) :: self
    integer, intent(in) :: nc
    real(dp), intent(in) :: tem, y(nc)
    real(dp), intent(out) :: am, bm, lnphis(nc), damdy(nc), dbmdy(nc), s1, s2, &
                             d2amdy2(nc, nc), d2bmdy2(nc, nc), ds1dy(nc), &
                             ds2dy(nc)
    integer :: i, j
    real(dp) :: sqrtem, u(nc), uj, yu(nc), yuj, s(nc), sj, dij
    sqrtem = sqrt(tem)
    do i = 1, nc
      u(i) = self%sqra(i) * (self%kp(i) * (1._dp - sqrtem * self%rsqrtc(i)) &
                             + 1._dp)
      yu(i) = y(i) * u(i)
    end do
    s = 0._dp
    do j = 1, nc
      sj = s(j)
      uj = u(j)
      yuj = yu(j)
      d2amdy2(j, j) = 2._dp * uj * uj
      do i = j + 1, nc
        dij = self%d(i, j)
        sj = sj + dij * yu(i)
        s(i) = s(i) + dij * yuj
        d2amdy2(i, j) = 2._dp * uj * dij * u(i)
        d2amdy2(j, i) = d2amdy2(i, j)
      end do
      s(j) = (sj + yuj) * uj
    end do
    am = dot_product(y, s)
    bm = dot_product(y, self%b)
    lnphis = self%vsb
    damdy = 2._dp * s
    dbmdy = self%b
    s1 = 0._dp
    s2 = 0._dp
    d2bmdy2 = 0._dp
    ds1dy = 0._dp
    ds2dy = 0._dp
  end subroutine vdwmr_lnphi_dy
end module vdw_mixing_rule


module pr78
  use types, only: dp
  use constants, only: RGAS
  use math, only: cbrt
  use abstract_mixing_rule, only: typemr
  use vdw_mixing_rule, only: vdwmr

  implicit none

  real(dp), parameter :: D1 = 1._dp - sqrt(2._dp)
  real(dp), parameter :: D2 = 1._dp + sqrt(2._dp)
  real(dp), parameter :: INVD1PD2 = .25_dp * sqrt(2._dp)

  class(typemr), allocatable :: mr

  contains

  subroutine init(nc0, kp0, rsqrtc0, sqra0, b0, vsb0, d0)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc0
    ! f2py real(8), intent(in) :: kp0(nc0), rsqrtc0(nc0), sqra0(nc0), &
    !                             b0(nc0), vsb0(nc0), d0(nc0, nc0)
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc0
    real(dp), intent(in) :: kp0(nc0), rsqrtc0(nc0), sqra0(nc0), b0(nc0), &
                            vsb0(nc0), d0(nc0, nc0)
    if (allocated(mr)) deallocate(mr)
    allocate(vdwmr :: mr)
    call mr%mr_init(nc0, kp0, rsqrtc0, sqra0, b0, vsb0, d0)
  end subroutine init

  subroutine getpt_z(nc, prs, tem, y, z)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: z
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: z
    real(dp) :: beta, prt, am, bm, vs
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_z(nc, tem, y, am, bm, vs)
    call solve(am * prt * beta, bm * prt, z)
    z = z - vs * prt
  end subroutine getpt_z

  subroutine getpt_z_dp(nc, prs, tem, y, z, dzdp)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: z, dzdp
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: z, dzdp
    real(dp) :: beta, prt, am, bm, a, b, vs, zs
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_z(nc, tem, y, am, bm, vs)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    dzdp = ((z * (b * (6._dp * b - z + 2._dp) - a) &
             - b * (b * (3._dp * b + 2._dp) - 2._dp * a)) &
            / (prs * (z * (3._dp * z + 2._dp * b - 2._dp) &
                      - b * (3._dp * b + 2._dp) + a)))
    zs = vs * prt
    z = z - zs
    dzdp = dzdp - zs / prs
  end subroutine getpt_z_dp

  subroutine getpt_z_dt(nc, prs, tem, y, z, dzdt)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: z, dzdt
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: z, dzdt
    real(dp) :: beta, prt, am, bm, vs, damdt, a, b, zs
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_z_dt(nc, tem, y, am, bm, vs, damdt)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    dzdt = ((b / tem * (z * (z - 6._dp * b - 2._dp) &
                        + b * (3._dp * b + 2._dp) &
                        - a) &
             + (prt * beta * damdt - 2._dp * a / tem) * (b - z)) &
            / (z * (3._dp * z + 2._dp * b - 2._dp) &
               - b * (3._dp * b + 2._dp) &
               + a))
    zs = vs * prt
    z = z - zs
    dzdt = dzdt + zs / tem
  end subroutine getpt_z_dt

  subroutine getpt_z_dp_dp2(nc, prs, tem, y, z, dzdp, d2zdp2)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: z, dzdp, d2zdp2
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: z, dzdp, d2zdp2
    real(dp) :: beta, prt, am, bm, vs, a, b, dadp, dbdp, dqdz, zs
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_z(nc, tem, y, am, bm, vs)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    dqdz = z * (3._dp * z + 2._dp * b - 2._dp) - b * (3._dp * b + 2._dp) + a
    dadp = a / prs
    dbdp = b / prs
    dzdp = ((z * (b * (6._dp * b - z + 2._dp) - a) &
             - b * (b * (3._dp * b + 2._dp) - 2._dp * a)) &
            / (prs * dqdz))
    d2zdp2 = 2._dp * (dbdp * (dbdp * (3._dp * z - 3._dp * b - 1._dp) + dadp) &
                      - dzdp * (2._dp * dbdp * (z - 3._dp * b - 1._dp) + dadp) &
                      - dzdp * dzdp * (3._dp * z + b - 1._dp)) / dqdz
    zs = vs * prt
    z = z - zs
    dzdp = dzdp - zs / prs
  end subroutine getpt_z_dp_dp2

  subroutine getpt_z_dt_dt2(nc, prs, tem, y, z, dzdt, d2zdt2)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: z, dzdt, d2zdt2
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: z, dzdt, d2zdt2
    real(dp) :: beta, prt, am, bm, vs, damdt, d2amdt2, a, b, &
                dadt, dbdt, dqda, dqdb, dqdz, d2adt2, d2bdt2, zs
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_z_dt_dt2(nc, tem, y, am, bm, vs, damdt, d2amdt2)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    dadt = prt * beta * damdt - 2._dp * a / tem
    dbdt = -b / tem
    dqda = b - z
    dqdb = a - z * (z - 6._dp * b - 2._dp) - b * (3._dp * b + 2._dp)
    dqdz = z * (3._dp * z + 2._dp * b - 2._dp) - b * (3._dp * b + 2._dp) + a
    dzdt = (dbdt * dqdb + dadt * dqda) / dqdz
    d2adt2 = prt * beta * (d2amdt2 - damdt / tem) - 3._dp * dadt / tem
    d2bdt2 = -2._dp * dbdt / tem
    d2zdt2 = (2._dp * dadt * dbdt &
              + d2adt2 * dqda &
              + d2bdt2 * dqdb &
              - 2._dp * dbdt * dbdt * (1._dp - 3._dp * (z - b)) &
              - 2._dp * dzdt * (2._dp * dbdt * (z - 3._dp * b - 1._dp) + dadt) &
              - 2._dp * dzdt * dzdt * (3._dp * z + b - 1._dp)) / dqdz
    zs = vs * prt
    z = z - zs
    dzdt = dzdt + zs / tem
    d2zdt2 = d2zdt2 - 2._dp * zs / (tem * tem)
  end subroutine getpt_z_dt_dt2

  subroutine getpt_z_dp_dt_dpdt(nc, prs, tem, y, z, dzdp, dzdt, d2zdpdt)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: z, dzdp, dzdt, d2zdpdt
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: z, dzdp, dzdt, d2zdpdt
    real(dp) :: beta, prt, am, bm, vs, damdt, a, b, zs, dadt, dbdt, &
                dqda, dqdb, dqdz
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_z_dt(nc, tem, y, am, bm, vs, damdt)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    dadt = prt * beta * damdt - 2._dp * a / tem
    dbdt = -b / tem
    dqda = b - z
    dqdb = a - z * (z - 6._dp * b - 2._dp) - b * (3._dp * b + 2._dp)
    dqdz = z * (3._dp * z + 2._dp * b - 2._dp) - b * (3._dp * b + 2._dp) + a
    dzdp = (b * dqdb + a * dqda) / (dqdz * prs)
    dzdt = (dbdt * dqdb + dadt * dqda) / dqdz
    d2zdpdt = ((dadt * (b + dqda) &
                + dbdt * (a - 2._dp * b * (3._dp * (b - z) + 1._dp) + dqdb) &
                - dzdt * (a + 2._dp * b * (z - 3._dp * b - 1._dp))) / prs &
               - dzdp * (2._dp * dbdt * (z - 3._dp * b - 1._dp) &
                         + 2._dp * dzdt * (3._dp * z + b - 1._dp) &
                         + dadt)) / dqdz
    zs = vs * prt
    z = z - zs
    dzdp = dzdp - zs / prs
    dzdt = dzdt + zs / tem
    d2zdpdt = d2zdpdt + zs / (prs * tem)
  end subroutine getpt_z_dp_dt_dpdt

  subroutine getpt_lnphi(nc, prs, tem, y, lnphi)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: lnphi(nc)
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: lnphi(nc)
    integer :: i
    real(dp) :: beta, prt, am, bm, lnphis(nc), damdy(nc), dbmdy(nc), s1, s2, &
                a, b, z, c1, c2, c3, c4
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_lnphi(nc, tem, y, am, bm, lnphis, damdy, dbmdy, s1, s2)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    c1 = INVD1PD2 * a / b * log((z + b * D1) / (z + b * D2))
    c2 = s1 * (1._dp - z) - s2 * c1 - log(z - b)
    c3 = c1 / am
    c4 = (c1 - z + 1._dp) / bm
    do i = 1, nc
      lnphi(i) = c2 + c3 * damdy(i) - c4 * dbmdy(i) - prt * lnphis(i)
    end do
  end subroutine getpt_lnphi

  subroutine getpt_lnphi_dp(nc, prs, tem, y, lnphi, dlnphidp)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: lnphi(nc), dlnphidp(nc)
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: lnphi(nc), dlnphidp(nc)
    integer :: i
    real(dp) :: beta, prt, am, bm, lnphis(nc), damdy(nc), dbmdy(nc), s1, s2, &
                a, b, z, dzdp, zbd1, zbd2, c0, c1, c2, c3, c4, c5, c6, c7, c8
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_lnphi(nc, tem, y, am, bm, lnphis, damdy, dbmdy, s1, s2)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    dzdp = ((z * (b * (6._dp * b - z + 2._dp) - a) &
             - b * (b * (3._dp * b + 2._dp) - 2._dp * a)) &
            / (prs * (z * (3._dp * z + 2._dp * b - 2._dp) &
                      - b * (3._dp * b + 2._dp) + a)))
    zbd1 = 1._dp / (z + b * D1)
    zbd2 = 1._dp / (z + b * D2)
    c0 = INVD1PD2 * a / b
    c1 = log(zbd2 / zbd1)
    c2 = s1 * (1._dp - z) - s2 * c0 * c1 - log(z - b)
    c3 = c0 * c1 / am
    c4 = (c0 * c1 - z + 1._dp) / bm
    c5 = dzdp * (zbd1 - zbd2) + b / prs * (D1 * zbd1 - D2 * zbd2)
    c6 = (b / prs - dzdp) / (z - b) - s1 * dzdp - s2 * c0 * c5
    c7 = c0 * c5 / am
    c8 = (c0 * c5 - dzdp) / bm
    do i = 1, nc
      lnphi(i) = c2 + c3 * damdy(i) - c4 * dbmdy(i) - prt * lnphis(i)
      dlnphidp(i) = c6 + c7 * damdy(i) - c8 * dbmdy(i) - beta * lnphis(i)
    end do
  end subroutine getpt_lnphi_dp

  subroutine getpt_lnphi_dt(nc, prs, tem, y, lnphi, dlnphidt)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: lnphi(nc), dlnphidt(nc)
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: lnphi(nc), dlnphidt(nc)
    integer :: i
    real(dp) :: beta, prt, am, bm, lnphis(nc), damdy(nc), dbmdy(nc), s1, s2, &
                damdt, d2amdydt(nc), ds2dt, a, b, z, dqdz, dqda, dqdb, dadt, &
                dbdt, dzdt, zbd1, zbd2, c0, c1, c2, c3, c4, c5, c6, c7, c8, &
                c9, c10, c11
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_lnphi_dt(nc, tem, y, am, bm, lnphis, damdy, dbmdy, s1, s2, &
                        damdt, d2amdydt, ds2dt)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    dqdz = z * (3._dp * z + 2._dp * b - 2._dp) - b * (3._dp * b + 2._dp) + a
    dqda = b - z
    dqdb = a - z * (z - 6._dp * b - 2._dp) - b * (3._dp * b + 2._dp)
    dadt = prt * beta * damdt - 2._dp * a / tem
    dbdt = -b / tem
    dzdt = (dqdb * dbdt + dqda * dadt) / dqdz
    zbd1 = 1._dp / (z + b * D1)
    zbd2 = 1._dp / (z + b * D2)
    c0 = INVD1PD2 * a / b
    c1 = log(zbd2 / zbd1)
    c2 = s1 * (1._dp - z) - s2 * c0 * c1 - log(z - b)
    c3 = c0 * c1 / am
    c4 = (c0 * c1 - z + 1._dp) / bm
    c5 = c0 * (damdt / am - 1._dp / tem)
    c6 = dzdt * (zbd1 - zbd2) + dbdt * (D1 * zbd1 - D2 * zbd2)
    c7 = c5 * c1 + c0 * c6
    c8 = (dzdt - dbdt) / dqda - s1 * dzdt - ds2dt * c0 * c1 - s2 * c7
    c9 = (c7 - c3 * damdt) / am
    c10 = (c7 - dzdt) / bm
    c11 = prt / tem
    do i = 1, nc
      lnphi(i) = c2 + c3 * damdy(i) - c4 * dbmdy(i) - prt * lnphis(i)
      dlnphidt(i) = c8 + c9 * damdy(i) + c3 * d2amdydt(i) - c10 * dbmdy(i) &
                    + c11 * lnphis(i)
    end do
  end subroutine getpt_lnphi_dt

  subroutine getpt_lnphi_dy(nc, prs, tem, y, lnphi, dlnphidy)
    ! ------------------------------------------------------------------------ !
    ! f2py integer, intent(hide) :: nc
    ! f2py real(8), intent(in) :: prs, tem, y(nc)
    ! f2py real(8), intent(out) :: lnphi(nc), dlnphidy(nc, nc)
    ! ------------------------------------------------------------------------ !
    integer, intent(in) :: nc
    real(dp), intent(in) :: prs, tem, y(nc)
    real(dp), intent(out) :: lnphi(nc), dlnphidy(nc, nc)
    integer :: i, j
    real(dp) :: beta, prt, am, bm, lnphis(nc), damdy(nc), dbmdy(nc), s1, s2, &
                d2amdy2(nc, nc), d2bmdy2(nc, nc), ds1dy(nc), ds2dy(nc), a, b, &
                z, dqdz, dqda, dqdb, dzdy, zbd1, zbd2, c0, c1, c2, c3, c4, &
                c5, c6, c7, c8, c9, c10
    beta = 1._dp / (RGAS * tem)
    prt = prs * beta
    call mr%mr_lnphi_dy(nc, tem, y, am, bm, lnphis, damdy, dbmdy, s1, s2, &
                        d2amdy2, d2bmdy2, ds1dy, ds2dy)
    a = am * prt * beta
    b = bm * prt
    call solve(a, b, z)
    dqdz = z * (3._dp * z + 2._dp * b - 2._dp) - b * (3._dp * b + 2._dp) + a
    dqda = b - z
    dqdb = a - z * (z - 6._dp * b - 2._dp) - b * (3._dp * b + 2._dp)
    zbd1 = 1._dp / (z + b * D1)
    zbd2 = 1._dp / (z + b * D2)
    c0 = INVD1PD2 * a / b
    c1 = log(zbd2 / zbd1)
    c2 = s1 * (1._dp - z) - s2 * c0 * c1 - log(z - b)
    c3 = c0 * c1 / am
    c4 = (c0 * c1 - z + 1._dp) / bm
    do j = 1, nc
      lnphi(j) = c2 + c3 * damdy(j) - c4 * dbmdy(j) - prt * lnphis(j)
      dzdy = prt * (dqdb * dbmdy(j) + beta * dqda * damdy(j)) / dqdz
      c5 = c0 * (damdy(j) / am - dbmdy(j) / bm)
      c6 = (zbd1 - zbd2) * dzdy + prt * (D1 * zbd1 - D2 * zbd2) * dbmdy(j)
      c7 = c5 * c1 + c0 * c6
      c8 = ds1dy(j) * (1._dp - z) - s1 * dzdy + s2 * c7 + ds2dy(j) * c0 * c1 &
           + (dzdy - prt * dbmdy(j)) / dqda
      c9 = (c7 - c3 * damdy(j)) / am
      c10 = (c7 - dzdy - c4 * dbmdy(j)) / bm
      do i = 1, nc
        dlnphidy(i, j) = c8 + c9 * damdy(i) + c3 * d2amdy2(i, j) &
                         - c10 * dbmdy(i) - c4 * d2bmdy2(i, j)
      end do
    end do
  end subroutine getpt_lnphi_dy

  function fdg(z1, z2, a, b) result(dg)
    ! ------------------------------------------------------------------------ !
    ! Calculate the Gibbs energy difference between two states corresponding
    ! to different roots (compressibility factors) of the modified Peng-
    ! Robinson equation of state (EOS).
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | z1         | real(8) | in     | The first root of the EOS.           |
    ! +------------+---------+--------+--------------------------------------+
    ! | z2         | real(8) | in     | The second root of the EOS.          |
    ! +------------+---------+--------+--------------------------------------+
    ! | a          | real(8) | in     | The coefficient of the cubic form of |
    ! |            |         |        | the  EOS.                            |
    ! +------------+---------+--------+--------------------------------------+
    ! | b          | real(8) | in     | The coefficient of the cubic form of |
    ! |            |         |        | the  EOS.                            |
    ! +------------+---------+--------+--------------------------------------+
    ! ------------------------------------------------------------------------ !
    real(dp), intent(in) :: z1, z2, a, b
    real(dp) :: dg
    dg = log((z2 - b) / (z1 - b)) &
         + (z1 - z2) &
         + INVD1PD2 * a / b * log((z1 + b * D1) * (z2 + b * D2) &
                                  / ((z1 + b * D2) * (z2 + b * D2)))
  end function fdg

  subroutine solve_cardano(aeos, beos, x)
    ! ------------------------------------------------------------------------ !
    ! Solve the modified Peng-Robinson equation of state (EOS).
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | aeos       | real(8) | in     | The coefficient of the cubic form of |
    ! |            |         |        | the  EOS.                            |
    ! +------------+---------+--------+--------------------------------------+
    ! | beos       | real(8) | in     | The coefficient of the cubic form of |
    ! |            |         |        | the  EOS.                            |
    ! +------------+---------+--------+--------------------------------------+
    ! | x          | real(8) | in     | The solution of the EOS consisting   |
    ! |            |         |        | to the lowest Gibbs energy.          |
    ! +------------+---------+--------+--------------------------------------+
    !
    ! This method implements Cardano's method to solve the cubic form of
    ! the equation of state.
    !
    ! When the equation of state has three real roots, the correct solution
    ! will be chosen based on the comparison of Gibbs energies corresponding
    ! to these roots.
    !
    ! In cases with three real roots, the first one is calculated using
    ! Cardano's formula. Once the first real root is known, the cubic
    ! polynomial can be "deflated", i.e., divided by a linear factor. The
    ! other roots are then found by solving the quadratic equation. For
    ! details, see the paper of U.K. Deiters and RGAS. Macias-Salinas, 2014
    ! (doi: 10.1021/ie4038664).
    !
    ! The accuracy of Cardano's method is known to be low near the critical
    ! point or in the low-temperature and low-pressure region. Therefore,
    ! one Newton iteration will be performed to refine the solution if the
    ! absolute value of the equation is greater than `1e-12`.
    ! ------------------------------------------------------------------------ !
    ! f2py real(8), intent(in) :: aeos, beos
    ! f2py real(8), intent(out) :: x
    ! ------------------------------------------------------------------------ !
    real(dp), intent(in) :: aeos, beos
    real(dp), intent(out) :: x
    real(dp) :: b, c, d, t, p, q, s, x0, x1, x2
    b = beos - 1._dp
    c = aeos - beos * (2._dp + 3._dp * beos)
    d = beos * (beos * (1._dp + beos) - aeos)
    t = b * b
    p = c + t / (-3._dp)
    q = d + b * (2._dp * t - 9._dp * c) / 27._dp
    s = q * q * .25_dp + p * p * p / 27._dp
    if (s > 0._dp) then
      s = sqrt(s)
      x = cbrt(-.5_dp * q + s) + cbrt(-.5_dp * q - s) + b / (-3._dp)
      t = d + x * (c + x * (b + x))
      if ((t > 1e-12_dp) .or. (t < -1e-12_dp)) then
        x = x - t / (c + x * (2._dp * b + 3._dp * x))
      end if
    else
      x0 = 2._dp * sqrt(p / (-3._dp)) &
           * cos(acos(1.5_dp * q * sqrt(-3._dp / p) / p) / 3._dp) &
           + b / (-3._dp)
      t = d + x0 * (c + x0 * (b + x0))
      if ((t > 1e-12_dp) .or. (t < -1e-12_dp)) then
        x0 = x0 - t / (c + x0 * (2._dp * b + 3._dp * x0))
      end if
      t = b + x0
      d = sqrt(t * t + 4._dp * d / x0)
      x1 = (-t + d) * .5_dp
      x2 = (-t - d) * .5_dp
      if (x2 > beos) then
        if (fdg(x0, x2, aeos, beos) < 0._dp) then
          x =  x0
        else
          x = x2
        end if
      else if (x1 > beos) then
        if (fdg(x0, x1, aeos, beos) < 0._dp) then
          x = x0
        else
          x = x1
        end if
      else
        x = x0
      end if
    end if
  end subroutine solve_cardano

  subroutine solve(aeos, beos, x)
    ! ------------------------------------------------------------------------ !
    ! Solve the modified Peng-Robinson equation of state (EOS).
    !
    ! +------------+---------+--------+--------------------------------------+
    ! | Parameters | Type    | Intent | Description                          |
    ! +============+=========+========+======================================+
    ! | aeos       | real(8) | in     | The coefficient of the cubic form of |
    ! |            |         |        | the  EOS.                            |
    ! +------------+---------+--------+--------------------------------------+
    ! | beos       | real(8) | in     | The coefficient of the cubic form of |
    ! |            |         |        | the  EOS.                            |
    ! +------------+---------+--------+--------------------------------------+
    ! | x          | real(8) | in     | The solution of the EOS consisting   |
    ! |            |         |        | to the lowest Gibbs energy.          |
    ! +------------+---------+--------+--------------------------------------+
    !
    ! This method implements Halley's method to solve the cubic form of the
    ! equation of state.
    !
    ! When the equation of state has three real roots, the correct solution
    ! will be chosen based on the comparison of Gibbs energies corresponding
    ! to these roots.
    !
    ! This implementation starts with determining the number of real roots.
    ! In cases with only one real root, the sign of the equation at the
    ! inflection point is used to determine whether this root is vapour-like
    ! or liquid-like. For details on this step, see the paper of J. Zhao
    ! et al, 2025 (doi: 10.1016/j.fluid.2025.114466). If the root is
    ! vapour-like, the initial guess is calculated according to RGAS. Gosset
    ! et al, 1986 (doi: 10.1016/0378-3812(86)85061-0). Otherwise, the Vieta
    ! initialization scheme is used (U.K. Deiters and RGAS. Macias-Salinas, 2014
    ! (doi: 10.1021/ie4038664)). To prevent oscillations of the iterative
    ! method, the sign of the first derivative is monitored as described by
    ! RGAS. Gosset et al, 1986 (doi: 10.1016/0378-3812(86)85061-0).
    !
    ! In cases with three real roots, the initial guess for the maximum root
    ! is calculated using the Laguerre-Nair-Samuelson initialization scheme
    ! as mentioned by U.K. Deiters and RGAS. Macias-Salinas, 2014 (doi:
    ! 10.1021/ie4038664). This procedure ensures that the starting point will
    ! be greater than the root, and the iterative procedure can be used safely
    ! due to Darboux's theorem. Once the first real root is known, the cubic
    ! polynomial can be "deflated", i.e., divided by a linear factor. The
    ! other roots are then found by solving the resulting quadratic equation.
    !
    ! It should be noted that in all cases, if Halley's method does not
    ! converge in ten iterations, Cardano's formula will be used with one
    ! Newton's step to refine the solution if the absolute value of the
    ! equation is greater than `1e-12`.
    ! ------------------------------------------------------------------------ !
    ! f2py real(8), intent(in) :: aeos, beos
    ! f2py real(8), intent(out) :: x
    ! ------------------------------------------------------------------------ !
    real(dp), intent(in) :: aeos, beos
    real(dp), intent(out) :: x
    logical :: not_solved
    integer :: i
    real(dp) :: b, c, d, t, p, q, s, x0, x1, x2, y, dydx, d2ydx2, dx
    b = beos - 1._dp
    c = aeos - beos * (2._dp + 3._dp * beos)
    d = beos * (beos * (1._dp + beos) - aeos)
    t = b * b
    p = c + t / (-3._dp)
    q = d + b * (2._dp * t - 9._dp * c) / 27._dp
    s = q * q * .25_dp + p * p * p / 27._dp
    if (s > 0._dp) then
      x = b / (-3._dp)
      if (d + x * (c + x * (b + x)) < 0._dp) then
        if (beos < 1._dp) then
          x = 1._dp
        else
          x = beos
        end if
        not_solved = .true.
        do i = 1, 10
          dydx = c + x * (2._dp * b + 3._dp * x)
          if (dydx < 0._dp) then
            x = x * 2._dp
          else
            y = d + x * (c + x * (b + x))
            d2ydx2 = 6._dp * x + 2._dp * b
            dx = y * dydx / (dydx * dydx - .5_dp * y * d2ydx2)
            if ((dx < 1e-12_dp) .and. (dx > -1e-12_dp)) then
              not_solved = .false.
              exit
            end if
            x = x - dx
          end if
        end do
        if (not_solved) then
          s = sqrt(s)
          x = cbrt(-.5_dp * q + s) + cbrt(-.5_dp * q - s) + b / (-3._dp)
          y = d + x * (c + x * (b + x))
          if ((y > 1e-12_dp) .or. (y < -1e-12_dp)) then
            x = x - y / (c + x * (2._dp * b + 3._dp * x))
          end if
        end if
      else
        x = -d / c
        if (x < beos) then
          x = beos
        end if
        not_solved = .true.
        do i = 1, 10
          dydx = c + x * (2._dp * b + 3._dp * x)
          if (dydx < 0._dp) then
            x = x * .3_dp
            if (x < beos) then
              x = beos
            end if
          else
            y = d + x * (c + x * (b + x))
            d2ydx2 = 6._dp * x + 2._dp * b
            dx = y * dydx / (dydx * dydx - .5_dp * y * d2ydx2)
            if ((dx > -1e-12_dp) .and. (dx < 1e-12_dp)) then
              not_solved = .false.
              exit
            end if
            x = x - dx
          end if
        end do
        if (not_solved) then
          s = sqrt(s)
          x = cbrt(-.5_dp * q + s) + cbrt(-.5_dp * q - s) + b / (-3._dp)
          y = d + x * (c + x * (b + x))
          if ((y > 1e-12_dp) .or. (y < -1e-12_dp)) then
            x = x - y / (c + x * (2._dp * b + 3._dp * x))
          end if
        end if
      end if
    else
      x0 = b / (-3._dp) + 2._dp / 3._dp * sqrt(-3._dp * p)
      not_solved = .true.
      do i = 1, 10
        dydx = c + x0 * (2._dp * b + 3._dp * x0)
        d2ydx2 = 6._dp * x0 + 2._dp * b
        y = d + x0 * (c + x0 * (b + x0))
        dx = y * dydx / (dydx * dydx - .5_dp * y * d2ydx2)
        if ((dx < 1e-12_dp) .and. (dx > -1e-12_dp)) then
          not_solved = .false.
          exit
        end if
        x0 = x0 - dx
      end do
      if (not_solved) then
        x0 = 2._dp * sqrt(p / (-3._dp)) &
             * cos(acos(1.5_dp * q * sqrt(-3._dp / p) / p) / 3._dp) &
             + b / (-3._dp)
        y = d + x0 * (c + x0 * (b + x0))
        if ((y > 1e-12_dp) .or. (y < -1e-12_dp)) then
          x0 = x0 - y / (c + x0 * (2._dp * b + 3._dp * x0))
        end if
      end if
      t = b + x0
      d = sqrt(t * t + 4._dp * d / x0)
      x1 = (-t + d) * .5_dp
      x2 = (-t - d) * .5_dp
      if (x2 > beos) then
        if (fdG(x0, x2, aeos, beos) < 0._dp) then
          x = x0
        else
          x = x2
        end if
      else if (x1 > beos) then
        if (fdG(x0, x1, aeos, beos) < 0._dp) then
          x = x0
        else
          x = x1
        end if
      else
        x = x0
      end if
    end if
  end subroutine solve

end module pr78

