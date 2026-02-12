! REQUIRES: flang, libc
! RUN: %libomptarget-compile-fortran-run-and-check-generic

! REQUIRES: flang, libc
! RUN: %libomptarget-compile-fortran-run-and-check-generic

program hello_gpu
  implicit none

  integer :: i
  real :: r
  complex :: c
  logical :: l

  i = 42
  r = 3.14
  c = (1.0, -1.0)
  l = .true.

  ! CHECK: Hello from GPU
  ! CHECK: Hello from GPU
  ! CHECK: Hello from GPU
  ! CHECK: Hello from GPU
  !$omp target teams num_teams(4)
  !$omp parallel num_threads(1)
    print *, "Hello from GPU"
  !$omp end parallel
  !$omp end target teams

  ! CHECK: 42
  !$omp target teams num_teams(1)
  !$omp parallel num_threads(1)
    print *, i
  !$omp end parallel
  !$omp end target teams

  ! CHECK: 3.14
  !$omp target teams num_teams(1)
  !$omp parallel num_threads(1)
    print *, r
  !$omp end parallel
  !$omp end target teams

  ! CHECK: (1.,-1.)
  !$omp target teams num_teams(1)
  !$omp parallel num_threads(1)
    print *, c
  !$omp end parallel
  !$omp end target teams

  ! CHECK: T
  !$omp target teams num_teams(1)
  !$omp parallel num_threads(1)
    print *, l
  !$omp end parallel
  !$omp end target teams

end program hello_gpu
