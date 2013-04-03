!!===----------------------------------------------------------------------===//
!!
!!     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
!!        compiler for NVIDIA GPUs, targeting numerical modeling code.
!!
!! This file is distributed under the University of Illinois Open Source
!! License. See LICENSE.TXT for details.
!!
!!===----------------------------------------------------------------------===//

! Generate input signal X and filter mask H.
subroutine generate_input(H, szh, X, szx)

  implicit none
  integer(kind=IKIND), intent(in) :: szh, szx
  real(kind=RKIND), intent(out) :: H(szh), X(szx)
  integer(kind=IKIND) :: i, ioffset
  real(kind=RKIND) :: roffset
  integer, parameter :: rkind = RKIND
  
  do i = 1, szh
    call random_number(H(i))
  enddo
  do i = 1, szx
    call random_number(X(i))
  enddo
  
  ! Put an exact filter match under random offset (to check
  ! correctness).
  call random_number(roffset)
  ioffset = int(roffset * (szx - szh))
  do i = 1, szh
    X(i + ioffset) = 1.0_rkind / H(i)
  enddo
  write(*,*) 'seeded match at X(i), i = ', 1 + ioffset

end subroutine generate_input



! Perform simple correctness test: check output result
! contains the correct number of exact matches.
subroutine check_output(Y, szy, szh)

  implicit none
  integer(kind=IKIND), intent(in) :: szy, szh
  real(kind=RKIND), intent(in) :: Y(szy)
  integer(kind=IKIND) :: i
  real(kind=RKIND) :: eps
  integer, parameter :: rkind = RKIND
  
  eps = epsilon(eps)
  do i = 1, szy
    if (abs(Y(i) - szh) .le. eps) then
      write(*,*) 'found match at Y(i), i = ', i
    endif
  enddo

end subroutine check_output



! Perform filter matching.
subroutine match_filter(H, szh, X, szx, Y, szy)

  implicit none
  integer(kind=IKIND), intent(in) :: szh, szx, szy
  real(kind=RKIND), intent(in) :: H(szh), X(szx)
  real(kind=RKIND), intent(out) :: Y(szy)
  integer(kind=IKIND) :: i, j
  integer, parameter :: rkind = RKIND

  do i = 1, szy
    Y(i) = 0.0_rkind
    do j = 1, szh
      Y(i) = Y(i) + X(i + j - 1) * H(j)
    enddo
  enddo

end subroutine match_filter



program main

  implicit none
  character(len=128) :: arg
  integer(kind=IKIND) :: szh, szx, szy
  real(kind=RKIND), allocatable, dimension(:) :: H, X, Y
  real(kind=RKIND) :: tstart, tfinish
  integer :: argc
  integer, parameter :: rkind = RKIND

  ! Check usage
  argc = command_argument_count()
  if (argc .ne. 2) then
    call get_command_argument(0, arg)
    write(*,*) 'Usage: ', trim(arg), ' <signal_length> <filter_length>'
    stop
  endif

  ! Setup szh, szx, szy
  call get_command_argument(1, arg)
  read(arg, '(I64)') szx
  call get_command_argument(2, arg)
  read(arg, '(I64)') szh
  szy = szx - (szh - 1)
  
  ! Check valid numbers
  if (szx < szh) then
    write(*,*) '<signal_length> must be greater or equal to the <filter_length>'
    stop
  endif

  ! Allocate H, X and Y arrays.
  allocate(H(szh), X(szx), Y(szy))

  ! Generate input signal X and filter mask H.
  call generate_input(H, szh, X, szx)

  ! Perform filter matching.
  call cpu_time(tstart)
  call match_filter(H, szh, X, szx, Y, szy)
  call cpu_time(tfinish)
  write(*,*) 'matching time = ', tfinish - tstart, ' sec'

  ! Perform simple correctness test.
  call check_output(Y, szy, szh)

  write(*,*) 'control H: ', sum(H), minval(H), maxval(H)
  write(*,*) 'control X: ', sum(X), minval(X), maxval(X)
  write(*,*) 'control Y: ', sum(Y), minval(Y), maxval(Y)

  deallocate(H, X, Y)

end program main

