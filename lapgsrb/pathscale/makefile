##===----------------------------------------------------------------------===##
##
##     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
##        compiler for NVIDIA GPUs, targeting numerical modeling code.
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##

-include ../../makefile.in

# Verbose setting
VERBOSE ?= 0
SILENT ?= @
ifeq ($(VERBOSE), 1)
SILENT =
endif

# Floating-point type: float or double
real ?= float

# PathScale/OpenACC compiler/linker options
PATHCC = $(PATHCCPATH)pathcc -acc -O3 -I. -I$(PATHCC_INCLUDE_PATH)
PATHF90 = $(PATHCCPATH)pathf90 -acc -O3 -I$(PATHCC_INCLUDE_PATH)

# NVIDIA compile options
NVCC ?= nvcc -O3

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256

all: lapgsrb

lapgsrb: lapgsrb.o timing.o openacc_profiling.o
	$(SILENT)$(PATHCC) $^ -o $@ -lrt -L$(shell dirname $(shell which nvcc))/../lib64 -lcudart -lcuda -lstdc++ -lgcc_s

lapgsrb.o: ../lapgsrb.c ../timing.h
	$(SILENT)$(PATHCC) -Dreal=$(real) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(PATHCC) -c $< -o $@

openacc_profiling.o: openacc_profiling.cu
	$(SILENT)$(NVCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o lapgsrb *.ptx *.gpu *.bin

test: lapgsrb
	$(SILENT)ACC_TIME=1 ./$< $(NX) $(NY) $(NS) $(NT)
