#!/usr/bin/env bash
# Build CalculiX 2.23 with MKL PARDISO and multithreaded SPOOLES into build/ccx/bin/ccx.
#
# Needs gcc, gfortran, make, curl, uv and the system ARPACK runtime (libarpack.so.2, which the
# apt calculix-ccx package pulls in). MKL comes from the PyPI static wheels (no sudo) and is
# linked statically. mkl_serv_intel_cpu_true is overridden so MKL takes its AVX2 code paths on
# AMD CPUs (about 13% faster factorization on a Ryzen 2700X; results unchanged).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/build/ccx
CCX_VER=2.23
MKL_VER=2024.2.2   # newest mkl-static wheel for manylinux x86_64
ARPACK=/usr/lib/x86_64-linux-gnu/libarpack.so.2
JOBS=$(nproc)

mkdir -p "$B/bin" && cd "$B"
[ -f "$ARPACK" ] || { echo "missing $ARPACK (sudo apt install libarpack2t64)"; exit 1; }

# Sources
[ -f ccx_$CCX_VER.src.tar.bz2 ] || curl -fsSLO https://www.dhondt.de/ccx_$CCX_VER.src.tar.bz2
[ -f spooles.2.2.tgz ] || curl -fsSLO https://www.netlib.org/linalg/spooles/spooles.2.2.tgz
rm -rf CalculiX spooles && tar xjf ccx_$CCX_VER.src.tar.bz2 && mkdir spooles && tar xzf spooles.2.2.tgz -C spooles

# MKL (static libraries + headers)
if [ ! -d mkl ]; then
    uv run --no-project --with pip python -m pip download -q --no-deps --only-binary=:all: \
        -d mkl_whl mkl-static==$MKL_VER mkl-include==$MKL_VER
    for w in mkl_whl/*.whl; do python3 -m zipfile -e "$w" mkl/; done
fi
MKL_LIB=$B/mkl/mkl_static-$MKL_VER.data/data/lib
MKL_INC=$B/mkl/mkl_include-$MKL_VER.data/data/include

# SPOOLES 2.2, serial + MT objects in one spooles.a
(
    cd spooles
    sed -i 's|^  CC = /usr/lang-4.0/bin/cc|  CC = gcc|; s|^  OPTLEVEL = -O$|  OPTLEVEL = -O3|' Make.inc
    sed -i 's/drawTree/draw/' Tree/src/makeGlobalLib   # known typo in the 2.2 makefile
    make lib > build.log 2>&1
    (cd MT/src && make -f makeGlobalLib > ../../build_mt.log 2>&1)
)

# CalculiX
cd CalculiX/ccx_$CCX_VER/src
printf '/* Make MKL take its Intel (AVX2) code paths on AMD CPUs. */\nint mkl_serv_intel_cpu_true(void) { return 1; }\n' > mkl_amd_fix.c
cat > Makefile_pardiso <<EOF
CFLAGS = -Wall -O2 -fopenmp -I $B/spooles -I $MKL_INC \\
	-DARCH="Linux" -DSPOOLES -DUSE_MT=1 -DARPACK -DMATRIXSTORAGE -DNETWORKOUT -DPARDISO
FFLAGS = -Wall -O2 -cpp -fopenmp -fallow-argument-mismatch
CC=gcc
FC=gfortran
.c.o :
	\$(CC) \$(CFLAGS) -c \$<
.f.o :
	\$(FC) \$(FFLAGS) -c \$<
include Makefile.inc
SCCXMAIN = ccx_$CCX_VER.c
OCCXF = \$(SCCXF:.f=.o)
OCCXC = \$(SCCXC:.c=.o)
OCCXMAIN = \$(SCCXMAIN:.c=.o)
# --allow-multiple-definition: mkl_amd_fix.o comes first, so its definition wins.
LIBS = mkl_amd_fix.o $B/spooles/spooles.a $ARPACK -Wl,--allow-multiple-definition \\
	-Wl,--start-group $MKL_LIB/libmkl_intel_lp64.a $MKL_LIB/libmkl_gnu_thread.a $MKL_LIB/libmkl_core.a -Wl,--end-group \\
	-lgomp -lpthread -lm -ldl
ccx: \$(OCCXMAIN) ccx_$CCX_VER.a mkl_amd_fix.o
	\$(FC) -O2 -o \$@ \$(OCCXMAIN) ccx_$CCX_VER.a \$(LIBS) -fopenmp
ccx_$CCX_VER.a: \$(OCCXF) \$(OCCXC)
	ar r \$@ \$?
EOF
make -j"$JOBS" -f Makefile_pardiso > "$B/ccx_build.log" 2>&1
install -m 755 ccx "$B/bin/ccx"
echo "built $B/bin/ccx"
