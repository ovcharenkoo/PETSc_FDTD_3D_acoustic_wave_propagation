#!/bin/bash

PETSC_MPIRUN=${PETSC_DIR}/${PETSC_ARCH}/bin/mpirun

rm -rf ./wavefields/tmp*
rm -rf ./seism/seis*

${PETSC_MPIRUN} -n 2 ./p3D_acoustic_O24.out -pc_type asm -pc_asm_overlap 2 -sub_pc_type ilu  -da_refine 1 -ksp_converged_reason

