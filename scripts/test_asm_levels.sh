#!/bin/bash

#SBATCH --account=s1002
#SBATCH --job-name=ps4
#SBATCH --time=00:10:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=32
#SBATCH --ntasks-per-socket=16
#SBATCH	--threads-per-core=1
#SBATCH --ntasks=64
#SBATCH	--nodes=2
#SBATCH	--output=%A.res
#SBATCH	--error=%A.err


PETSC_MPIRUN=${PETSC_DIR}/${PETSC_ARCH}/bin/mpirun

rm ./wavefields/*

for LVL in 0 1 2 3 4 5 6
do
	${PETSC_MPIRUN} -n 4 ./p3D_acoustic.out -pc_type asm -pc_asm_overlap ${LVL}  -da_refine 2 -ksp_converged_reason | grep -iw "Total time*"
done
