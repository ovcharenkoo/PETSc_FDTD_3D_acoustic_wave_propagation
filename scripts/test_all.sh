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

for KSP in cg gmres hypre
do
for PC in bjacobi asm ilu mg gamg
do
	echo PAR ${KSP} ${PC}
	${PETSC_MPIRUN} -n 24 ./p3D_acoustic.out -ksp_type ${KSP} -pc_type ${PC}  -da_refine 3 | grep -iwe "PAR*" -iwe "Total time*" > output.txt
done
done
