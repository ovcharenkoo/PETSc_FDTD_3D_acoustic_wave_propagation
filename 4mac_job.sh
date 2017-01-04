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

#PETSC_MPIRUN=/Users/ovcharoo/Software/petsc-3.7.3/arch-darwin-c-debug/bin/mpirun

PETSC_MPIRUN=${PETSC_DIR}/${PETSC_ARCH}/bin/mpirun


rm -rf ./wavefields/*
rm -rf ./seism/*

#${PETSC_MPIRUN} -n 4 ./p3D_acoustic_O24.out -da_refine 1

#${PETSC_MPIRUN} -n 4 ./p3D_acoustic_O24.out -pc_type asm -pc_asm_overlap 2 -sub_pc_type ilu  -da_refine 2 -ksp_converged_reason


${PETSC_MPIRUN} -n 32 ./p3D_acoustic_O24.out -pc_type  asm -pc_asm_overlap 2 -sub_pc_type ilu -da_refine 4 -ksp_converged_reason

#${PETSC_MPIRUN} -n 32 ./p3D_acoustic_O24.out -pc_type  asm -pc_asm_overlap 2 -sub_pc_type ilu -da_grid_x 510 -da_grid_y 510 -da_grid_z 510 -ksp_converged_reason
