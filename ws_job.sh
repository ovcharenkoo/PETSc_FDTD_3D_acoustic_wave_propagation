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

PETSC_MPIRUN=/Users/ovcharoo/Software/petsc-3.7.3/arch-darwin-c-debug/bin/mpirun

#echo
#echo 1D
#echo


# mpirun -n 4 ./ps_1D.out -ksp_type cg -pc_type mg -da_refine 2 -ksp_converged_reason -ksp_monitor #-log_view

#${PETSC_MPIRUN} -n 4 ./p2D.out -ksp_type cg -pc_type mg -da_refine 2 -ksp_converged_reason -ksp_monitor

#echo
#echo 3D
#echo

rm tmp_*
#${PETSC_MPIRUN} -n 4 ./p3D1.out -ksp_type cg -pc_type mg -da_refine 3 -ksp_converged_reason

${PETSC_MPIRUN} -n 4 ./p3D1.out -ksp_type cg -da_refine 1 -ksp_converged_reason

