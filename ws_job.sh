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

# mpirun -n 17 ./ps4.out -ksp_type cg -pc_type mg -da_refine 2 -ksp_converged_reason -ksp_monitor -ksp_view -log_view

#echo
#echo 1D
#echo


# mpirun -n 4 ./ps_1D.out -ksp_type cg -pc_type mg -da_refine 2 -ksp_converged_reason -ksp_monitor #-log_view

# mpirun -n 4 ./ps_2D.out -ksp_type cg -pc_type mg -da_refine 2 -ksp_converged_reason -ksp_monitor

#echo
#echo 3D
#echo

mpirun -n 1 ./p3D1.out -ksp_type preonly -pc_type ilu -ksp_converged_reason -ksp_monitor