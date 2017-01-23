# **3D acoustic wave propagation in homogeneous isotropic media using PETSc**

**PETSc** - [Portable, Extensible Toolkit for Scientific Computation](https://www.mcs.anl.gov/petsc/)

Here you find an example of C + PETSc implementation solving acoustic wave equation in 3D.
Krylov methods are used to find iteratively approximate solution of _Ax=b_

![Wavefield example](https://github.com/ovcharenkoo/PETSc_3D_acoustic_wave_propagation/tree/acoustic/doc/step70.png)

### **DISCRETIZATION DETAILS**:
* Finite-Differences in Time Domain (FDTD)
* Implicit time stepping
* O(2,4)
* Schemes derived from Taylor series: 
    * in space [-1:16:-30:16:-1]/12dx2
    * in time [2:-5:4:-1]/dt2

### **MODEL DETAILS**
* Isotropic
* Homogeneous
* Dirichlet boundary conditions

### **HOW TO USE**: 
PETSc has to be installed. Make sure that PETSC_DIR and PETSC_LIB environmental libraries are set

`make all`  
`./run_O22.sh`  
or  
`./run_O24.sh`

Runtime options and number of processors could be changed in shell scripts. 
Changing flags in the code or from runtime one can save and plot either the whole wavefields 
or just seismograms at receiver positions.

**RUNTIME OPTIONS**  
_-vel_ float - propagation velocity [km/s]  
_-xmax_ float - model dimentions [km]  
&nbsp;&nbsp;&nbsp;&nbsp; _-ymax_ float  
&nbsp;&nbsp;&nbsp;&nbsp; _-zmax_ float  
_-dt_ float  - time step [sec]  
_-tmax_ float - max simulation time  
_-isrc_ int - source location [grid points]  
&nbsp;&nbsp;&nbsp;&nbsp; _-jsrc_ int  
&nbsp;&nbsp;&nbsp;&nbsp; _-ksrc_ int  
_-f0_ float - dominant frequency of Ricker wavelet [Hz]  
_-nrec_ int - number of receivers on diagonal  

All options listed above have default values so all of them could be skipped
for a trial run

**EXAMPLE**  
`mpirun -n 2 ./p3D_acoustic_O24.out -xmax 8.0 -ymax 8.0 -zmax 8.0 -vel 3.5 -pc_type asm -pc_asm_overlap 2 -da_refine 1 -ksp_converged_reason`

### **FOLDER STRUCTURE**
/_mfiles_ - matlab routines for seismograms and wavefields visualisation         
/_doc_ - documentation, figures and slides  
/_seism_ - seismigrams in .txt  
/_wavefields_ - wavefields in .m


