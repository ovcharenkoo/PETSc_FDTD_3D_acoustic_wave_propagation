
/*
  Author: Oleg Ovcharenko, PhD student at ErSE, affiliated to ECRC
          King Abdullah University of Science and Technology
  Email:  oleg.ovcharenko@kaust.edu.sa

  3D acoustic wave propagation in homogeneous isotropic media, using PETSc

  Finite-Differences in Time Domain (FDTD)
  Implicit time stepping
  O(2,4), schemes: in space [-1:16:-30:16:-1]/12dx2, in time [2:-5:4:-1]/dt2
*/

#include <stdio.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <math.h>
#include <time.h>

#define debprint(expr) PetscPrintf(PETSC_COMM_WORLD, #expr " = %f \n", expr);

// Constants
#define PI 3.1415926535
#define DEGREES_TO_RADIANS PI/180.f

//User-functions prototypes
PetscErrorCode compute_A_u(KSP, Mat, Mat, void *);  // Build A, for Ax=b
PetscErrorCode update_b_u(KSP, Vec, void *);        // Build b, for Ax=b
PetscErrorCode save_Vec_to_m_file(Vec, void *);     // Save wavefield into MATLAB .m file
PetscErrorCode Save_seismograms_to_txt_files(KSP, void *);  // Save seism. to .txt files
PetscErrorCode source_term(void *);                 // Compute source term for current time step
PetscErrorCode Write_seismograms(KSP, Vec, void *); // Append new value to the seismograms
PetscScalar    ***f3tensor(PetscInt, PetscInt, PetscInt, PetscInt,PetscInt, PetscInt); // Create 3D array

/*
  User-defined structures
*/

// Wavefield
typedef struct {
  Vec u;                      // Pressure wavefield at T
  Vec um1;                    // Pressure wavefield at T-1
  Vec um2;                    // Pressure wavefield at T-2 
  Vec um3;                    // Pressure wavefield at T-3 
} wfield;

// Model parameters
typedef struct {
  PetscInt nx;                // Number of grid points along X
  PetscInt ny;                
  PetscInt nz;                
  PetscScalar dx;             // Grid spacing [km]
  PetscScalar dy;
  PetscScalar dz;
  PetscScalar xmax;           // Limits along OX, xmin ... xmax [km]
  PetscScalar xmin;
  PetscScalar ymax;
  PetscScalar ymin;
  PetscScalar zmax;
  PetscScalar zmin;
  PetscScalar vel;            // Wave propagation velocity [km/s]
} model_par;

typedef struct{
  PetscScalar dt;             // Time step [s]
  PetscScalar t0;
  PetscScalar tmax;           // Total simulation time [s]
  PetscScalar t;              // Current simulation time [s]
  PetscInt it;                // Current simulation step
  PetscInt nt;                // Total simulation steps
} time_par;

typedef struct{
  PetscInt isrc;              // Source position, in grid points
  PetscInt jsrc;
  PetscInt ksrc;
  PetscScalar factor;         // Source ampliturde
  PetscScalar angle_force;
  PetscScalar f0;             // Source frequency
  PetscScalar fx;             // Force x component
  PetscScalar fy;             
  PetscScalar fz;             
} source;

typedef struct
{
  PetscInt nrec;              // Number of receivers
  PetscInt *irec;             // Receiver positions, in grid points
  PetscInt *jrec;
  PetscInt *krec;
  PetscScalar ***seis;        // Array to store seismograms [nrec][nt][2]
} receivers;

typedef struct {              // User context that gathers all the structures above
  wfield wf;
  model_par model;
  time_par time;
  source src;
  receivers rec;
} ctx_t;


typedef int bool;             // TRUE-FALSE definition
#define true 1
#define false 0






/*
  Main function
*/
int
main(int argc, char * args[]) 
{
  /*
    VARIABLES
  */

  bool  FOUTPUT                 = true;
  bool  SAVE_WAVEFIELD_MATLAB   = false;
  int   IT_DISPLAY              = 50;

  PetscErrorCode ierr;                              // PETSc error code
  DM da;

  // Initialize the PETSc database and MPI
  ierr = PetscInitialize(&argc, &args, NULL, NULL);   CHKERRQ(ierr);
  MPI_Comm comm = PETSC_COMM_WORLD;                 // The global PETSc MPI communicator


  Vec b, *pu;
  Vec *pum1;
  Vec *pum2;
  Vec *pum3;
  PetscScalar *pvel;
  PetscScalar *pdx, *pdy, *pdz, *pxmax, *pymax, *pzmax;
  PetscScalar *pt0, *pdt, *ptmax;
  PetscScalar norm;
  PetscInt *pnx, *pny, *pnz, *pnt;
  PetscInt tmp;

  ctx_t ctx, *pctx;

  clock_t total_time_begin, total_time_end;         
  total_time_begin = clock();                       // Start total time counter


  /*
    LIST OF POINTERS
  */
  pctx = &ctx;
  
  pu = &ctx.wf.u;
  pum1 = &ctx.wf.um1;
  pum2 = &ctx.wf.um2;
  pum3 = &ctx.wf.um3;

  pnx = &ctx.model.nx;
  pny = &ctx.model.ny;
  pnz = &ctx.model.nz;

  pdx = &ctx.model.dx;
  pdy = &ctx.model.dy;
  pdz = &ctx.model.dz;
  
  pxmax = &ctx.model.xmax;
  pymax = &ctx.model.ymax;
  pzmax = &ctx.model.zmax;
  
  pvel = &ctx.model.vel;

  pnt = &ctx.time.nt;
  pt0 = &ctx.time.t0;
  pdt = &ctx.time.dt;
  ptmax = &ctx.time.tmax;

  /*
    CREATE DMDA OBJECT. MESH
  */
  ierr = DMDACreate3d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,  // Create mesh
                      DMDA_STENCIL_STAR, -32, -32, -32, PETSC_DECIDE, PETSC_DECIDE,
                      PETSC_DECIDE, 1, 2, NULL, NULL, NULL, &da);   CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,pnx, pny, pnz, 0,0,0,0,0,0,0,0,0);  CHKERRQ(ierr);          // Get NX, NY, NZ

  /*
    CREATE GLOBAL VEC OBJECTS
  */
  ierr = DMCreateGlobalVector(da, pu);   CHKERRQ(ierr);   // Create a global u vector derived from the DM object
  
  ierr = VecDuplicate(*pu, &b);   CHKERRQ(ierr);          // RHS of the system
  ierr = VecDuplicate(*pu, pum1); CHKERRQ(ierr);          // u at time n-1
  ierr = VecDuplicate(*pu, pum2); CHKERRQ(ierr);          // u at time n-2
  ierr = VecDuplicate(*pu, pum3); CHKERRQ(ierr);          // u at time n-3 

  /*
    SET MODEL PATRAMETERS
  */
  // Wave propagation VELOCITY
  *pvel = 3.5f;
  ierr = PetscOptionsGetReal(NULL, NULL, "-vel",&ctx.model.vel, NULL); CHKERRQ(ierr);   //input on-the-fly
  
  // MODEL SIZE Xmax Ymax Zmax in meters
  *pxmax = 8.f;                     //[km]
  *pymax = 8.f;
  *pzmax = 8.f;
  ierr = PetscOptionsGetReal(NULL, NULL, "-xmax",&ctx.model.xmax, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-ymax",&ctx.model.ymax, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-zmax",&ctx.model.zmax, NULL); CHKERRQ(ierr);

  // GRID STEP DX DY and DZ
  *pdx = *pxmax / *pnx;             //[km]
  *pdy = *pymax / *pny;
  *pdz = *pzmax / *pnz;

  PetscScalar cmax, cmin, lambda_min;

  cmin = *pvel;
  cmax = *pvel;

  // TIME STEPPING PARAMETERS
  *pdt =  (*pdx) / cmax;            //[sec], to have CFL = 1, could be set from runtime
  ierr = PetscOptionsGetReal(NULL, NULL, "-dt",&ctx.time.dt, NULL); CHKERRQ(ierr);

  *ptmax = 1.f;                     //[sec]
  ierr = PetscOptionsGetReal(NULL, NULL, "-tmax",&ctx.time.tmax, NULL); CHKERRQ(ierr);
  
  *pnt = *ptmax / *pdt;

  // SOURCE PARAMETERS
  ctx.src.isrc = (PetscInt) *pnx / 2;
  ctx.src.jsrc = (PetscInt) *pny / 2;
  ctx.src.ksrc = (PetscInt) *pnz / 2;
  ierr = PetscOptionsGetInt(NULL, NULL, "-isrc",&ctx.src.isrc, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-jsrc",&ctx.src.jsrc, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-ksrc",&ctx.src.ksrc, NULL); CHKERRQ(ierr);
  
  
  ctx.src.f0 = 20.f;                //[Hz]
  ierr = PetscOptionsGetReal(NULL, NULL, "-f0",&ctx.src.f0, NULL); CHKERRQ(ierr);

  ctx.src.factor = pow(10.f,7);     //amplitude
  ctx.src.angle_force = 90;         // degrees

  lambda_min = cmin / ctx.src.f0;   // Min wavelength in model

  // RECEIVERS
  ctx.rec.nrec = 40;                // Number of receivers
  ierr = PetscOptionsGetInt(NULL, NULL, "-nrec",&ctx.rec.nrec, NULL); CHKERRQ(ierr);


  PetscInt irec[ctx.rec.nrec], *pirec; // Arrays for rec positions
  PetscInt jrec[ctx.rec.nrec], *pjrec;
  PetscInt krec[ctx.rec.nrec], *pkrec;

  pirec = &irec[0];
  pjrec = &jrec[0];
  pkrec = &krec[0];

  // Place receivers on diogonal
  int i;
  for (i = 0; i < ctx.rec.nrec; i++)
  {
    *(pirec + i) = (PetscInt) (ctx.rec.nrec - i) * (ctx.model.nx) / ctx.rec.nrec;
    *(pjrec + i) = (PetscInt) (ctx.rec.nrec - i) * (ctx.model.ny) / ctx.rec.nrec;
    *(pkrec + i) = (PetscInt) (ctx.rec.nrec - i) * (ctx.model.nz) / ctx.rec.nrec;
  }
  
  ctx.rec.irec = irec;
  ctx.rec.jrec = jrec;
  ctx.rec.krec = krec;

  //Array with seismograms [NREC][NT][2], 2 is for time and displacement colums
  PetscScalar ***seis;              
  seis = f3tensor(0,ctx.rec.nrec,0,*pnt,0,2);
  ctx.rec.seis = seis;


  // OUTPUT
  PetscPrintf(PETSC_COMM_WORLD,"MODEL:\n");
  PetscPrintf(PETSC_COMM_WORLD,"\t XMAX %f \t DX %f km \t NX %i\n", *pxmax, *pdx, *pnx);
  PetscPrintf(PETSC_COMM_WORLD,"\t YMAX %f \t DY %f km \t NY %i\n", *pymax, *pdy, *pny);
  PetscPrintf(PETSC_COMM_WORLD,"\t ZMAX %f \t DZ %f km \t NZ %i\n", *pzmax, *pdz, *pnz);
  PetscPrintf(PETSC_COMM_WORLD,"\t MAX C \t %f km/s \n", cmax);
  PetscPrintf(PETSC_COMM_WORLD,"\t MIN C \t %f km/s \n", cmin);
  PetscPrintf(PETSC_COMM_WORLD,"\n");

  PetscPrintf(PETSC_COMM_WORLD,"SOURCE:\n");
  PetscPrintf(PETSC_COMM_WORLD,"\t ISRC %i \t JSRC %i \t KSRC %i\n", ctx.src.isrc, ctx.src.jsrc, ctx.src.ksrc);
  PetscPrintf(PETSC_COMM_WORLD,"\t F0 \t %f Hz \n", ctx.src.f0);
  PetscPrintf(PETSC_COMM_WORLD,"\t MIN Lambda \t %f km \n", lambda_min);
  PetscPrintf(PETSC_COMM_WORLD,"\t POINTS PER WAvelENGTH \t %f \n", lambda_min/(*pdx));
  PetscPrintf(PETSC_COMM_WORLD,"\n");

  PetscPrintf(PETSC_COMM_WORLD,"RECEIVERS:\n");
  PetscPrintf(PETSC_COMM_WORLD,"\t NREC \t %i\n", ctx.rec.nrec);
  PetscPrintf(PETSC_COMM_WORLD,"\t IREC \t JREC \t KSREC \n");

  // PRINT RECEIVER POSITIONS'
  int rr;
  for (rr = 0; rr < ctx.rec.nrec; rr++)
  {
    PetscPrintf(PETSC_COMM_WORLD,"\t %i \t %i \t %i \n", ctx.rec.irec[rr], ctx.rec.jrec[rr], ctx.rec.krec[rr]);
  }

  PetscPrintf(PETSC_COMM_WORLD,"\n");

  PetscPrintf(PETSC_COMM_WORLD,"TIME STEPPING: \n");
  PetscPrintf(PETSC_COMM_WORLD,"\t TMAX %f \t DT %f \t NT %i\n", *ptmax, *pdt, *pnt);
  PetscPrintf(PETSC_COMM_WORLD,"\n");

  PetscPrintf(PETSC_COMM_WORLD,"CFL CONDITION: \t %f \n", cmax * (*pdt)/(*pdx));
  PetscPrintf(PETSC_COMM_WORLD,"\n");
  
  VecGetSize(*pu, &tmp);
  PetscPrintf(PETSC_COMM_WORLD,"MATRICES AND VECTORS: \n");
  PetscPrintf(PETSC_COMM_WORLD,"\t Vec elements \t %i\n", tmp);
  PetscPrintf(PETSC_COMM_WORLD,"\t Mat \t %i x %i x %i \n", *pnx, *pny, *pnz);
  PetscPrintf(PETSC_COMM_WORLD,"\n");



  /*  
    CREATE KSP, KRYLOV SUBSPACE OBJECTS 
  */
  KSP ksp_u;

  // Create Krylov solver for u component
  ierr = KSPCreate(comm, &ksp_u);   CHKERRQ(ierr);                       // Create the KPS object
  ierr = KSPSetDM(ksp_u, (DM) da);   CHKERRQ(ierr);                      // Set the DM to be used as preconditioner
  ierr = KSPSetComputeOperators(ksp_u, compute_A_u, &ctx);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp_u);   CHKERRQ(ierr);                      // KSP options can be changed during the runtime

  /*
    TIME LOOP
  */
  clock_t begin=clock();
  clock_t end;

  int it;
  int shoot_time;
  for (it  = 1; it <= *pnt; it ++)
  {
    ctx.time.it = it;
    ctx.time.t = (PetscScalar) (it-1) * ctx.time.dt;
    
    ierr = KSPSetComputeRHS(ksp_u, update_b_u, &ctx);   CHKERRQ(ierr);  // new rhs for next iteration
    ierr = KSPSolve(ksp_u, b, *pu);   CHKERRQ(ierr);                    // Solve the linear system using KSP
    
    ierr = Write_seismograms(ksp_u, *pu, &ctx); CHKERRQ(ierr);          // Append value to the seismograms
    
    ierr = VecCopy(*pum2, *pum3);   CHKERRQ(ierr);                      // copy vector um2 to um3
    ierr = VecCopy(*pum1, *pum2);   CHKERRQ(ierr);                      // copy vector um1 to um2
    ierr = VecCopy(*pu, *pum1);   CHKERRQ(ierr);                        // copy vector u to um1



    shoot_time = (int) it%IT_DISPLAY;
    if (FOUTPUT && shoot_time == 0)
    { 
      end = clock();
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Time step: \t %i of %i\n", ctx.time.it, ctx.time.nt);   CHKERRQ(ierr);

      ierr = VecMax(*pu, NULL, &cmax); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "u max: \t %g \n", cmax); CHKERRQ(ierr);
      
      ierr = VecMin(*pu, NULL, &cmin); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "u min: \t %g \n", cmin); CHKERRQ(ierr);

      ierr = VecNorm(*pu,NORM_2,&norm); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "NORM: \t %g \n", norm); CHKERRQ(ierr);

      double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Elapsed time: \t %f sec \n", time_spent); CHKERRQ(ierr);

      if (SAVE_WAVEFIELD_MATLAB)
      {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "./wavefields/tmp_Bvec_%i.m", it);
        ierr = save_Vec_to_m_file(*pu, &buffer); CHKERRQ(ierr);
      }
      
      ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
      begin = clock();
    }
  }

  ierr = Save_seismograms_to_txt_files(ksp_u, pctx);   CHKERRQ(ierr);     // Write seismograms into .txt files

  /*
    CLEAN ALLOCATIONS AND EXIT
  */
  ierr = VecDestroy(&b);     CHKERRQ(ierr);
  ierr = VecDestroy(pu);     CHKERRQ(ierr);
  ierr = VecDestroy(pum1);   CHKERRQ(ierr);
  ierr = VecDestroy(pum2);   CHKERRQ(ierr);
  ierr = VecDestroy(pum3);   CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp_u); CHKERRQ(ierr);
  ierr = DMDestroy(&da);     CHKERRQ(ierr);
  
  // Print out total elapsed time
  total_time_end = clock();
  double time_spent = (double)(total_time_end - total_time_begin) / CLOCKS_PER_SEC;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Total time: \t %f sec \n", time_spent); CHKERRQ(ierr);

  ierr = PetscFinalize();   CHKERRQ(ierr);

  return 0;
}















// APPEND VALUE TO A SEISMOGRAM
PetscErrorCode
Write_seismograms(KSP ksp, Vec u ,void *ctx)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;

  PetscScalar ***_u;

  ctx_t *c = (ctx_t *) ctx;

  DM da;
  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr);           //Get the DM oject of the KSP

  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr);  //Get the global information of the DM grid

  ierr = DMDAVecGetArray(da, u, &_u);   CHKERRQ(ierr);
  
  PetscScalar t = c->time.t; 
  PetscInt it = c->time.it;

  PetscInt nrec = c->rec.nrec;
  PetscInt *irec = c->rec.irec;
  PetscInt *jrec = c->rec.jrec;
  PetscInt *krec = c->rec.krec;

  int xrec;
  for (xrec = 0; xrec < nrec; xrec++)
  {
    if ((irec[xrec] > grid.xs) && (irec[xrec] < (grid.xs + grid.xm)) &&
        (jrec[xrec] > grid.ys) && (jrec[xrec] < (grid.ys + grid.ym)) &&
        (krec[xrec] > grid.zs) && (krec[xrec] < (grid.zs + grid.zm)))
        {
          c->rec.seis[xrec][it-1][0] = t;
          c->rec.seis[xrec][it-1][1] = _u[krec[xrec]][jrec[xrec]][irec[xrec]];
        }
  }
  
  ierr = DMDAVecRestoreArray(da, u, &_u);   CHKERRQ(ierr);

  PetscFunctionReturn(0);
} 






// SAVE VECTOR TO .m FILE
PetscErrorCode
save_Vec_to_m_file(Vec u, void * filename)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  char * filename2 = (char *) filename;

  ierr = PetscPrintf(PETSC_COMM_WORLD, "File created: %s .m \n",filename2); CHKERRQ(ierr);
  PetscViewer viewer;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename2, &viewer);
  PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
  VecView(u, viewer);
  PetscViewerPopFormat(viewer);
  PetscViewerDestroy(&viewer);

  PetscFunctionReturn(0);
}


// SOURCE TERM
PetscErrorCode
source_term(void * ctx)
{
  PetscFunctionBegin;

  PetscScalar f0, t, t0, a, source_term, factor;
  PetscScalar force_x, force_y, force_z, angle_force;
  
  ctx_t *c = (ctx_t *) ctx;

  f0 = c->src.f0;
  t0 = 1.2f / f0;
  t = c->time.t;
  factor = c->src.factor;
  angle_force = c->src.angle_force;

  //add the source (force vector located at a given grid point)
  a = PI*PI*f0*f0;

  //Gaussian
  // source_term = factor * exp(-a * pow((t-t0),2));

  //first derivative of a Gaussian
  // source_term = - factor * 2.d0*a*(t-t0)*exp(-a*(t-t0)**2)

  //Ricker source time function (second derivative of a Gaussian)
  source_term = factor * (1.f - 2.f * a * pow(t-t0,2)) * exp(-a*pow(t-t0,2));

  force_x = sin(angle_force * DEGREES_TO_RADIANS) * source_term;
  force_y = cos(angle_force * DEGREES_TO_RADIANS) * source_term;
  force_z = sin(angle_force * DEGREES_TO_RADIANS) * source_term;

  c->src.fx = force_x;
  c->src.fy = force_y;
  c->src.fz = force_z;

  PetscFunctionReturn(0);
}





// UPDATE RHS AT NEW TIME STEP
PetscErrorCode
update_b_u(KSP ksp, Vec b, void * ctx) 
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscScalar dt2;

  ctx_t *c = (ctx_t *) ctx;

  source_term(c);
  dt2 = pow(c->time.dt,2);

  DM da;
  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr); //Get the DM oject of the KSP

  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr); //Get the global information of the DM grid

  PetscScalar hx = c->model.dx;
  PetscScalar hy = c->model.dy;
  PetscScalar hz = c->model.dz;
  
  double *** _b;
  double *** _um1, ***_um2, ***_um3;

  ierr = DMDAVecGetArray(da, b, &_b);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.um1, &_um1);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.um2, &_um2);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.um3, &_um3);   CHKERRQ(ierr);
  
  //  Fill b
  double f, source_term;
  unsigned int k;
  for(k = grid.zs; k < (grid.zs + grid.zm); k++)      // Depth
  {
    unsigned int j;
    for(j = grid.ys; j < (grid.ys + grid.ym); j++)    // Columns
    {
      unsigned int i;
      for(i = grid.xs; i < (grid.xs + grid.xm); i++)  // Rows
      {
        // Nodes on the boundary layers
        if((i == 0) || (i == (grid.mx - 1)) ||
          (j == 0) || (j == (grid.my - 1)) ||
          (k == 0) || (k == (grid.mz - 1)))
        {
          _b[k][j][i] = 0.f;
        }
        //Interior nodes
        else 
        { 
          if ((i==c->src.isrc) && (j==c->src.jsrc) && (k==c->src.ksrc))
          {
            source_term = c->src.fx;
          }
          else
          {
            source_term = 0.f;
          }

          f = hx * hy * hz *
          (5.f * _um1[k][j][i] - 4.f * _um2[k][j][i] + 1.f * _um3[k][j][i] + dt2 * source_term);

          _b[k][j][i] = f;
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, b, &_b);             CHKERRQ(ierr);   // Release the resource
  ierr = DMDAVecRestoreArray(da, c->wf.um1, &_um1);   CHKERRQ(ierr);   // Release the resource
  ierr = DMDAVecRestoreArray(da, c->wf.um2, &_um2);   CHKERRQ(ierr);   // Release the resource
  ierr = DMDAVecRestoreArray(da, c->wf.um3, &_um3);   CHKERRQ(ierr);   // Release the resource

  // FIX NULLSPACE-CAUSED PROBLEMS
  MatNullSpace   nullspace;
  MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);
  MatNullSpaceRemove(nullspace,b);
  MatNullSpaceDestroy(&nullspace);
  
  PetscFunctionReturn(0);
}



// BUILD MATRIX A
PetscErrorCode 
compute_A_u(KSP ksp, Mat A, Mat J, void * ctx)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;
  PetscScalar v[13], hx, hy, hz, hyhzdhx, hxhzdhy, hxhydhz;
  PetscScalar dt, dt2;
  PetscScalar vel, vel2;  
  PetscInt n;
  DM da;
  DMDALocalInfo grid;
  MatStencil idxm;  //A PETSc data structure to store information about a single row or column in the stencil
  MatStencil idxn[13];
  
  ctx_t *c = (ctx_t *) ctx;

  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr);             // Get the DMDA object
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr);    // Get the grid information

  vel = c->model.vel;
  vel2 = pow(vel, 2);

  dt = c->time.dt;
  dt2 = dt * dt;

  hx = c->model.dx;
  hy = c->model.dy;
  hz = c->model.dz;

  hyhzdhx = hy * hz / (12.f * hx);
  hxhzdhy = hx * hz / (12.f * hy);
  hxhydhz = hx * hy / (12.f * hz);

  /* Loop over the grid points */
  unsigned int k;
  for(k = grid.zs; k < (grid.zs + grid.zm); k++)          // Depth 
  {
    unsigned int j;
    for(j = grid.ys; j < (grid.ys + grid.ym); j++)        // Columns 
    {
      unsigned int i;
      for(i = grid.xs; i < (grid.xs + grid.xm); i++)      // Rows
      { 
        n = 1;
        idxm.k = k;
        idxm.j = j;
        idxm.i = i;
        
        idxn[0].k = k;
        idxn[0].j = j;
        idxn[0].i = i;

        // Nodes on the boundary layers
        if((i == 0) || (i == (grid.mx - 1)) ||
          (j == 0) || (j == (grid.my - 1)) ||
          (k == 0) || (k == (grid.mz - 1)))
        {
          v[0]=1.f;
        }
        // Interior nodes
        else 
        {
          v[0] = 30.f * vel2 * dt2 * (hyhzdhx + hxhzdhy + hxhydhz);
        // If neighbor is not a known boundary value
        // then we put an entry

        if((i - 2) > 0) 
        {
          idxn[n].j = j;                // Get the column indices
          idxn[n].i = i - 2;
          idxn[n].k = k;

          v[n] = vel2 *  dt2 * hyhzdhx; // Fill with the value      
          n++;                          // One column added

          idxn[n].j = j;
          idxn[n].i = i - 1;
          idxn[n].k = k;

          v[n] = - 16.f * vel2 *  dt2 * hyhzdhx;
          n++;
        }
        if((i + 2) < (grid.mx - 1)) 
        {
          idxn[n].j = j;
          idxn[n].i = i + 2;
          idxn[n].k = k;

          v[n] = vel2 * dt2 * hyhzdhx;
          n++;

          idxn[n].j = j;
          idxn[n].i = i + 1;
          idxn[n].k = k;

          v[n] = - 16.f * vel2 * dt2 * hyhzdhx;
          n++;
        }
        if((j - 2) > 0) 
        {
          idxn[n].j = j - 2;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = vel2 * dt2 * hxhzdhy;
          n++;

          idxn[n].j = j - 1;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = - 16.f * vel2 * dt2 * hxhzdhy;
          n++;
        }
        if((j + 2) < (grid.my - 1)) 
        {
          idxn[n].j = j + 2;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = vel2 * dt2 * hxhzdhy;
          n++;

          idxn[n].j = j + 1;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = - 16.f * vel2 * dt2 * hxhzdhy;
          n++;
        }

        if((k - 2) > 0) 
        {
          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k - 2;

          v[n] = vel2 * dt2 * hxhydhz;
          n++;

          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k - 1;

          v[n] = - 16.f * vel2 * dt2 * hxhydhz;
          n++;
        }

        if((k + 2) < (grid.mz - 1)) 
        {
          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k + 2;

          v[n] = vel2 * dt2 * hxhydhz;
          n++;

          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k + 1;

          v[n] = - 16.f * vel2 * dt2 * hxhydhz;
          n++;
        }
      
        v[0]+= 2.f * hx * hy * hz;
      }
      
      // Insert one row of the matrix A
      ierr = MatSetValuesStencil(A, 1, (const MatStencil *) &idxm, 
                                (PetscInt) n, (const MatStencil *) &idxn,
                                (PetscScalar *) v, INSERT_VALUES);       CHKERRQ(ierr);
      }
    }
  }
  
  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);   CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A ,MAT_FINAL_ASSEMBLY);   CHKERRQ(ierr);

  PetscFunctionReturn(0);
}






// This function allocates memory for a 3D array. The function is taken from SOFI3D_acoustic
//https://git.scc.kit.edu/GPIAG-Software/SOFI3D/tree/0ca72edf3ef977813372dd26ccfeaf4c19361a69

PetscScalar ***f3tensor(PetscInt nrl, PetscInt nrh, PetscInt ncl, PetscInt nch,PetscInt ndl, PetscInt ndh)
{
  PetscFunctionBegin;

	/* allocate a float 3tensor with subscript range m[nrl..nrh][ncl..nch][ndl..ndh]
		   and intializing the matrix, e.g. m[nrl..nrh][ncl..nch][ndl..ndh]=0.0 */
	PetscInt i,j,d, nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1, NR_END=1;
	PetscScalar ***t;

	/* allocate pointers to pointers to rows */
	t=(PetscScalar ***) malloc((size_t) ((nrow+NR_END)*sizeof(PetscScalar**)));
	// if (!t) err("allocation failure 1 in function f3tensor() ");
	t += NR_END;
	t -= nrl;

	/* allocate pointers to rows and set pointers to them */
	t[nrl]=(PetscScalar **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(PetscScalar*)));
	// if (!t[nrl]) err("allocation failure 2 in function f3tensor() ");
	t[nrl] += NR_END;
	t[nrl] -= ncl;

	/* allocate rows and set pointers to them */
	t[nrl][ncl]=(PetscScalar *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(PetscScalar)));
	t[nrl][ncl] += NR_END;
	t[nrl][ncl] -= ndl;

	for (j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
	for (i=nrl+1;i<=nrh;i++){
		t[i]=t[i-1]+ncol;
		t[i][ncl]=t[i-1][ncl]+ncol*ndep;
		for (j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
	}

	/* initializing 3tensor */
	for (i=nrl;i<=nrh;i++)
		for (j=ncl;j<=nch;j++)
			for (d=ndl;d<=ndh;d++) t[i][j][d]=0.0;

	/* return pointer to array of pointer to rows */
  PetscFunctionReturn(t);
}



// WRITE DOWN FILES WITH SEISMOGRAMS
PetscErrorCode
Save_seismograms_to_txt_files(KSP ksp, void *ctx) 
{
  PetscFunctionBegin;

  PetscErrorCode ierr;
  
  ctx_t *c = (ctx_t *) ctx;

  PetscInt nrec = c->rec.nrec;
  PetscInt nt = c->time.nt;

  PetscInt *irec = c->rec.irec;
  PetscInt *jrec = c->rec.jrec;
  PetscInt *krec = c->rec.krec;

  DM da;
  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr);           //Get the DM oject of the KSP

  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr); //Get the global information of the DM grid
  
  int xrec;
  for (xrec = 0; xrec < nrec; xrec++)
  {
    if ((irec[xrec] > grid.xs) && (irec[xrec] < (grid.xs + grid.xm)) &&
        (jrec[xrec] > grid.ys) && (jrec[xrec] < (grid.ys + grid.ym)) &&
        (krec[xrec] > grid.zs) && (krec[xrec] < (grid.zs + grid.zm)))
        {
          char buffer[64];
          snprintf(buffer, sizeof(buffer), "./seism/seis_%i_%i_%i_%i_%i_%i.txt", 
          xrec, c->rec.irec[xrec], c->rec.jrec[xrec], c->rec.krec[xrec], (int) c->src.f0, (int) c-> model.xmax);

          FILE *fout = fopen(buffer, "wb");     

          int i;
          for (i = 0; i < nt ; i++)
          {
            fprintf(fout, "%f \t %f \n", c->rec.seis[xrec][i][0], c->rec.seis[xrec][i][1]);
          }            
          fclose(fout); 
        }
  }

  PetscFunctionReturn(0);
}
