
/*
  Author: Oleg Ovcharenko, PhD student at ErSE, KAUST
  Email:  oleg.ovcharenko@kaust.edu.sa

  ps4.c: A PETSc example that solves a 2D linear Poisson equation
*/

#include <stdio.h>
#include <petscdmda.h>
#include <petscksp.h>
#include <math.h>
#include <time.h>

#define debprint(expr) PetscPrintf(PETSC_COMM_WORLD, #expr " = %f \n", expr);
#define PI 3.1415926535
#define DEGREES_TO_RADIANS PI/180.f

//User-functions prototypes
PetscErrorCode compute_A_ux(KSP, Mat, Mat, void *);
PetscErrorCode update_b_ux(KSP, Vec, void *);
PetscErrorCode save_Vec_to_m_file(Vec, void *);
PetscErrorCode Save_seismograms_to_txt_files(void *);
PetscErrorCode source_term(void *);
PetscErrorCode Write_seismograms(KSP, Vec, void *);
PetscScalar    ***f3tensor(PetscInt, PetscInt, PetscInt, PetscInt,PetscInt, PetscInt);


// User-defined structures
// Wavefield
typedef struct {
  Vec ux;
  Vec uxm1; 
  Vec uxm2; 
  Vec uxm3; 

} wfield;

// Model parameters
typedef struct {
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  PetscScalar dx;
  PetscScalar dy;
  PetscScalar dz;
  PetscScalar xmax;
  PetscScalar xmin;
  PetscScalar ymax;
  PetscScalar ymin;
  PetscScalar zmax;
  PetscScalar zmin;
  Vec c11;
  Vec rho;
} model_par;

typedef struct{
  PetscScalar dt;
  PetscScalar t0;
  PetscScalar tmax;
  PetscScalar t;
  PetscInt it;
  PetscInt nt;
} time_par;

typedef struct{
  PetscInt isrc;
  PetscInt jsrc;
  PetscInt ksrc;
  PetscScalar factor;         // ampliturde
  PetscScalar angle_force;
  PetscScalar f0;             // frequency
  PetscScalar fx;             // force x
  PetscScalar fy;             // force y
  PetscScalar fz;             // force z
} source;

typedef struct
{
  PetscInt nrec;
  PetscInt *irec;
  PetscInt *jrec;
  PetscInt *krec;
  PetscScalar ***seis;
} receivers;

// User context to use with PETSc functions
typedef struct {
  wfield wf;
  model_par model;
  time_par time;
  source src;
  receivers rec;
} ctx_t;


typedef int bool;
#define true 1
#define false 0








/*
  Main C function
*/
int
main(int argc, char * args[]) 
{
  /*
    VARIABLES
  */
  PetscErrorCode ierr; // PETSc error code
  DM da;

  Vec b, *pux;
  Vec *puxm1;
  Vec *puxm2;
  Vec *puxm3;

  Vec *pc11, *prho;
  PetscScalar *pdx, *pdy, *pdz, *pxmax, *pymax, *pzmax;
  PetscScalar *pt0, *pdt, *ptmax;
  PetscInt *pnx, *pny, *pnz, *pnt;
  PetscScalar norm;

  bool FOUTPUT                = true;
  bool SAVE_WAVEFIELD_MATLAB  = false;

  clock_t total_time_begin, total_time_end;
  total_time_begin = clock();

  ctx_t ctx, *pctx;

  PetscInt tmp;


  ierr = PetscInitialize(&argc, &args, NULL, NULL);   CHKERRQ(ierr);   // Initialize the PETSc database and MPIeo
  MPI_Comm comm = PETSC_COMM_WORLD;   // The global PETSc MPI communicator


  /*
    LIST OF POINTERS
  */
  pctx = &ctx;
  
  pux = &ctx.wf.ux;
  puxm1 = &ctx.wf.uxm1;
  puxm2 = &ctx.wf.uxm2;
  puxm3 = &ctx.wf.uxm3;

  pnx = &ctx.model.nx;
  pny = &ctx.model.ny;
  pnz = &ctx.model.nz;

  pdx = &ctx.model.dx;
  pdy = &ctx.model.dy;
  pdz = &ctx.model.dz;
  
  pxmax = &ctx.model.xmax;
  pymax = &ctx.model.ymax;
  pzmax = &ctx.model.zmax;
  
  pc11 = &ctx.model.c11;
  prho = &ctx.model.rho;

  pnt = &ctx.time.nt;
  pt0 = &ctx.time.t0;
  pdt = &ctx.time.dt;
  ptmax = &ctx.time.tmax;

  /*
    CREATE DMDA OBJECT. MESH
  */
  ierr = DMDACreate3d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,  // Create mesh
                      DMDA_STENCIL_STAR, -25, -25, -25, PETSC_DECIDE, PETSC_DECIDE,
                      PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da);   CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,pnx, pny, pnz, 0,0,0,0,0,0,0,0,0);  CHKERRQ(ierr); // Get NX, Ny, NZ

  /*
    CREATE VEC OBJECTS
  */
  ierr = DMCreateGlobalVector(da, pux);   CHKERRQ(ierr);  // Create a global ux vector derived from the DM object
  
  ierr = VecDuplicate(*pux, &b);   CHKERRQ(ierr);         // RHS of the system
  ierr = VecDuplicate(*pux, pc11);   CHKERRQ(ierr);       // Duplicate vectors from grid object to each vector component
  ierr = VecDuplicate(*pux, prho);   CHKERRQ(ierr); 
  ierr = VecDuplicate(*pux, puxm1); CHKERRQ(ierr);        // ux at time n-1
  ierr = VecDuplicate(*pux, puxm2); CHKERRQ(ierr);        // ux at time n-2
  ierr = VecDuplicate(*pux, puxm3); CHKERRQ(ierr);        // ux at time n-3 

  ierr = VecSet(*pc11, 2.0f); CHKERRQ(ierr);            // c velocity, m/sec
  ierr = VecSet(*prho, 1.f); CHKERRQ(ierr);            // kg/m3


  /*----------------------------------------------------------------------
    SET MODEL PATRAMETERS
  ------------------------------------------------------------------------*/
  // MODEL SIZE Xmax Ymax Zmax in meters
  *pxmax = 1.f; //[km]
  *pymax = 1.f;
  *pzmax = 1.f;

  // GRID STEP DX DY and DZ
  *pdx = *pxmax / *pnx; //[km]
  *pdy = *pymax / *pny;
  *pdz = *pzmax / *pnz;

  PetscScalar cmax, cmin, lambda_min;

  VecMax(ctx.model.c11, NULL, &cmax);
  VecMin(ctx.model.c11, NULL, &cmin);

  // TIME STEPPING PARAMETERS
  *pdt =  (*pdx) / cmax; //[sec]
  // *pdt = 0.001f;
  *ptmax = 0.5f; //[sec]
  *pnt = *ptmax / *pdt;

  // SOURCE PARAMETERS
  ctx.src.isrc = (PetscInt) *pnx / 2;
  ctx.src.jsrc = (PetscInt) *pny / 2;
  ctx.src.ksrc = (PetscInt) *pnz / 2;
  ctx.src.f0 = 20.f; //[Hz]
  ctx.src.factor = pow(10.f,8); //amplitude
  ctx.src.angle_force = 90; // degrees

  lambda_min = cmin / ctx.src.f0;                 // Min wavelength in model

  // RECEIVERS

  ctx.rec.nrec = 40;

  // PetscInt irec[]={ctx.src.isrc, (PetscInt) ctx.src.isrc/2, (PetscInt) ctx.src.isrc/4};
  // PetscInt jrec[]={ctx.src.jsrc, ctx.src.jsrc, ctx.src.jsrc};
  // PetscInt krec[]={ctx.src.ksrc, ctx.src.ksrc, ctx.src.ksrc}; 

  PetscInt irec[ctx.rec.nrec], *pirec;
  PetscInt jrec[ctx.rec.nrec], *pjrec;
  PetscInt krec[ctx.rec.nrec], *pkrec;

  pirec = &irec[0];
  pjrec = &jrec[0];
  pkrec = &krec[0];

  int i;
  for (i = 0; i < ctx.rec.nrec; i++)
  {
    *(pirec + i) = (PetscInt) (ctx.rec.nrec - i) * (ctx.src.isrc) / ctx.rec.nrec;
    *(pjrec + i) = (PetscInt) (ctx.rec.nrec - i) * (ctx.src.jsrc) / ctx.rec.nrec;
    *(pkrec + i) = (PetscInt) (ctx.rec.nrec - i) * (ctx.src.ksrc) / ctx.rec.nrec;
  }
  
  ctx.rec.irec = irec;
  ctx.rec.jrec = jrec;
  ctx.rec.krec = krec;
  // ctx.rec.nrec = (PetscInt) sizeof(irec)/sizeof(irec[0]);

  PetscScalar ***seis; //Array with seismograms [NREC][NT][2], 2 is for time and displacement colums
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
  PetscPrintf(PETSC_COMM_WORLD,"\t POINTS PER WAVELENGTH \t %f \n", lambda_min/(*pdx));
  PetscPrintf(PETSC_COMM_WORLD,"\n");

  PetscPrintf(PETSC_COMM_WORLD,"RECEIVERS:\n");
  PetscPrintf(PETSC_COMM_WORLD,"\t NREC \t %i\n", ctx.rec.nrec);
  PetscPrintf(PETSC_COMM_WORLD,"\t IREC \t JREC \t KSREC \n");

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
  
  VecGetSize(*pux, &tmp);
  PetscPrintf(PETSC_COMM_WORLD,"MATRICES AND VECTORS: \n");
  PetscPrintf(PETSC_COMM_WORLD,"\t Vec elements \t %i\n", tmp);
  PetscPrintf(PETSC_COMM_WORLD,"\t Mat \t %i x %i x %i \n", *pnx, *pny, *pnz);
  PetscPrintf(PETSC_COMM_WORLD,"\n");



  /*  
    CREATE KSP, KRYLOV SUBSPACE OBJECTS 
  */
  KSP ksp_ux;

  // Create Krylov solver for ux component
  ierr = KSPCreate(comm, &ksp_ux);   CHKERRQ(ierr);                       // Create the KPS object
  ierr = KSPSetDM(ksp_ux, (DM) da);   CHKERRQ(ierr);                      // Set the DM to be used as preconditioner
  ierr = KSPSetComputeOperators(ksp_ux, compute_A_ux, &ctx);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp_ux);   CHKERRQ(ierr);                      // KSP options can be changed during the runtime

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
    
    ierr = KSPSetComputeRHS(ksp_ux, update_b_ux, &ctx);   CHKERRQ(ierr);  // new rhs for next iteration
    ierr = KSPSolve(ksp_ux, b, *pux);   CHKERRQ(ierr);                    // Solve the linear system using KSP
    
    ierr = Write_seismograms(ksp_ux, *pux, &ctx); CHKERRQ(ierr);
    
    ierr = VecCopy(*puxm2, *puxm3);   CHKERRQ(ierr);                      // copy vector um2 to um3
    ierr = VecCopy(*puxm1, *puxm2);   CHKERRQ(ierr);                      // copy vector um1 to um2
    ierr = VecCopy(*pux, *puxm1);   CHKERRQ(ierr);                        // copy vector u to um1



    shoot_time = (int) it%10;
    if (FOUTPUT && shoot_time == 0)
    { 
      end = clock();
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Time step: \t %i of %i\n", ctx.time.it, ctx.time.nt);   CHKERRQ(ierr);

      ierr = VecMax(*pux, NULL, &cmax); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "UX max: \t %g \n", cmax); CHKERRQ(ierr);
      
      ierr = VecMin(*pux, NULL, &cmin); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "UX min: \t %g \n", cmin); CHKERRQ(ierr);

      ierr = VecNorm(*pux,NORM_2,&norm); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "NORM: \t %g \n", norm); CHKERRQ(ierr);

      double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Elapsed time: \t %f sec \n", time_spent); CHKERRQ(ierr);

      if (SAVE_WAVEFIELD_MATLAB)
      {
        char buffer[32];                                 // The filename buffer.
        snprintf(buffer, sizeof(buffer), "./wavefields/tmp_Bvec_%i.m", it);
        ierr = save_Vec_to_m_file(*pux, &buffer); CHKERRQ(ierr);
      }
      
      ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
      begin = clock();
    }
  }

  ierr = Save_seismograms_to_txt_files(pctx);   CHKERRQ(ierr);

  /*
    CLEAN ALLOCATIONS AND EXIT
  */
  ierr = VecDestroy(&b);   CHKERRQ(ierr);
  ierr = VecDestroy(pux);   CHKERRQ(ierr);
  ierr = VecDestroy(puxm1);   CHKERRQ(ierr);
  ierr = VecDestroy(puxm2);   CHKERRQ(ierr);
  ierr = VecDestroy(puxm3);   CHKERRQ(ierr);
  ierr = VecDestroy(pc11);   CHKERRQ(ierr);
  ierr = VecDestroy(prho);   CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp_ux);   CHKERRQ(ierr);
  ierr = DMDestroy(&da);   CHKERRQ(ierr);
  
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
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr); //Get the global information of the DM grid

  ierr = DMDAVecGetArray(da, u, &_u);   CHKERRQ(ierr);
  
  PetscScalar t = c->time.t; 
  PetscInt it = c->time.it;

  PetscInt nrec = c->rec.nrec;
  PetscInt *irec = c->rec.irec;
  PetscInt *jrec = c->rec.jrec;
  PetscInt *krec = c->rec.krec;
 
  // PetscPrintf(PETSC_COMM_WORLD, " %i %i \n %i %i \n %i %i \n \n", 
  //                                                     grid.xs, grid.xs + grid.xm,
  //                                                     grid.ys, grid.ys + grid.ym,
  //                                                     grid.zs, grid.zs + grid.zm);

  int xrec;
  for (xrec = 0; xrec < nrec; xrec++)
  {
    if ((irec[xrec] > grid.xs) && (irec[xrec] < (grid.xs + grid.xm)) &&
        (jrec[xrec] > grid.ys) && (jrec[xrec] < (grid.ys + grid.ym)) &&
        (krec[xrec] > grid.zs) && (krec[xrec] < (grid.zs + grid.zm)))
        {
  // PetscPrintf(PETSC_COMM_WORLD, "%i\n %i %i %i \n %i %i %i \n %i %i %i \n \n", nrec, 
  //                                                     irec[xrec], grid.xs, grid.xs + grid.xm,
  //                                                     jrec[xrec], grid.ys, grid.ys + grid.ym,
  //                                                     krec[xrec], grid.zs, grid.zs + grid.zm);
          c->rec.seis[xrec][it-1][0] = t;
          c->rec.seis[xrec][it-1][1] = _u[krec[xrec]][jrec[xrec]][irec[xrec]];
          // ierr = PetscPrintf(PETSC_COMM_WORLD, "!!!!! %i u \t %f \n", xrec, _u[krec[xrec]][jrec[xrec]][irec[xrec]]); CHKERRQ(ierr);
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





// UPDATE RHS OF UX EQUATION
PetscErrorCode
update_b_ux(KSP ksp, Vec b, void * ctx) 
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

  double hx = (1.f / (double) (grid.mx - 1));
  double hy = (1.f / (double) (grid.my - 1));
  double hz = (1.f / (double) (grid.mz - 1));
  
  double *** _b;
  double *** _uxm1, ***_uxm2, ***_uxm3, ***_rho;

  ierr = DMDAVecGetArray(da, b, &_b);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.uxm1, &_uxm1);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.uxm2, &_uxm2);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.uxm3, &_uxm3);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->model.rho, &_rho);   CHKERRQ(ierr);

  /*
    Loop over the grid points, and fill b 
  */
  double f, source_term;
  unsigned int k;
  for(k = grid.zs; k < (grid.zs + grid.zm); k++) // Depth
  {
    unsigned int j;
    for(j = grid.ys; j < (grid.ys + grid.ym); j++) // Columns
    {
      unsigned int i;
      for(i = grid.xs; i < (grid.xs + grid.xm); i++) // Rows
      {
        /* Nodes on the boundary layers (\Gamma) */
        if((i == 0) || (i == (grid.mx - 1)) ||
          (j == 0) || (j == (grid.my - 1)) ||
          (k == 0) || (k == (grid.mz - 1)))
        {
          _b[k][j][i] = 0.f;
        }
        /* Interior nodes in the domain (\Omega) */
        else 
        { 
          if ((i==c->src.isrc) && (j==c->src.jsrc) && (k==c->src.ksrc))
          {
            source_term = dt2 / _rho[k][j][i] * (c->src.fx);
          }
          else
          {
            source_term = 0.f;
          }

          f = hx * hy * hz *
          (5.f * _uxm1[k][j][i] - 4.f * _uxm2[k][j][i] + 1.f * _uxm3[k][j][i] + source_term);

          _b[k][j][i] = f;
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, b, &_b);   CHKERRQ(ierr);   // Release the resource
  ierr = DMDAVecRestoreArray(da, c->wf.uxm1, &_uxm1);   CHKERRQ(ierr);   // Release the resource
  ierr = DMDAVecRestoreArray(da, c->wf.uxm2, &_uxm2);   CHKERRQ(ierr);   // Release the resource
  ierr = DMDAVecRestoreArray(da, c->wf.uxm3, &_uxm3);   CHKERRQ(ierr);   // Release the resource

  MatNullSpace   nullspace;
  MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);
  MatNullSpaceRemove(nullspace,b);
  MatNullSpaceDestroy(&nullspace);
  
  PetscFunctionReturn(0);
}




/*=========================================================================================
 Using example 34 from
 http://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex34.c.html \
  =========================================================================================*/

// COMPUTE OPERATOR A FOR UX EQUATION
PetscErrorCode 
compute_A_ux(KSP ksp, Mat A, Mat J, void * ctx)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;
  PetscScalar v[7], hx, hy, hz, hyhzdhx, hxhzdhy, hxhydhz;
  PetscScalar dt, dt2;
  double ***rho, ***c11;
  PetscInt n;
  DM da;
  DMDALocalInfo grid;
  MatStencil idxm;   /*  A PETSc data structure to store information about a single row or column in the stencil */
  MatStencil idxn[7];
  
  ctx_t *c = (ctx_t *) ctx;

  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr);   // Get the DMDA object
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr);   // Get the grid information

  ierr = DMDAVecGetArray(da, c->model.c11, &c11);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->model.rho, &rho);   CHKERRQ(ierr);
  
  dt = c->time.dt;
  dt2 = dt * dt;

  hx = (1.f / (double) (grid.mx - 1));
  hy = (1.f / (double) (grid.my - 1));
  hz = (1.f / (double) (grid.mz - 1));

  hyhzdhx = hy * hz / hx;
  hxhzdhy = hx * hz / hy;
  hxhydhz = hx * hy / hz;

  /* Loop over the grid points */
  unsigned int k;
  for(k = grid.zs; k < (grid.zs + grid.zm); k++) // Depth 
  {
    unsigned int j;
    for(j = grid.ys; j < (grid.ys + grid.ym); j++) // Columns 
    {
      unsigned int i;
      for(i = grid.xs; i < (grid.xs + grid.xm); i++)  // Rows
      { 
        n = 1;
        idxm.k = k;
        idxm.j = j;
        idxm.i = i;
        
        idxn[0].k = k;
        idxn[0].j = j;
        idxn[0].i = i;

        /* Nodes on the boundary layers (\Gamma) */
        if((i == 0) || (i == (grid.mx - 1)) ||
          (j == 0) || (j == (grid.my - 1)) ||
          (k == 0) || (k == (grid.mz - 1)))
        {
          v[0]=1.f;
        }
        /* Interior nodes in the domain (\Omega) */
        else 
        {
          v[0] = pow(c11[k][j][i],2) * dt2/rho[k][j][i] * 2.f * (hyhzdhx + hxhzdhy + hxhydhz);
        // If neighbor is not a known boundary value
        // then we put an entry

        if((i - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i - 1;
          idxn[n].k = k;

          v[n] = - pow(c11[k][j][i],2) *  dt2/rho[k][j][i] * hyhzdhx; // Fill with the value
          
          n++; // One column added
        }
        if((i + 1) < (grid.mx - 1)) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i + 1;
          idxn[n].k = k;

          v[n] = - pow(c11[k][j][i],2) * dt2/rho[k][j][i] * hyhzdhx;  // Fill with the value

          n++; // One column added
        }
        if((j - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j - 1;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = - pow(c11[k][j][i],2) * dt2/rho[k][j][i] * hxhzdhy; // Fill with the value
  
          n++; // One column added
        }
        if((j + 1) < (grid.my - 1)) 
        {
          // Get the column indices
          idxn[n].j = j + 1;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = - pow(c11[k][j][i],2) * dt2/rho[k][j][i] * hxhzdhy; // Fill with the value

          n++; // One column added
        }

        if((k - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k - 1;

          v[n] = - pow(c11[k][j][i],2) * dt2/rho[k][j][i] * hxhydhz; // Fill with the value

          n++; // One column added
        }

        if((k + 1) < (grid.mz - 1)) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k + 1;

          v[n] = - pow(c11[k][j][i],2) * dt2/rho[k][j][i] * hxhydhz; // Fill with the value

          n++; // One column added
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

  ierr = DMDAVecRestoreArray(da, c->model.c11, &c11);   CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, c->model.rho, &rho);   CHKERRQ(ierr);

  PetscFunctionReturn(0);
}







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



PetscErrorCode
Save_seismograms_to_txt_files(void *ctx) 
{
  PetscFunctionBegin;
  
  ctx_t *c = (ctx_t *) ctx;

  PetscInt nrec = c->rec.nrec;
  PetscInt nt = c->time.nt;
  
  int xrec;
  for (xrec = 0; xrec < nrec; xrec++)
  {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "./seism/seis_%i_%i_%i_%i.txt", 
            xrec, c->rec.irec[xrec], c->rec.jrec[xrec], c->rec.krec[xrec]);

    FILE *fout = fopen(buffer, "wb");     

    int i;
    for (i = 0; i < nt ; i++)
    {
      fprintf(fout, "%f \t %f \n", c->rec.seis[xrec][i][0], c->rec.seis[xrec][i][1]);
    }            
    fclose(fout); 
  }

  PetscFunctionReturn(0);
}