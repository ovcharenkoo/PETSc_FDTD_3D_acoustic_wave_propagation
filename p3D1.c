
/*
  Author: Mohammed A. Al Farhan, PhD student at ECRC, KAUST
  Email:  mohammed.farhan@kaust.edu.sa

  ps4.c: A PETSc example that solves a 2D linear Poisson equation

  The example code is based upon:
    1) "PETSC FOR PARTIAL DIFFERENTIAL EQUATIONS" book by
        Professor ED BUELER:
        https://github.com/bueler/p4pdes/releases
    2) PETSc KSP Example 50 (ex50):
        $PETSC_DIR/src/ksp/ksp/examples/tutorials/ex50.c
    3) PETSc KSP Example 15 (ex15):
        $PETSC_DIR/src/ksp/ksp/examples/tutorials/ex15.c
*/

#include <stdio.h>
#include <petscdmda.h>
#include <petscksp.h>

/*#define debprint(expr) printf(#expr " = %f\n", expr)*/
#define debprint(expr) PetscPrintf(PETSC_COMM_WORLD, #expr " = %g \n", expr);

//User-functions prototypes
PetscErrorCode compute_A_ux(KSP, Mat, Mat, void *);
PetscErrorCode compute_A_uy(KSP, Mat, Mat, void *); 
PetscErrorCode compute_A_uz(KSP, Mat, Mat, void *);
PetscErrorCode compute_b_ux(KSP, Vec, void *);
PetscErrorCode update_b_ux(KSP, Vec, void *);
PetscErrorCode save_wavefield_to_m_file(Vec, void *);


// User-defined structures
// Wavefield
typedef struct {
  Vec ux;
  Vec uy;
  Vec uz;
  Vec u; 
  
  Vec uxm1; 
  Vec uxm2; 
  Vec uxm3; 
  
  Vec uym1; 
  Vec uym2; 
  Vec uym3; 
  
  Vec uzm1; 
  Vec uzm2; 
  Vec uzm3; 
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
  Vec c12;
  Vec c13;
  Vec c44;
  Vec c66;
  Vec rho;
} model_par;

typedef struct{
  PetscScalar dt;
  PetscScalar t0;
  PetscInt nt;
  PetscScalar tf;
} time_par;

typedef struct{
  PetscInt isrc;
  PetscInt jsrc;
  PetscInt ksrc;
  PetscScalar f0;
} source;

// User context to use with PETSc functions
typedef struct {
  wfield wf;
  model_par model;
  time_par time;
} ctx_t;








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

  Vec b, *pux, *puy, *puz;
  Vec *puxm1, *puym1, *puzm1;
  Vec *puxm2, *puym2, *puzm2;
  Vec *puxm3, *puym3, *puzm3;

  Vec *pc11, *pc12, *pc13, *pc44, *pc66, *prho;

  PetscScalar *pdx, *pdy, *pdz, *pxmax, *pymax, *pzmax;

  PetscInt *pnx, *pny, *pnz, *pnt;

  PetscScalar *pt0, *pdt, *ptf;

  ctx_t ctx, *pctx;
  PetscInt tmp;


  /*
    LIST OF POINTERS
  */
  pctx = &ctx;
  
  pux = &ctx.wf.ux;
  puy = &ctx.wf.uy;
  puz = &ctx.wf.uz;

  puxm1 = &ctx.wf.uxm1;
  puym1 = &ctx.wf.uym1;
  puzm1 = &ctx.wf.uzm1;

  puxm2 = &ctx.wf.uxm2;
  puym2 = &ctx.wf.uym2;
  puzm2 = &ctx.wf.uzm2;

  puxm3 = &ctx.wf.uxm3;
  puym3 = &ctx.wf.uym3;
  puzm3 = &ctx.wf.uzm3;

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
  pc12 = &ctx.model.c12;
  pc13 = &ctx.model.c13;
  pc44 = &ctx.model.c44;
  pc66 = &ctx.model.c66;
  prho = &ctx.model.rho;

  pnt = &ctx.time.nt;
  pt0 = &ctx.time.t0;
  pdt = &ctx.time.dt;
  ptf = &ctx.time.tf;



  ierr = PetscInitialize(&argc, &args, NULL, NULL);   CHKERRQ(ierr);   // Initialize the PETSc database and MPIeo
  MPI_Comm comm = PETSC_COMM_WORLD;   // The global PETSc MPI communicator






  /*
    CREATE DMDA OBJECT. MESH
  */
  ierr = DMDACreate3d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,  // Create mesh
                      DMDA_STENCIL_STAR, -17, -17, -17, PETSC_DECIDE, PETSC_DECIDE,
                      PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da);   CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,pnx, pny, pnz, 0,0,0,0,0,0,0,0,0);  CHKERRQ(ierr); // Get NX, Ny, NZ

  /*
    CREATE VEC OBJECTS
  */
  ierr = DMCreateGlobalVector(da, pux);   CHKERRQ(ierr);   // Create a global ux vector derived from the DM object
  ierr = DMCreateGlobalVector(da, puy);   CHKERRQ(ierr);   // Create a global uy vector derived from the DM object
  ierr = DMCreateGlobalVector(da, puz);   CHKERRQ(ierr);   // Create a global uz vector derived from the DM object 

  ierr = VecDuplicate(*pux, &b);   CHKERRQ(ierr);          // RHS of the system

  ierr = VecDuplicate(*pux, pc11);   CHKERRQ(ierr);         // Duplicate vectors from grid object to each vector component
  ierr = VecDuplicate(*pux, pc12);   CHKERRQ(ierr);
  ierr = VecDuplicate(*pux, pc13);   CHKERRQ(ierr);
  ierr = VecDuplicate(*pux, pc44);   CHKERRQ(ierr);
  ierr = VecDuplicate(*pux, pc66);   CHKERRQ(ierr);
  ierr = VecDuplicate(*pux, prho);   CHKERRQ(ierr); 

  ierr = VecDuplicate(*pux, puxm1); CHKERRQ(ierr);  // ux at time n-1
  ierr = VecDuplicate(*pux, puxm2); CHKERRQ(ierr);  // ux at time n-2
  ierr = VecDuplicate(*pux, puxm3); CHKERRQ(ierr);  // ux at time n-3 
  ierr = VecDuplicate(*puy, puym1); CHKERRQ(ierr);  // uy at time n-1
  ierr = VecDuplicate(*puy, puym2); CHKERRQ(ierr);  // uy at time n-2
  ierr = VecDuplicate(*puy, puym3); CHKERRQ(ierr);  // uy at time n-3 
  ierr = VecDuplicate(*puz, puzm1); CHKERRQ(ierr);  // uz at time n-1
  ierr = VecDuplicate(*puz, puzm2); CHKERRQ(ierr);  // uz at time n-2
  ierr = VecDuplicate(*puz, puzm3); CHKERRQ(ierr);  // uz at time n-3 

  ierr = VecSet(*pc11, 2.f); CHKERRQ(ierr);      // Fill elastic properties vectors
  ierr = VecSet(*pc12, 1.f); CHKERRQ(ierr);
  ierr = VecSet(*pc13, 1.f); CHKERRQ(ierr);
  ierr = VecSet(*pc44, 1.f); CHKERRQ(ierr);
  ierr = VecSet(*pc66, 1.f); CHKERRQ(ierr);
  ierr = VecSet(*prho, 1.f); CHKERRQ(ierr);


  /*----------------------------------------------------------------------
    SET MODEL PATRAMETERS
  ------------------------------------------------------------------------*/

  *pxmax = 1.f; //[km]
  *pymax = 1.f;
  *pzmax = 1.f;

  *pdx = *pxmax / *pnx; //[km]
  *pdy = *pymax / *pny;
  *pdz = *pzmax / *pnz;

  double cmax;

  VecMax(ctx.model.c11, NULL, &cmax);

  *pdt = *pxmax / cmax; //[sec]
  *pt0 = 0.0;
  *ptf = 3.f; //[sec]
  *pnt = *ptf / *pdt;

  PetscPrintf(PETSC_COMM_WORLD,"MODEL:\n");
  PetscPrintf(PETSC_COMM_WORLD,"\t XMAX %f \t DX %f \t NX %i\n",*pxmax, *pdx, *pnx);
  PetscPrintf(PETSC_COMM_WORLD,"\t YMAX %f \t DY %f \t NY %i\n",*pymax, *pdy, *pny);
  PetscPrintf(PETSC_COMM_WORLD,"\t ZMAX %f \t DZ %f \t NZ %i\n",*pzmax, *pdz, *pnz);
  PetscPrintf(PETSC_COMM_WORLD,"TIME STEPPING: \n");
  PetscPrintf(PETSC_COMM_WORLD,"\t TMAX %f \t DT %f \t NT %i\n", *ptf, *pdt, *pnt);
  
  VecGetSize(*pux, &tmp);
  PetscPrintf(PETSC_COMM_WORLD,"MATRICES AND VECTORS: \n");
  PetscPrintf(PETSC_COMM_WORLD,"\t Vec elements %i\n", tmp);
  PetscPrintf(PETSC_COMM_WORLD,"\t Mat elements %i\n", tmp * tmp);

  /*  
    CREATE KSP, KRYLOV SUBSPACE OBJECTS 
  */

  KSP ksp_ux, ksp_uy, ksp_uz;

  // Create Krylov solver for ux component
  ierr = KSPCreate(comm, &ksp_ux);   CHKERRQ(ierr);                       // Create the KPS object
  ierr = KSPSetDM(ksp_ux, (DM) da);   CHKERRQ(ierr);                      // Set the DM to be used as preconditioner
  ierr = KSPSetComputeRHS(ksp_ux, compute_b_ux, &ctx);   CHKERRQ(ierr);   // Compute the right-hand side vector b
  ierr = KSPSetComputeOperators(ksp_ux, compute_A_ux, &ctx);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp_ux);   CHKERRQ(ierr);                      // KSP options can be changed during the runtime

  // Create Krylov solver for uy component
  ierr = KSPCreate(comm, &ksp_uy);   CHKERRQ(ierr);                       // Create the KPS object
  ierr = KSPSetDM(ksp_uy, (DM) da);   CHKERRQ(ierr);                      // Set the DM to be used as preconditioner
  ierr = KSPSetComputeRHS(ksp_uy, compute_b_ux, &ctx);   CHKERRQ(ierr);   // Compute the right-hand side vector b
  ierr = KSPSetComputeOperators(ksp_uy, compute_A_ux, &ctx);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp_uy);   CHKERRQ(ierr);                      // KSP options can be changed during the runtime

  // Create Krylov solver for uz component
  ierr = KSPCreate(comm, &ksp_uz);   CHKERRQ(ierr);                       // Create the KPS object
  ierr = KSPSetDM(ksp_uz, (DM) da);   CHKERRQ(ierr);                      // Set the DM to be used as preconditioner
  ierr = KSPSetComputeRHS(ksp_uz, compute_b_ux, &ctx);   CHKERRQ(ierr);   // Compute the right-hand side vector b
  ierr = KSPSetComputeOperators(ksp_uz, compute_A_ux, &ctx);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp_uz);   CHKERRQ(ierr);                      // KSP options can be changed during the runtime







  /*
    TIME LOOP
  */
  for (int it  = 1; it <= *pnt; it ++)
  {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Time step: \t %i of %i\n", it, ctx.time.nt);   CHKERRQ(ierr);
    
    ierr = KSPSolve(ksp_ux, b, *pux);   CHKERRQ(ierr);  // Solve the linear system using KSP

    ierr = VecCopy(*puxm2, *puxm3);   CHKERRQ(ierr);    // copy vector um2 to um3
    ierr = VecCopy(*puxm1, *puxm2);   CHKERRQ(ierr);    // copy vector um1 to um2
    ierr = VecCopy(*pux, *puxm1);   CHKERRQ(ierr);      // copy vector u to um1

    ierr = KSPSetComputeRHS(ksp_ux, update_b_ux, &ctx);   CHKERRQ(ierr); // new rhs for next iteration

    char buffer[32];                                 // The filename buffer.
    snprintf(buffer, sizeof(buffer), "tmp_Bvec_%i.m", it);
    ierr = save_wavefield_to_m_file(*pux, &buffer); CHKERRQ(ierr);
    // ierr = save_wavefield_to_m_file(*pc11, &buffer); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
  }





  /*
    CLEAN ALLOCATIONS AND EXIT
  */

  ierr = VecDestroy(&b);   CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_ux);   CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_uy);   CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_uz);   CHKERRQ(ierr);
  ierr = DMDestroy(&da);   CHKERRQ(ierr);

  ierr = PetscFinalize();   CHKERRQ(ierr);

  return 0;
}



// SAVE VECTOR TO .m FILE
PetscErrorCode
save_wavefield_to_m_file(Vec u, void * filename)
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


// UPDATE RHS OF UX EQUATION
PetscErrorCode
update_b_ux(KSP ksp, Vec b, void * ctx) 
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  ctx_t *c = (ctx_t *) ctx;

  // ierr = VecCopy(c->wf.ux, b);   CHKERRQ(ierr);
/* Get the DM oject of the KSP */
  DM da;
  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr);

  /* Get the global information of the DM grid*/
  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr);

  /* Grid spacing */
  double hx = (1.f / (double) (grid.mx - 1));
  double hy = (1.f / (double) (grid.my - 1));
  double hz = (1.f / (double) (grid.mz - 1));

  /*  A pointer to access b, the right-hand side PETSc vector
      viewed as a C array */
  
  double *** _b;
  double *** _uxm1, ***_uxm2, ***_uxm3;

  ierr = DMDAVecGetArray(da, b, &_b);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.uxm1, &_uxm1);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.uxm2, &_uxm2);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->wf.uxm3, &_uxm3);   CHKERRQ(ierr);

  /*
    Loop over the grid points, and fill b 
  */
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
          double f = hx * hy * hz; // Scaling
          
          f*= 2.5f * _uxm1[k][j][i] - 2.f * _uxm2[k][j][i] + 0.5f * _uxm3[k][j][i];

          _b[k][j][i] = f;
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, b, &_b);   CHKERRQ(ierr);   // Release the resource

  MatNullSpace   nullspace;
  MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);
  MatNullSpaceRemove(nullspace,b);
  MatNullSpaceDestroy(&nullspace);
  
  PetscFunctionReturn(0);
}



// COMPUTE INITIAL RHS OF UX EQUATION
PetscErrorCode
compute_b_ux(KSP ksp, Vec b, void * ctx) 
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ctx_t *c = (ctx_t *) ctx;
  PetscInt midnx, midny, midnz;
  PetscScalar dt;

  dt = c->time.dt;
  
  midnx = (int) (c->model.nx / 2); 
  midny = (int) (c->model.ny / 2); 
  midnz = (int) (c->model.nz / 2); 
  
  // ierr = PetscPrintf(PETSC_COMM_WORLD, "%i\n",midnx); CHKERRQ(ierr);

  /* Get the DM oject of the KSP */
  DM da;
  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr);

  /* Get the global information of the DM grid*/
  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr);

  /* Grid spacing */
  double hx = (1.f / (double) (grid.mx - 1));
  double hy = (1.f / (double) (grid.my - 1));
  double hz = (1.f / (double) (grid.mz - 1));

  /*  A pointer to access b, the right-hand side PETSc vector
      viewed as a C array */
  double *** _b;
  double *** rho;

  ierr = DMDAVecGetArray(da, b, &_b);   CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, c->model.rho, &rho);   CHKERRQ(ierr);

  unsigned int k;
  for(k = grid.zs; k < (grid.zs + grid.zm); k++) // Depth
  {
    unsigned int j;
    for(j = grid.ys; j < (grid.ys + grid.ym); j++) // Columns
    {
      unsigned int i;
      for(i = grid.xs; i < (grid.xs + grid.xm); i++) // Rows
      {
        if((i == 0) || (i == (grid.mx - 1)) ||
          (j == 0) || (j == (grid.my - 1)) ||
          (k == 0) || (k == (grid.mz - 1)))         /* Nodes on the boundary layers (\Gamma) */
        {
          _b[k][j][i] = 0.f;
        }
        else                                        /* Interior nodes in the domain (\Omega) */
        {  
          double f = dt * dt * hx * hy * hz / (2.f * rho[k][j][i]); // Scaling

          if ((i>=midnx) && (i<=midnx+1) && (j>=midny) && (j<=midny+1) && (k>=midnz) && (k<=midnz+1))
          {
            f *= 1.f;
          }
          else
          {
            f *= 0.f;
          }

          _b[k][j][i] = f;

        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, b, &_b);   CHKERRQ(ierr);   // Release the resource

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
  PetscScalar dt, dt2_2;
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
  dt2_2 = dt * dt / 2.f;

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
          v[0] = dt2_2/rho[k][j][i] * 2.f * (hyhzdhx + hxhzdhy + hxhydhz);
        // If neighbor is not a known boundary value
        // then we put an entry

        if((i - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i - 1;
          idxn[n].k = k;

          v[n] = - dt2_2/rho[k][j][i] * hyhzdhx; // Fill with the value
          
          n++; // One column added
        }
        if((i + 1) < (grid.mx - 1)) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i + 1;
          idxn[n].k = k;

          v[n] = - dt2_2/rho[k][j][i] * hyhzdhx;  // Fill with the value

          n++; // One column added
        }
        if((j - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j - 1;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = - dt2_2/rho[k][j][i] * hxhzdhy; // Fill with the value
  
          n++; // One column added
        }
        if((j + 1) < (grid.my - 1)) 
        {
          // Get the column indices
          idxn[n].j = j + 1;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = - dt2_2/rho[k][j][i] * hxhzdhy; // Fill with the value

          n++; // One column added
        }

        if((k - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k - 1;

          v[n] = - dt2_2/rho[k][j][i] * hxhydhz; // Fill with the value

          n++; // One column added
        }

        if((k + 1) < (grid.mz - 1)) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k + 1;

          v[n] = - dt2_2/rho[k][j][i] * hxhydhz; // Fill with the value

          n++; // One column added
        }
      
        // for(int ii=0;ii<6;ii++)
        // {
        //   v[ii]*= dt*dt / (2.0 * 1.f);
        // }
        v[0]+= hx * hy * hz;
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

  // ierr = DMDAVecRestoreArray(da, c->model.c11, &c11);   CHKERRQ(ierr);   // Release the resource
  // ierr = DMDAVecRestoreArray(da, c->model.rho, &rho);   CHKERRQ(ierr);   // Release the resource

  PetscFunctionReturn(0);
}
