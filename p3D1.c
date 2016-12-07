
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

#define debprint(expr) printf(#expr " = %g\n", expr)

//User-functions prototypes
PetscErrorCode compute_rhs(KSP, Vec, void *);
PetscErrorCode compute_A_ux(KSP, Mat, Mat, void *);
PetscErrorCode compute_A_uy(KSP, Mat, Mat, void *); 
PetscErrorCode compute_A_uz(KSP, Mat, Mat, void *);
PetscErrorCode compute_rhs2(KSP, Vec, void *);
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
  Vec c1;
  Vec c2;
  Vec c3;
  Vec c4;
  Vec c5;
  Vec rho;
} model_par;

// User context to use with PETSc functions
typedef struct {
  wfield wf;
  model_par model;
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
  Vec *pc1, *pc2, *pc3, *pc4, *pc5, *prho;
  PetscScalar *pdx, *pdy, *pdz, *pxmax, *pymax, *pzmax;
  PetscInt *pnx, *pny, *pnz;
  ctx_t ctx, *pctx;
  int tmp;


  /*
    LIST OF POINTERS
  */
  pctx = &ctx;
  
  pux = &ctx.wf.ux;
  puy = &ctx.wf.uy;
  puz = &ctx.wf.uz;

  pnx = &ctx.model.nx;
  pny = &ctx.model.ny;
  pnz = &ctx.model.nz;

  pdx = &ctx.model.dx;
  pdy = &ctx.model.dy;
  pdz = &ctx.model.dz;
  
  pxmax = &ctx.model.xmax;
  pymax = &ctx.model.ymax;
  pzmax = &ctx.model.zmax;
  
  pc1 = &ctx.model.c1;
  pc2 = &ctx.model.c2;
  pc3 = &ctx.model.c3;
  pc4 = &ctx.model.c4;
  pc5 = &ctx.model.c5;
  prho = &ctx.model.rho;



  ierr = PetscInitialize(&argc, &args, NULL, NULL);   CHKERRQ(ierr);   // Initialize the PETSc database and MPIeo
  MPI_Comm comm = PETSC_COMM_WORLD;   // The global PETSc MPI communicator


  /*
    CREATE DMDA OBJECT. MESH
  */
  ierr = DMDACreate3d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,  // Create mesh
                      DMDA_STENCIL_STAR, -17, -17, -17, PETSC_DECIDE, PETSC_DECIDE,
                      PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da);   CHKERRQ(ierr);

  /*
    CREATE VEC OBJECTS
  */
  ierr = DMCreateGlobalVector(da, pux);   CHKERRQ(ierr);   // Create a global ux vector derived from the DM object
  ierr = DMCreateGlobalVector(da, puy);   CHKERRQ(ierr);   // Create a global uy vector derived from the DM object
  ierr = DMCreateGlobalVector(da, puz);   CHKERRQ(ierr);   // Create a global uz vector derived from the DM object 

  ierr = VecDuplicate(*pux, &b);   CHKERRQ(ierr);          // RHS of the system

  ierr = VecDuplicate(*pux, pc1);   CHKERRQ(ierr);      // Duplicate vectors from grid object to each vector component
  ierr = VecDuplicate(*pux, pc2);   CHKERRQ(ierr);
  ierr = VecDuplicate(*pux, pc3);   CHKERRQ(ierr);
  ierr = VecDuplicate(*pux, pc4);   CHKERRQ(ierr);
  ierr = VecDuplicate(*pux, pc5);   CHKERRQ(ierr);
  ierr = VecDuplicate(*pux, prho);   CHKERRQ(ierr); 

  ierr = VecDuplicate(*pux, &ctx.wf.uxm1); CHKERRQ(ierr);  // ux at time n-1
  ierr = VecDuplicate(*pux, &ctx.wf.uxm2); CHKERRQ(ierr);  // ux at time n-2
  ierr = VecDuplicate(*pux, &ctx.wf.uxm3); CHKERRQ(ierr);  // ux at time n-3 
  ierr = VecDuplicate(*puy, &ctx.wf.uym1); CHKERRQ(ierr);  // uy at time n-1
  ierr = VecDuplicate(*puy, &ctx.wf.uym2); CHKERRQ(ierr);  // uy at time n-2
  ierr = VecDuplicate(*puy, &ctx.wf.uym3); CHKERRQ(ierr);  // uy at time n-3 
  ierr = VecDuplicate(*puz, &ctx.wf.uzm1); CHKERRQ(ierr);  // uz at time n-1
  ierr = VecDuplicate(*puz, &ctx.wf.uzm2); CHKERRQ(ierr);  // uz at time n-2
  ierr = VecDuplicate(*puz, &ctx.wf.uzm3); CHKERRQ(ierr);  // uz at time n-3 


  /*
    SET MODEL PATRAMETERS
  */
  ierr = DMDAGetInfo(da,0,pnx, pny, pnz, 0,0,0,&tmp,0,0,0,0,0);  CHKERRQ(ierr); // Get NX, Ny, NZ

  // ierr = DMDAGetCorners(da, &pctx->model.xmin, 
  //                           &pctx->model.ymin, 
  //                           &pctx->model.zmin,
  //                           &pctx->model.xmax,
  //                           &pctx->model.ymax, 
  //                           &pctx->model.zmax);  CHKERRQ(ierr);

  *pdx = 10.f;      // Set grid step, dx, dy, dz
  *pdy = 10.f;
  *pdz = 10.f;

  *pxmax = *pdx * (*pnx - 1);   // Max values over OX, OY, OZ
  *pymax = *pdy * (*pny - 1);
  *pzmax = *pdz * (*pnz - 1);

  ierr = VecSet(*pc1, 1.f); CHKERRQ(ierr);      // Fill elastic properties vectors
  ierr = VecSet(*pc2, 1.f); CHKERRQ(ierr);
  ierr = VecSet(*pc3, 1.f); CHKERRQ(ierr);
  ierr = VecSet(*pc4, 1.f); CHKERRQ(ierr);
  ierr = VecSet(*pc5, 1.f); CHKERRQ(ierr);
  ierr = VecSet(*prho, 1.f); CHKERRQ(ierr);

  // PetscPrintf(PETSC_COMM_WORLD,"%g \n",pctx->model.dx);

  /* 
    Krylov Subspace (KSP) object to solve the linear system
  */
  KSP ksp_ux, ksp_uy, ksp_uz;

  // Create Krylov solver for ux component
  ierr = KSPCreate(comm, &ksp_ux);   CHKERRQ(ierr);   // Create the KPS object
  ierr = KSPSetDM(ksp_ux, (DM) da);   CHKERRQ(ierr);   // Set the DM to be used as preconditioner
  ierr = KSPSetComputeRHS(ksp_ux, compute_rhs, &ctx);   CHKERRQ(ierr);   // Compute the right-hand side vector b
  ierr = KSPSetComputeOperators(ksp_ux, compute_A_ux, &ctx);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp_ux);   CHKERRQ(ierr);   // KSP options can be changed during the runtime

  // Create Krylov solver for uy component
  ierr = KSPCreate(comm, &ksp_uy);   CHKERRQ(ierr);   // Create the KPS object
  ierr = KSPSetDM(ksp_uy, (DM) da);   CHKERRQ(ierr);   // Set the DM to be used as preconditioner
  ierr = KSPSetComputeRHS(ksp_uy, compute_rhs, &ctx);   CHKERRQ(ierr);   // Compute the right-hand side vector b
  ierr = KSPSetComputeOperators(ksp_uy, compute_A_ux, &ctx);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp_uy);   CHKERRQ(ierr);   // KSP options can be changed during the runtime

  // Create Krylov solver for uz component
  ierr = KSPCreate(comm, &ksp_uz);   CHKERRQ(ierr);   // Create the KPS object
  ierr = KSPSetDM(ksp_uz, (DM) da);   CHKERRQ(ierr);   // Set the DM to be used as preconditioner
  ierr = KSPSetComputeRHS(ksp_uz, compute_rhs, &ctx);   CHKERRQ(ierr);   // Compute the right-hand side vector b
  ierr = KSPSetComputeOperators(ksp_uz, compute_A_ux, &ctx);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp_uz);   CHKERRQ(ierr);   // KSP options can be changed during the runtime

  for (int it  = 1; it <= 4; it ++)
  {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Time step: \t %i \n", it);   CHKERRQ(ierr);
    // ierr = VecZeroEntries(*pux);   CHKERRQ(ierr);  // Set all vector values equal to zero
    ierr = KSPSolve(ksp_ux, b, *pux);   CHKERRQ(ierr);   // Solve the linear system using KSP

    ierr = VecCopy(ctx.wf.uxm2, ctx.wf.uxm3);   CHKERRQ(ierr);   // copy vector um2 to um3
    ierr = VecCopy(ctx.wf.uxm1, ctx.wf.uxm2);   CHKERRQ(ierr);   // copy vector um1 to um2
    ierr = VecCopy(*pux, ctx.wf.uxm1);   CHKERRQ(ierr);   // copy vector u to um1

    ierr = KSPSetComputeRHS(ksp_ux, compute_rhs2, &ctx);   CHKERRQ(ierr); // new rhs for next iteration
    
    // save_wavefield_to_m_file(ctx.wf.uxm1, "wfux.m");

    char buffer[32];                                 // The filename buffer.
    snprintf(buffer, sizeof(buffer), "tmp_Bvec_%i.m", it);
    // ierr = save_wavefield_to_m_file(*pux, &buffer); CHKERRQ(ierr);
    ierr = save_wavefield_to_m_file(b, &buffer); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);

  }
  /*
    Cleanup the allocations, and exit
  */
  // ierr = VecDestroy(&pux);   CHKERRQ(ierr);
  // ierr = VecDestroy(&ctx);   CHKERRQ(ierr);

  ierr = VecDestroy(&b);   CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_ux);   CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_uy);   CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp_uz);   CHKERRQ(ierr);
  ierr = DMDestroy(&da);   CHKERRQ(ierr);

  // Exit the MPI communicator and finalize the PETSc initialization objects
  ierr = PetscFinalize();   CHKERRQ(ierr);

  return 0;
}


PetscErrorCode
save_wavefield_to_m_file(Vec u, void * filename)
{
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



PetscErrorCode
compute_rhs2(KSP ksp, Vec b, void * ctx) 
{
  PetscErrorCode ierr;

  ctx_t *c = (ctx_t *) ctx;

  ierr = VecCopy(c->wf.ux, b);   CHKERRQ(ierr);
  // ierr = save_wavefield_to_m_file(c->wf.ux, "wfux.m"); CHKERRQ(ierr);
    // ierr = save_wavefield_to_m_file(c->wf.ux, "wfux.m"); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}



/*
  Compute the right-hand side vector
*/
PetscErrorCode
compute_rhs(KSP ksp, Vec b, void * ctx) 
{
  PetscFunctionBegin;

  PetscErrorCode ierr;

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
  ierr = DMDAVecGetArray(da, b, &_b);   CHKERRQ(ierr);

  /*
    Loop over the grid points, and fill b 
  */
  unsigned int k;
  for(k = grid.zs; k < (grid.zs + grid.zm); k++) // Depth
  {
    // double z = k * hz; // Z

    unsigned int j;
    for(j = grid.ys; j < (grid.ys + grid.ym); j++) // Columns
    {
      // double y = j * hy; // Y

      unsigned int i;
      for(i = grid.xs; i < (grid.xs + grid.xm); i++) // Rows
      {
        // double x = i * hx; // X

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
          // double x2 = x * x; // x^2
          // double y2 = y * y; // y^2
          // double z2 = z * z; // y^2
          // double x4 = x2 * x2;
          // double y4 = y2 * y2;
          // double z4 = z2 * z2;

          // debprint(z4); 

          double f = hx * hy * hz; // Scaling

          // f *=  2.f * (-y2 * (-1.f + y2) * z2 * (-1.f + z2)  + 
          //       x4 * (z2 - z4 + y4 * (-1.f + 6.f * z2) + y2 * (1.f - 12.f * z2 + 6.f * z4)) + 
          //       x2 * (z2 * (-1.f + z2) + y2 * (-1.f + 18.f * z2 - 12.f * z4) + y4 * (1.f - 12.f * z2 + 6.f * z4)));

          if ((i>=6) && (i<=8) && (j>=6) && (j<=8) && (k>=6) && (k<=8))
          {
            f *= 5.f;
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

  PetscFunctionReturn(0);
}

/* Compute the operator matrix A */
PetscErrorCode 
compute_A_ux(KSP ksp, Mat A, Mat J, void * ctx)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;

  DM da;
  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr);   // Get the DMDA object

  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr);   // Get the grid information

  /*  A PETSc data structure to store information
      about a single row or column in the stencil */
  MatStencil idxm;
  MatStencil idxn[7];

/*=========================================================================================
 Using example 34 from
 http://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex34.c.html \
  =========================================================================================*/

  // The matrix values
  PetscScalar v[7], hx, hy, hz, hyhzdhx, hxhzdhy, hxhydhz;
  PetscInt n;


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
          v[0] = 2.f * (hyhzdhx + hxhzdhy + hxhydhz);
        // If neighbor is not a known boundary value
        // then we put an entry

        if((i - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i - 1;
          idxn[n].k = k;

          v[n] = - hyhzdhx; // Fill with the value
          
          n++; // One column added
        }
        if((i + 1) < (grid.mx - 1)) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i + 1;
          idxn[n].k = k;

          v[n] = - hyhzdhx;  // Fill with the value

          n++; // One column added
        }
        if((j - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j - 1;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = - hxhzdhy; // Fill with the value
  
          n++; // One column added
        }
        if((j + 1) < (grid.my - 1)) 
        {
          // Get the column indices
          idxn[n].j = j + 1;
          idxn[n].i = i;
          idxn[n].k = k;

          v[n] = - hxhzdhy; // Fill with the value

          n++; // One column added
        }

        if((k - 1) > 0) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k - 1;

          v[n] = - hxhydhz; // Fill with the value

          n++; // One column added
        }

        if((k + 1) < (grid.mz - 1)) 
        {
          // Get the column indices
          idxn[n].j = j;
          idxn[n].i = i;
          idxn[n].k = k + 1;

          v[n] = - hxhydhz; // Fill with the value

          n++; // One column added
        }
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
  
  // PetscViewer viewer;
  // PetscViewerASCIIOpen(PETSC_COMM_WORLD, "Amat.m", &viewer);
  // PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
  // MatView(A,viewer);
  // PetscViewerPopFormat(viewer);
  // PetscViewerDestroy(&viewer);

  PetscFunctionReturn(0);
}
