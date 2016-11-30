
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

/*
  User-functions prototypes
*/
PetscErrorCode
compute_rhs(KSP, Vec, void *);
PetscErrorCode
compute_opt(KSP, Mat, Mat, void *);
PetscErrorCode
test_convergence_rate(KSP, Vec);

/*
  Main C function
*/
int
main(int argc, char * args[]) 
{
  PetscErrorCode ierr; // PETSc error code

  // Initialize the PETSc database and MPI
  // "petsc.opt" is the PETSc database file
  ierr = PetscInitialize(&argc, &args, NULL, NULL);   CHKERRQ(ierr); // PETSc error handler

  MPI_Comm comm = PETSC_COMM_WORLD;   // The global PETSc MPI communicator
  /*
    PETSc DM
    Create a default 16x16 2D grid object
    The minus sign means that the grid x and y dimensions
    are changeable through the command-line options
  */

  DM da;

    ierr = DMDACreate3d(comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                      DMDA_STENCIL_STAR, -17, -17, -17, PETSC_DECIDE, PETSC_DECIDE,
                      PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &da);   CHKERRQ(ierr);

  /*
    PETSc Vec
  */
  /*  Vector of unknowns approximating \varphi_{i,j}
      on the grid */
  Vec u;
  // Create a global vector derived from the DM object
  // "Global" means "distributed" in MPI language
  ierr = DMCreateGlobalVector(da, &u);   CHKERRQ(ierr);

  /* 
    The right-hand side vector approximating the values of f_{i,j}
  */
  Vec b;
  ierr = VecDuplicate(u, &b);   CHKERRQ(ierr);   // Duplicate creates a new vector of the same type as u

  Vec um1, um2, um3;
  ierr = VecDuplicate(u, &um1); CHKERRQ(ierr);  // u at time n-1
  ierr = VecDuplicate(u, &um2); CHKERRQ(ierr);  // u at time n-2
  ierr = VecDuplicate(u, &um3); CHKERRQ(ierr);  // u at time n-3 

  /* 
    Krylov Subspace (KSP) object to solve the linear system
  */
  KSP ksp;

  ierr = KSPCreate(comm, &ksp);   CHKERRQ(ierr);   // Create the KPS object
  ierr = KSPSetDM(ksp, (DM) da);   CHKERRQ(ierr);   // Set the DM to be used as preconditioner
  ierr = KSPSetComputeRHS(ksp, compute_rhs, NULL);   CHKERRQ(ierr);   // Compute the right-hand side vector b
  ierr = KSPSetComputeOperators(ksp, compute_opt, NULL);   CHKERRQ(ierr);   // Compute and assemble the coefficient matrix A
  ierr = KSPSetFromOptions(ksp);   CHKERRQ(ierr);   // KSP options can be changed during the runtime

  // PetscBool flg;
  for (int ctr = 1; ctr<=2; ctr++)
  {
  ierr = VecZeroEntries(u);   CHKERRQ(ierr);  // Set all vector values equal to zero

  ierr = KSPSolve(ksp, b, u);   CHKERRQ(ierr);   // Solve the linear system using KSP

  // VecView(b,PETSC_VIEWER_STDOUT_WORLD);

  PetscViewer viewer;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "tmp_Uvec.m", &viewer);
  PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
  VecView(u,viewer);
  PetscViewerPopFormat(viewer);
  PetscViewerDestroy(&viewer);

  ierr = VecCopy(u,b);   CHKERRQ(ierr);   // copy vector u to b
  ierr = VecCopy(um2,um3);   CHKERRQ(ierr);   // copy vector um2 to um3
  ierr = VecCopy(um1,um2);   CHKERRQ(ierr);   // copy vector um1 to um2
  ierr = VecCopy(u,um1);   CHKERRQ(ierr);   // copy vector u to um1

  // VecEqual(u,b,&flg);
  // if (flg)
  // {
  //   PetscPrintf(PETSC_COMM_WORLD,"Vectors are equal\n");
  // }



  // Verifies the implementation by comparing the
  // numerical solution to the analytical solution
  // The function computes a norm of the difference
  // between the computed solution and the exact solution
  ierr = test_convergence_rate(ksp, u);   CHKERRQ(ierr);
  }
  /*
    Cleanup the allocations, and exit
  */
  ierr = VecDestroy(&u);   CHKERRQ(ierr);
  ierr = VecDestroy(&um1);   CHKERRQ(ierr);
  ierr = VecDestroy(&um2);   CHKERRQ(ierr);
  ierr = VecDestroy(&um3);   CHKERRQ(ierr);
  ierr = VecDestroy(&b);   CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);   CHKERRQ(ierr);
  ierr = DMDestroy(&da);   CHKERRQ(ierr);

  // Exit the MPI communicator and finalize the PETSc initialization objects
  ierr = PetscFinalize();   CHKERRQ(ierr);

  return 0;
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
    double z = k * hz; // Z

    unsigned int j;
    for(j = grid.ys; j < (grid.ys + grid.ym); j++) // Columns
    {
      double y = j * hy; // Y

      unsigned int i;
      for(i = grid.xs; i < (grid.xs + grid.xm); i++) // Rows
      {
        double x = i * hx; // X

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
          double x2 = x * x; // x^2
          double y2 = y * y; // y^2
          double z2 = z * z; // y^2
          double x4 = x2 * x2;
          double y4 = y2 * y2;
          double z4 = z2 * z2;

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
compute_opt(KSP ksp, Mat A, Mat J, void * ctx)
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

/* Test the convergence rate */
PetscErrorCode
test_convergence_rate(KSP ksp, Vec u)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;

  // Get the DMDA object
  DM da;
  ierr = KSPGetDM(ksp, &da);   CHKERRQ(ierr);

  // Get the grid information
  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);   CHKERRQ(ierr);

  // Create a global vector
  Vec u_;
  ierr = VecDuplicate(u, &u_);   CHKERRQ(ierr);

  double *** _u;

  double hx = (1.f / (double) (grid.mx - 1));
  double hy = (1.f / (double) (grid.my - 1));
  double hz = (1.f / (double) (grid.mz - 1));

  // Get a pointer to the PETSc vector
  ierr = DMDAVecGetArray(da, u_, &_u);   CHKERRQ(ierr);
  
  unsigned int k;
  for(k = grid.zs; k < (grid.zs + grid.zm); k++)
  {
    double z = k * hz;

    unsigned int j;
    for(j = grid.ys; j < (grid.ys + grid.ym); j++)
    {
      double y = j * hy;

      unsigned int i;
      for(i = grid.xs; i < (grid.xs + grid.xm); i++)
      {
        double x = i * hx;

        double x2 = x * x;
        double y2 = y * y;
        double z2 = z * z;

        _u[k][j][i] = (x2 - x2 * x2) * (y2 * y2 - y2) * (z2 - z2 * z2);
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, u_, &_u);   CHKERRQ(ierr);

  double val = 0.f;

  ierr = VecAXPY(u, -1.f, u_);   CHKERRQ(ierr);
  ierr = VecNorm(u, NORM_INFINITY, (PetscScalar *) &val);   CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,
                    "Numerical Error [NORM_INFINITY]: \t %g\n", val);   CHKERRQ(ierr);

  ierr = VecDestroy(&u_); CHKERRQ(ierr);

  return 0;
}
