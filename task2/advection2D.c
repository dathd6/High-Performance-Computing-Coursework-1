/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed velocity



Outputs: initial.dat - inital values of u(x,y)
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files
**********************************************************************/

#include <math.h>
#include <omp.h>
#include <stdio.h>

/*********************************************************************
                      Main function
**********************************************************************/

int main() {

  /* Grid properties */
  const int NX = 1000;    // Number of x points
  const int NY = 1000;    // Number of y points
  const float xmin = 0.0; // Minimum x value
  const float ymin = 0.0; // Minimum y value
  /* Task 2
   *  Change the computational domain
   *  - 0 <= x <= 30.0m
   *  - 0 <= y <= 30.0m
   * */
  const float xmax = 30.0; // Maximum x value
  const float ymax = 30.0; // Maximum y value

  /* Parameters for the Gaussian initial conditions */
  /* Task 2
   *    set new values for y0 and sigmay
   *    add new values t0 and sigmat
   * */
  const float x0 = 0.1; // Centre(x)
  const float y0 = 15;  // Centre(y)
  const float t0 = 3;
  const float sigmax = 0.03;    // Width(x)
  const float sigmay = 5; // Width(y)
  const float sigmat = 1;
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared
  const float sigmat2 = sigmat * sigmat;

  /* Boundary conditions */
  const float bval_left = 0.0;  // Left boudnary value
  const float bval_right = 0.0; // Right boundary value
  const float bval_lower = 0.0; // Lower boundary
  const float bval_upper = 0.0; // Upper bounary

  /* Time stepping parameters */
  const float CFL = 0.9; // CFL number
  /* Task 2
   *  Change maximum number of time steps to 1000
   *  so that the material dose not advect out of
   *  the computational domain
   * */
  const int nsteps = 1000; // Number of time steps

  /* Velocity */
  /* Task 2
   *   Change the horizontal velocity velx = 1.0 m/s
   *   Change the vertical velocity vely= 0 m/s
   * */
  const float velx = 1.0; // Velocity in x direction
  const float vely = 0.0; // Velocity in y direction

  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX + 2];            // x-axis values
  float y[NX + 2];            // y-axis values
  float u[NX + 2][NY + 2];    // Array of u values
  float dudt[NX + 2][NY + 2]; // Rate of change of u

  float x2; // x squared (used to calculate iniital conditions)
  float y2; // y squared (used to calculate iniital conditions)
  float t2; // t squared

  /* Calculate distance between points */
  float dx = (xmax - xmin) / ((float)NX);
  float dy = (ymax - ymin) / ((float)NY);

  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ((fabs(velx) / dx) + (fabs(vely) / dy));

  /*** Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt * (float)nsteps);
  printf("Distance advected x = %g\n", velx * dt * (float)nsteps);
  printf("Distance advected y = %g\n", vely * dt * (float)nsteps);

  /*** Place x points in the middle of the cell ***/
  /* LOOP 1 */
#pragma omp parallel for shared(x, dx)
  for (int i = 0; i < NX + 2; i++) {
    x[i] = ((float)i - 0.5) * dx;
  }

  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 */
#pragma omp parallel for shared(y, dy)
  for (int j = 0; j < NY + 2; j++) {
    y[j] = ((float)j - 0.5) * dy;
  }

  /*** Set up Gaussian initial conditions ***/
  /* LOOP 3 */
#pragma omp parallel for collapse(2) shared(u)
  for (int i = 0; i < NX + 2; i++) {
    for (int j = 0; j < NY + 2; j++) {
      /* Task 2
       *  Start with an empty computational domain
       *  u(x, y) = 0
       * */
      u[i][j] = 0;
    }
  }

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  /* LOOP 4 */
  /*
   * This nested loop cannot be paralleled
   *    - The 'fprintf' need to execute in order so the
   *      'gnuplot' command can visualize the plot
   *    - Because multiple threads attempt to write to a file at
   *      the same time (race conditions) could lead to output from
   *      different threads is mixed together lead to corrupted file
   * */
  for (int i = 0; i < NX + 2; i++) {
    for (int j = 0; j < NY + 2; j++) {
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);

  /*** Update solution by looping over time steps ***/
  /* LOOP 5 */
  /*
   * This loop can't run parallel safely and effectively
   * Due to data dependencies
   *    - Types of loop-carried dependency: Output dependency
   *    - Reason: Write the same element of u lead to race conditions.
   *              Loop 6 & 7 update boundary elements of 'u',
   *              while Loop 9 update lead to a certain element could
   *              be changed before updating it.
   * */
  for (int m = 0; m < nsteps; m++) {

    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 */
#pragma omp parallel for private(t2, y2) shared(u, dt, m, y)
    for (int j = 0; j < NY + 2; j++) {
      /* Task 2:
       *    x = 0 (the left boundary) so that it varies with height (y)
       *    and time (t = step(m) * dt) accroding to the given equation
       * */
      t2 = (m * dt - t0) * (m * dt - t0);
      y2 = (y[j] - y0) * (y[j] - y0);
      u[0][j] = exp(-1.0 * ((t2 / (2.0 * sigmat2)) + (y2 / (2.0 * sigmay2))));

      u[NX + 1][j] = bval_right;
    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 */
#pragma omp parallel for shared(u)
    for (int i = 0; i < NX + 2; i++) {
      u[i][0] = bval_lower;
      u[i][NY + 1] = bval_upper;
    }

    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 */
#pragma omp parallel for collapse(2) shared(dudt, u, dx, dy)
    for (int i = 1; i < NX + 1; i++) {
      for (int j = 1; j < NY + 1; j++) {
        dudt[i][j] = -velx * (u[i][j] - u[i - 1][j]) / dx -
                     vely * (u[i][j] - u[i][j - 1]) / dy;
      }
    }

    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
#pragma omp parallel for collapse(2) shared(u, dudt, dt)
    for (int i = 1; i < NX + 1; i++) {
      for (int j = 1; j < NY + 1; j++) {
        u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }

  } // time loop

  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10 */
  /* This nested loop cannot be paralleled
   *    - The 'fprintf' need to execute in order so the
   *      'gnuplot' command can visualize the plot
   *    - Because multiple threads attempt to write to a file at
   *      the same time (race conditions) could lead to output from
   *      different threads is mixed together lead to corrupted file
   * */
  for (int i = 0; i < NX + 2; i++) {
    for (int j = 0; j < NY + 2; j++) {
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);

  return 0;
}

/* End of file ******************************************************/
