#ifndef lint
static char vcid[] = "$Id: iterativ.c,v 1.56 1997/01/01 03:35:54 bsmith Exp balay $";
#endif

/*
   This file contains some simple default routines.  
   These routines should be SHORT, since they will be included in every
   executable image that uses the iterative routines (note that, through
   the registry system, we provide a way to load only the truely necessary
   files) 
 */
#include "src/ksp/kspimpl.h"   /*I "ksp.h" I*/

#undef __FUNC__  
#define __FUNC__ "KSPDefaultFreeWork"
/*
  KSPDefaultFreeWork - Free work vectors

  Input Parameters:
. ksp  - iterative context
 */
int KSPDefaultFreeWork( KSP ksp )
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->work)  return VecDestroyVecs(ksp->work,ksp->nwork);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPSingularValueMonitor"
/*@C
    KSPSingularValueMonitor - Prints the two norm of the true residual and
    estimation of the extreme eigenvalues of the preconditioned problem
    at each iteration.
 
    Input Parameters:
.   ksp - the iterative context
.   n  - the iteration
.   rnorm - the two norm of the residual

    Options Database Key:
$     -ksp_singmonitor

    Notes:
    The CG solver uses the Lanczos technique for eigenvalue computation, 
    while GMRES uses the Arnoldi technique; other iterative methods do
    not currently compute singular values.

.keywords: KSP, CG, default, monitor, extreme, eigenvalues, Lanczos

.seealso: KSPComputeExtremeSingularValues()
@*/
int KSPSingularValueMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  double emin,emax,c;
  int    ierr;

  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (!ksp->calc_sings) {
    PetscPrintf(ksp->comm,"%d KSP Residual norm %14.12e \n",n,rnorm);
  }
  else {
    ierr = KSPComputeExtremeSingularValues(ksp,&emax,&emin); CHKERRQ(ierr);
    c = emax/emin;
    PetscPrintf(ksp->comm,"%d KSP Residual norm %14.12e %% max %g min %g max/min %g\n",n,rnorm,emax,emin,c);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultMonitor"
/*ARGSUSED*/
/*@C
   KSPDefaultMonitor - Print the residual norm at each iteration of an
   iterative solver.

   Input Parameters:
.  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).  
.  dummy - unused monitor context 

.keywords: KSP, default, monitor, residual

.seealso: KSPSetMonitor(), KSPTrueMonitor(), KSPLGMonitorCreate()
@*/
int KSPDefaultMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  PetscPrintf(ksp->comm,"%d KSP Residual norm %14.12e \n",n,rnorm); return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPTrueMonitor"
/*@C
   KSPTrueMonitor - Prints the true residual norm as well as the preconditioned
   residual norm at each iteration of an iterative solver.

   Input Parameters:
.  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm (preconditioned) residual value (may be estimated).  
.  dummy - unused monitor context 

   Options Database Key:
$   -ksp_truemonitor

   Notes:
   When using right preconditioning, these values are equivalent.

   When using either ICC or ILU preconditioners in BlockSolve95 
   (via MATMPIROWBS matrix format), then use the option
$      -ksp_bsmonitor
   to print both the true and scaled residual norms.  By default,
   only the scaled residual norm is printed.

.keywords: KSP, default, monitor, residual

.seealso: KSPSetMonitor(), KSPDefaultMonitor(), KSPLGMonitorCreate()
@*/
int KSPTrueMonitor(KSP ksp,int n,double rnorm,void *dummy)
{
  int          ierr;
  Vec          resid,work;
  double       scnorm;
  

  ierr = VecDuplicate(ksp->vec_rhs,&work); CHKERRQ(ierr);
  ierr = KSPBuildResidual(ksp,0,work,&resid); CHKERRQ(ierr);
  ierr = VecNorm(resid,NORM_2,&scnorm); CHKERRQ(ierr);
  VecDestroy(work);
  PetscPrintf(ksp->comm,"%d KSP preconditioned resid norm %14.12e true resid norm %14.12e\n",n,rnorm,scnorm); 
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultSMonitor"
int KSPDefaultSMonitor(KSP ksp,int its, double fnorm,void *dummy)
{
  if (fnorm > 1.e-9) {
    PetscPrintf(ksp->comm, "iter = %d, KSP Residual norm %g \n",its,fnorm);
  }
  else if (fnorm > 1.e-11){
    PetscPrintf(ksp->comm, "iter = %d, KSP Residual norm %5.3e \n",its,fnorm);
  }
  else {
    PetscPrintf(ksp->comm, "iter = %d, KSP Residual norm < 1.e-11\n",its);
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultConverged"
/*ARGSUSED*/
/*@C
   KSPDefaultConverged - Determines convergence of
   the iterative solvers (default code).

   Input Parameters:
.  ksp   - iterative context
.  n     - iteration number
.  rnorm - 2-norm residual value (may be estimated)
.  dummy - unused convergence context 

   Returns:
   1 if the iteration has converged
  -1 if residual norm exceeds divergence threshold;
   0 otherwise.

   Notes:
   KSPDefaultConverged() reaches convergence when
$        rnorm < MAX ( rtol * rnorm_0, atol );
$  Divergence is detected if
$        rnorm > dtol * rnorm_0,
$  where rtol = relative tolerance,
$        atol = absolute tolerance.
$        dtol = divergence tolerance,
$        rnorm_0 = initial residual norm

   Use KSPSetTolerances() to alter the defaults for 
   rtol, atol, dtol.

.keywords: KSP, default, convergence, residual

.seealso: KSPSetConvergenceTest(), KSPSetTolerances()
@*/
int KSPDefaultConverged(KSP ksp,int n,double rnorm,void *dummy)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if ( n == 0 ) {
    ksp->ttol   = PetscMax(ksp->rtol*rnorm,ksp->atol);
    ksp->rnorm0 = rnorm;
  }
  if ( rnorm <= ksp->ttol )      return 1;
  if ( rnorm >= ksp->divtol*ksp->rnorm0 || rnorm != rnorm) return -1;
  return(0);
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultBuildSolution"
/*
   KSPDefaultBuildSolution - Default code to create/move the solution.

   Input Parameters:
.  ksp - iterative context
.  v   - pointer to the user's vector  

   Output Parameter:
.  V - pointer to a vector containing the solution

.keywords:  KSP, build, solution, default

.seealso: KSPGetSolution(), KSPDefaultBuildResidual()
*/
int KSPDefaultBuildSolution(KSP ksp,Vec v,Vec *V)
{
  int ierr;
  if (ksp->pc_side == PC_RIGHT) {
    if (ksp->B) {
      if (v) {ierr = PCApply(ksp->B,ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else {SETERRQ(1,0,"Not working with right preconditioner");}
    }
    else        {
      if (v) {ierr = VecCopy(ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else { *V = ksp->vec_sol;}
    }
  }
  else if (ksp->pc_side == PC_SYMMETRIC) {
    if (ksp->B) {
      if (v) {ierr = PCApplySymmetricRight(ksp->B,ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else {SETERRQ(1,0,"Not working with symmetric preconditioner");}
    }
    else        {
      if (v) {ierr = VecCopy(ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
      else { *V = ksp->vec_sol;}
    }
  }
  else {
    if (v) {ierr = VecCopy(ksp->vec_sol,v); CHKERRQ(ierr); *V = v;}
    else { *V = ksp->vec_sol; }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultBuildResidual"
/*
   KSPDefaultBuildResidual - Default code to compute the residual.

   Input Parameters:
.  ksp - iterative context
.  t   - pointer to temporary vector
.  v   - pointer to user vector  

   Output Parameter:
.  V - pointer to a vector containing the residual

.keywords:  KSP, build, residual, default

.seealso: KSPDefaultBuildSolution()
*/
int KSPDefaultBuildResidual(KSP ksp,Vec t,Vec v,Vec *V)
{
  int          ierr;
  MatStructure pflag;
  Vec          T;
  Scalar       mone = -1.0;
  Mat          Amat, Pmat;

  PCGetOperators(ksp->B,&Amat,&Pmat,&pflag);
  ierr = KSPBuildSolution(ksp,t,&T); CHKERRQ(ierr);
  ierr = MatMult(Amat, t, v ); CHKERRQ(ierr);
  ierr = VecAYPX(&mone, ksp->vec_rhs, v ); CHKERRQ(ierr);
  *V = v; return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultGetWork"
/*
  KSPDefaultGetWork - Gets a number of work vectors.

  Input Parameters:
. ksp  - iterative context
. nw   - number of work vectors to allocate

  Notes:
  Call this only if no work vectors have been allocated 
 */
int  KSPDefaultGetWork( KSP ksp, int nw )
{
  int ierr;
  if (ksp->work) KSPDefaultFreeWork( ksp );
  ksp->nwork = nw;
  ierr = VecDuplicateVecs(ksp->vec_rhs,nw,&ksp->work); CHKERRQ(ierr);
  PLogObjectParents(ksp,nw,ksp->work);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultAdjustWork"
/*
  KSPDefaultAdjustWork - Adjusts work vectors.

  Input Parameters:
. ksp  - iterative context
 */
int KSPDefaultAdjustWork( KSP ksp )
{
  if ( ksp->adjust_work_vectors ) {
    return (ksp->adjust_work_vectors)(ksp, ksp->work,ksp->nwork); 
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "KSPDefaultDestroy"
/*
  KSPDefaultDestroy - Destroys a iterative context variable for methods with
  no separate context.  Preferred calling sequence KSPDestroy().

  Input Parameters: 
. ksp - the iterative context
*/
int KSPDefaultDestroy(PetscObject obj)
{
  KSP ksp = (KSP) obj;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  if (ksp->data) PetscFree(ksp->data);

  /* free work vectors */
  KSPDefaultFreeWork( ksp );
  return 0;
}

