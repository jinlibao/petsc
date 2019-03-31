
#include <../src/ksp/ksp/impls/mlbcgs/mlbcgsimpl.h>       /*I  "petscksp.h"  I*/

/* PetscErrorCode randn(PetscScalar *, PetscInt, PetscRandom)
 * Generate an array of which entries are normally distributed using Marsaglia polar method
 * see https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
 */
PetscErrorCode randn(PetscScalar *randnArray, PetscInt n, PetscRandom ran)
{
  PetscInt       i;
  PetscScalar    u1,u2;
  PetscReal      t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<n; i+=2) {
    ierr = PetscRandomGetValue(ran,&u1);CHKERRQ(ierr);
    ierr = PetscRandomGetValue(ran,&u2);CHKERRQ(ierr);

    t               = PetscSqrtReal(-2*PetscLogReal(PetscRealPart(u1)));
    randnArray[i]   = t * PetscCosReal(2*PETSC_PI*PetscRealPart(u2));
    randnArray[i+1] = t * PetscSinReal(2*PETSC_PI*PetscRealPart(u2));
  }

  PetscFunctionReturn(0);
}

/* PetscErrorCode setQ(Vec *, Vec, PetscInt, PetscInt)
 * Generate Q such that q_0 = r_0, q_1, ..., q_{n - 1} is normally distributed
 */
PetscErrorCode setQ(Vec *q, Vec r, PetscInt n)
{
  PetscErrorCode ierr;
  PetscInt       i,m,*indices;
  PetscScalar    *randnArray;
  PetscRandom    ran;
  ierr = VecGetSize(r,&m);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&ran);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(ran);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&randnArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&indices);CHKERRQ(ierr);

  for (i = 0; i < m; ++i) indices[i] = i;

  for (i = 0; i < n; ++i) {
    if (i == 0) {
      VecCopy(r, q[0]);
    }
    else {
      ierr = randn(randnArray,m,ran);CHKERRQ(ierr);
      VecSetValues(q[i],m,indices,randnArray,INSERT_VALUES);
      VecAssemblyBegin(q[i]);
      VecAssemblyEnd(q[i]);
    }
  }

  ierr = PetscFree(randnArray);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetFromOptions_MLBCGS(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_MLBCGS     *mlbcgs = (KSP_MLBCGS *)ksp->data;
  PetscInt       this_n = mlbcgs->n;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP MLBCGS Options");CHKERRQ(ierr);

  /* Set n for ML(n)BiCGStab */
  ierr = PetscOptionsInt("-ksp_mlbcgs_n","Number of iterations to minimize ||r_k||_2","mlbcgs.c",this_n,&this_n,&flag);CHKERRQ(ierr);
  if (flag) {
    mlbcgs->n = this_n;
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSetUp_MLBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_MLBCGS     *mlbcgs = (KSP_MLBCGS *)ksp->data;
  PetscInt       n = mlbcgs->n;

  PetscFunctionBegin;
  ierr = KSPSetWorkVecs(ksp,n*4+6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_MLBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_MLBCGS     *mlbcgs = (KSP_MLBCGS*)ksp->data;
  PetscInt       i,j,n,s,idx;
  PetscScalar    alpha,*beta,*c,d1,d2,e,f,r_norm,omega;
  Vec            b,x,r,gt,u,ut,v,work,*D,*G,*Q,*W;

  PetscFunctionBegin;
  x     = ksp->vec_sol;
  b     = ksp->vec_rhs;
  n     = mlbcgs->n;

  /* Inititalize work vectors */
  idx   = 0;
  r     = ksp->work[idx++];
  gt    = ksp->work[idx++];
  u     = ksp->work[idx++];
  ut    = ksp->work[idx++];
  v     = ksp->work[idx++];
  work  = ksp->work[idx++];
  D     = ksp->work + idx; idx += n;
  G     = ksp->work + idx; idx += n;
  Q     = ksp->work + idx; idx += n;
  W     = ksp->work + idx;

  ierr = PetscMalloc1(n,&c);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&beta);CHKERRQ(ierr);

  /* Compute initial preconditioned residual (work, gt are work vectors here) */
  ierr = KSPInitialResidual(ksp,x,work,gt,r,b);CHKERRQ(ierr);                         /* r <- b - Ax */

  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(r,NORM_2,&r_norm);CHKERRQ(ierr);
    KSPCheckNorm(ksp,r_norm);
  } else {
    r_norm = 0.0;
  }
  ierr       = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = r_norm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
  ierr       = KSPLogResidualHistory(ksp,r_norm);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,r_norm);CHKERRQ(ierr);
  ierr       = (*ksp->converged)(ksp,0,r_norm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  /* Generate Q */
  ierr = setQ(Q,r,n);CHKERRQ(ierr);

  ierr = VecCopy(r,G[0]);CHKERRQ(ierr);                                             /* G[0] <- r */
  ierr = KSP_PCApply(ksp,G[0],gt);CHKERRQ(ierr);                                    /* gt <- inv(M) * G[0] */
  ierr = KSP_PCApplyBAorAB(ksp,G[0],W[0],work);CHKERRQ(ierr);                         /* W[0] <- A * inv(M) * G[0] */
  ierr = VecDot(Q[0],W[0],&c[0]);CHKERRQ(ierr);                                     /* c[0] <- <Q[0], W[0]> */
  ierr = VecDot(Q[0],r,&e);CHKERRQ(ierr);                                           /* e <- <Q[0], r> */

  KSPCheckDot(ksp,c[0]);
  if (c[0] == 0.0) {
    if (ksp->errorifnotconverged)
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve has not converged due to Nan or Inf inner product");
    else {
      ksp->reason = KSP_DIVERGED_NANORINF;
      PetscFunctionReturn(0);
    }
  }

  for (j = 0; j < ksp->max_it && ksp->its <= ksp->max_it; ++j) {
    alpha = e / c[0];
    ierr = VecWAXPY(u,-alpha,W[0],r);CHKERRQ(ierr);                                 /* u <- -alpha W[0] + r */
    ierr = VecAXPY(x,alpha,gt);CHKERRQ(ierr);                                       /* x <- x + alpha * gt */
    ierr = KSP_PCApply(ksp,u,ut);CHKERRQ(ierr);                                     /* ut <- inv(M) * u     */
    ierr = KSP_PCApplyBAorAB(ksp,u,v,work);CHKERRQ(ierr);                             /* v <- A * inv(M) * u     */
    ierr = VecDotNorm2(u,v,&d1,&d2);CHKERRQ(ierr);

    if (d2 == 0) {
      ierr = VecDot(u,u,&d1);CHKERRQ(ierr);
      if (d1 != 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN;
        break;
      }
      ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm  = 0.0;
      ksp->reason = KSP_CONVERGED_RTOL;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,r_norm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,ksp->its,ksp->rnorm);CHKERRQ(ierr);
      break;
    }

    omega = d1 / d2;                                                                /* omega <- <v, u> / <v, v>     */
    ierr = VecAXPY(x,omega,ut);CHKERRQ(ierr);                                       /* x <- x + omega ut     */
    ierr = VecWAXPY(r,-omega,v,u);CHKERRQ(ierr);                                    /* r <- -omega * v + u    */

    if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < ksp->its + 2) {
      ierr = VecNorm(r,NORM_2,&r_norm);CHKERRQ(ierr);
      KSPCheckNorm(ksp,r_norm);
    } else {
      r_norm = 0.0;
    }

    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = r_norm;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
    ierr = KSPLogResidualHistory(ksp,r_norm);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,ksp->its,r_norm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,ksp->its,r_norm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    for (i = 1; i <= n - 1 && ksp->its <= ksp->max_it; ++i) {
      ierr = VecDot(Q[i],u,&f);CHKERRQ(ierr);                                       /* f <- <Q[i], u>    */
      if (j >= 1) {
        beta[i] = - f / c[i];                                                       /* beta[i] <- -f / c[i]     */
        if (i <= n - 2) {
          ierr = VecAYPX(D[i],beta[i],u);CHKERRQ(ierr);
          ierr = VecScale(G[i],beta[i]);CHKERRQ(ierr);
          ierr = VecScale(W[i],beta[i]);CHKERRQ(ierr);
          ierr = VecDot(Q[i+1],D[i],&alpha);CHKERRQ(ierr);
          beta[i+1] = -alpha / c[i+1];
          for (s = i + 1; s <= n - 2; ++s) {
            ierr = VecAXPY(D[i],beta[s],D[s]);CHKERRQ(ierr);
            ierr = VecAXPY(G[i],beta[s],G[s]);CHKERRQ(ierr);
            ierr = VecAXPY(W[i],beta[s],W[s]);CHKERRQ(ierr);
            ierr = VecDot(Q[s+1],D[i],&alpha);CHKERRQ(ierr);
            beta[s+1] = -alpha / c[s+1];
          }
          ierr = VecAXPY(G[i],beta[n-1],G[n-1]);CHKERRQ(ierr);
          ierr = VecAXPY(W[i],beta[n-1],W[n-1]);CHKERRQ(ierr);
          ierr = VecAYPX(W[i],-omega,r);CHKERRQ(ierr);
        } else {
          ierr = VecScale(G[n-1],beta[n-1]);CHKERRQ(ierr);
          ierr = VecAYPX(W[n-1],-omega * beta[n-1],r);CHKERRQ(ierr);
        }
        ierr = VecDot(Q[0],W[i],&alpha);CHKERRQ(ierr);
        beta[0] = alpha / (omega * c[0]);
        ierr = VecAXPY(W[i],-omega * beta[0],W[0]);CHKERRQ(ierr);
        ierr = VecAXPBYPCZ(G[i],1.0,beta[0],1.0,W[i],G[0]);CHKERRQ(ierr);
      } else {
        ierr = VecDot(Q[0],r,&alpha);CHKERRQ(ierr);
        beta[0] = alpha / (omega * c[0]);
        ierr = VecWAXPY(W[i],-omega * beta[0],W[0],r);CHKERRQ(ierr);
        ierr = VecWAXPY(G[i],beta[0],G[0],W[i]);CHKERRQ(ierr);
      }
      for (s = 1; s <= i - 1; ++s) {
        ierr = VecDot(Q[s],W[i],&alpha);CHKERRQ(ierr);
        beta[s] = -alpha / c[s];
        ierr = VecAXPY(G[i],beta[s],G[s]);CHKERRQ(ierr);
        ierr = VecAXPY(W[i],beta[s],D[s]);CHKERRQ(ierr);
      }
      if (i < n - 1) {
        ierr = VecWAXPY(D[i],-1,u,W[i]);CHKERRQ(ierr);
        ierr = VecDot(Q[i],D[i],&c[i]);CHKERRQ(ierr);

        KSPCheckDot(ksp,c[0]);
        if (c[0] == 0.0) {
          if (ksp->errorifnotconverged)
            SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve has not converged due to Nan or Inf inner product");
          else {
            ksp->reason = KSP_DIVERGED_NANORINF;
            break;
          }
        }

        alpha = -f / c[i];
        ierr = VecAXPY(u,alpha,D[i]);CHKERRQ(ierr);
      } else {
        /* here work is a temparily work vector */
        ierr = VecCopy(W[i],work);CHKERRQ(ierr);
        ierr = VecAXPY(work,-1,u);CHKERRQ(ierr);
        ierr = VecDot(Q[i],work,&c[i]);CHKERRQ(ierr);

        KSPCheckDot(ksp,c[0]);
        if (c[0] == 0.0) {
          if (ksp->errorifnotconverged)
            SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve has not converged due to Nan or Inf inner product");
          else {
            ksp->reason = KSP_DIVERGED_NANORINF;
            break;
          }
        }

        alpha = -f / c[i];
      }
      ierr = KSP_PCApply(ksp,G[i],gt);CHKERRQ(ierr);
      ierr = KSP_PCApplyBAorAB(ksp,G[i],W[i],work);CHKERRQ(ierr);
      ierr = VecAXPY(x,omega*alpha,gt);CHKERRQ(ierr);
      ierr = VecAXPY(r,-omega*alpha,W[i]);CHKERRQ(ierr);

      if (ksp->normtype != KSP_NORM_NONE && ksp->chknorm < ksp->its + 2) {
        ierr = VecNorm(r,NORM_2,&r_norm);CHKERRQ(ierr);
        KSPCheckNorm(ksp,r_norm);
      } else {
       r_norm = 0.0;
      }

      ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
      ksp->its++;
      ksp->rnorm = r_norm;
      ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
      ierr = KSPLogResidualHistory(ksp,r_norm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,ksp->its,r_norm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,ksp->its,r_norm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      if (ksp->reason) break;
    }
    ierr = VecDot(Q[0],r,&e);CHKERRQ(ierr);
    beta[0] = e / (omega * c[0]);
    ierr = VecAYPX(W[0],-omega * beta[0],r);CHKERRQ(ierr);
    ierr = VecAYPX(G[0],beta[0],W[0]);CHKERRQ(ierr);

    if (n >= 2) {
      ierr = VecDot(Q[1],W[0],&alpha);CHKERRQ(ierr);
      beta[1] = -alpha / c[1];
      for (s = 1; s <= n - 2; ++s) {
        ierr = VecAXPY(G[0],beta[s],G[s]);CHKERRQ(ierr);
        ierr = VecAXPY(W[0],beta[s],D[s]);CHKERRQ(ierr);
        ierr = VecDot(Q[s+1],W[0],&alpha);CHKERRQ(ierr);
        beta[s+1] = -alpha / c[s+1];
      }
      ierr = VecAXPY(G[0],beta[n-1],G[n-1]);CHKERRQ(ierr);
    }
    ierr = KSP_PCApply(ksp,G[0],gt);CHKERRQ(ierr);
    ierr = KSP_PCApplyBAorAB(ksp,G[0],W[0],work);CHKERRQ(ierr);
    ierr = VecDot(Q[0],W[0],&c[0]);CHKERRQ(ierr);
  }

  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;

  ierr = PetscFree(c);CHKERRQ(ierr);
  ierr = PetscFree(beta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPReset_MLBCGS(KSP ksp)
{
  KSP_MLBCGS       *cg = (KSP_MLBCGS*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&cg->guess);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_MLBCGS(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_MLBCGS(ksp);CHKERRQ(ierr);
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPMLBCGS - Implements the ML(n)BiCGStab (Multiple Lanczos Stabilized version of BiConjugate Gradient) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes:
    See KSPMLBCGSL for additional stabilization
          Supports left and right preconditioning but not symmetric

   References:
   1. Yeung, M. (2012). ML(n) BiCGStab: Reformulation, Analysis and Implementation. Numerical Mathematics: Theory, Methods and Applications, 5(3), 447-492. doi:10.1017/S1004897900000891

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBICG, KSPMLBCGSL, KSPFBICG, KSPSetPCSide()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_MLBCGS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_MLBCGS    *mlbcgs;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&mlbcgs);CHKERRQ(ierr);

  ksp->data                = mlbcgs;
  ksp->ops->setup          = KSPSetUp_MLBCGS;
  ksp->ops->solve          = KSPSolve_MLBCGS;
  ksp->ops->destroy        = KSPDestroy_MLBCGS;
  ksp->ops->reset          = KSPReset_MLBCGS;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_MLBCGS;

  /* Let the user pick n */
  mlbcgs->n = 2;
  ksp->pc_side = PC_RIGHT;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_RIGHT,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
