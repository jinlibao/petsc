/*  The LGMRES method 

Contributed by: Allison Baker

Augments the standard GMRES approximation space with approximation to
the error from previous restart cycles.

Can be combined with left or right preconditioning.

Described in:
A. H. Baker, E.R. Jessup, and T.A. Manteuffel. A technique for
accelerating the convergence of restarted GMRES. Submitted to SIAM
Journal on Matrix Analysis and Applications. Also available as
Technical Report #CU-CS-945-03, University of Colorado, Department of
Computer Science, January, 2003. 

*/

#include "lgmresp.h"   /*I allipetscksp.h I*/

#define LGMRES_DELTA_DIRECTIONS 10
#define LGMRES_DEFAULT_MAXK     30
#define LGMRES_DEFAULT_AUGDIM   2 /*default number of augmentation vectors */ 
static int    LGMRESGetNewVectors(KSP,int);
static int    LGMRESUpdateHessenberg(KSP,int,PetscTruth,PetscReal *);
static int    BuildLgmresSoln(PetscScalar*,Vec,Vec,KSP,int);

/*
    KSPSetUp_LGMRES - Sets up the workspace needed by lgmres.

    This is called once, usually automatically by SLESSolve() or SLESSetUp(),
    but can be called directly by KSPSetUp().

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_LGMRES"
int    KSPSetUp_LGMRES(KSP ksp)
{
  unsigned  int size,hh,hes,rs,cc;
  int           ierr,max_k,k, aug_dim;
  KSP_LGMRES    *lgmres = (KSP_LGMRES *)ksp->data;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,"no symmetric preconditioning for KSPLGMRES");
  }



  max_k         = lgmres->max_k;
  aug_dim       = lgmres->aug_dim;
  hh            = (max_k + 2) * (max_k + 1);
  hes           = (max_k + 1) * (max_k + 1);
  rs            = (max_k + 2);
  cc            = (max_k + 1);  /* SS and CC are the same size */
  size          = (hh + hes + rs + 2*cc) * sizeof(PetscScalar);

  /* Allocate space and set pointers to beginning */
  ierr = PetscMalloc(size,&lgmres->hh_origin);CHKERRQ(ierr);
  ierr = PetscMemzero(lgmres->hh_origin,size);CHKERRQ(ierr); 
  PetscLogObjectMemory(ksp,size);                      /* HH - modified (by plane 
                                                      rotations) hessenburg */
  lgmres->hes_origin = lgmres->hh_origin + hh;     /* HES - unmodified hessenburg */
  lgmres->rs_origin  = lgmres->hes_origin + hes;   /* RS - the right-hand-side of the 
                                                      Hessenberg system */
  lgmres->cc_origin  = lgmres->rs_origin + rs;     /* CC - cosines for rotations */
  lgmres->ss_origin  = lgmres->cc_origin + cc;     /* SS - sines for rotations */

  if (ksp->calc_sings) {
    /* Allocate workspace to hold Hessenberg matrix needed by Eispack */
    size = (max_k + 3)*(max_k + 9)*sizeof(PetscScalar);
    ierr = PetscMalloc(size,&lgmres->Rsvd);CHKERRQ(ierr);
    ierr = PetscMalloc(5*(max_k+2)*sizeof(PetscReal),&lgmres->Dsvd);CHKERRQ(ierr);
    PetscLogObjectMemory(ksp,size+5*(max_k+2)*sizeof(PetscReal));
  }

  /* Allocate array to hold pointers to user vectors.  Note that we need
  we need it+1 vectors, and it <= max_k)  - vec_offset indicates some initial work vectors*/
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *),&lgmres->vecs);CHKERRQ(ierr);
  lgmres->vecs_allocated = VEC_OFFSET + 2 + max_k;
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(void *),&lgmres->user_work);CHKERRQ(ierr);
  ierr = PetscMalloc((VEC_OFFSET+2+max_k)*sizeof(int),&lgmres->mwork_alloc);CHKERRQ(ierr);
  PetscLogObjectMemory(ksp,(VEC_OFFSET+2+max_k)*(2*sizeof(void *)+sizeof(int)));

  /* LGMRES_MOD: need array of pointers to augvecs*/
  ierr = PetscMalloc((2 * aug_dim + AUG_OFFSET)*sizeof(void *),&lgmres->augvecs);CHKERRQ(ierr);
  lgmres->aug_vecs_allocated = 2 *aug_dim + AUG_OFFSET;
  ierr = PetscMalloc((2* aug_dim + AUG_OFFSET)*sizeof(void *),&lgmres->augvecs_user_work);CHKERRQ(ierr);
  ierr = PetscMalloc(aug_dim*sizeof(int),&lgmres->aug_order);CHKERRQ(ierr);
  PetscLogObjectMemory(ksp,(aug_dim)*(4*sizeof(void *) + sizeof(int)) + AUG_OFFSET*2*sizeof(void *) );

 
 /* if q_preallocate = 0 then only allocate one "chunk" of space (for 
     5 vectors) - additional will then be allocated from LGMREScycle() 
     as needed.  Otherwise, allocate all of the space that could be needed */
  if (lgmres->q_preallocate) {
    lgmres->vv_allocated   = VEC_OFFSET + 2 + max_k;
    ierr = VecDuplicateVecs(VEC_RHS,lgmres->vv_allocated,&lgmres->user_work[0]);CHKERRQ(ierr);
    PetscLogObjectParents(ksp,lgmres->vv_allocated,lgmres->user_work[0]);
    lgmres->mwork_alloc[0] = lgmres->vv_allocated;
    lgmres->nwork_alloc    = 1;
    for (k=0; k<lgmres->vv_allocated; k++) {
      lgmres->vecs[k] = lgmres->user_work[0][k];
    }
  } else {
    lgmres->vv_allocated    = 5;
    ierr = VecDuplicateVecs(ksp->vec_rhs,5,&lgmres->user_work[0]);CHKERRQ(ierr);
    PetscLogObjectParents(ksp,5,lgmres->user_work[0]);
    lgmres->mwork_alloc[0]  = 5;
    lgmres->nwork_alloc     = 1;
    for (k=0; k<lgmres->vv_allocated; k++) {
      lgmres->vecs[k] = lgmres->user_work[0][k];
    }
  }
  /* LGMRES_MOD - for now we will preallocate the augvecs - because aug_dim << restart
     ... also keep in mind that we need to keep augvecs from cycle to cycle*/  
  lgmres->aug_vv_allocated = 2* aug_dim + AUG_OFFSET;
  lgmres->augwork_alloc =  2* aug_dim + AUG_OFFSET;
  ierr = VecDuplicateVecs(VEC_RHS,lgmres->aug_vv_allocated,&lgmres->augvecs_user_work[0]);CHKERRQ(ierr);
  PetscLogObjectParents(ksp,lgmres->aug_vv_allocated,lgmres->augvecs_user_work[0]);
  for (k=0; k<lgmres->aug_vv_allocated; k++) {
      lgmres->augvecs[k] = lgmres->augvecs_user_work[0][k];
    }

  PetscFunctionReturn(0);
}


/*

    LGMRESCycle - Run lgmres, possibly with restart.  Return residual 
                  history if requested.

    input parameters:
.	 lgmres  - structure containing parameters and work areas

    output parameters:
.        nres    - residuals (from preconditioned system) at each step.
                  If restarting, consider passing nres+it.  If null, 
                  ignored
.        itcount - number of iterations used.   nres[0] to nres[itcount]
                  are defined.  If null, ignored.  If null, ignored.
.        converged - 0 if not converged

		  
    Notes:
    On entry, the value in vector VEC_VV(0) should be 
    the initial residual.


 */
#undef __FUNCT__  
#define __FUNCT__ "LGMREScycle"
int LGMREScycle(int *itcount,KSP ksp)
{

  KSP_LGMRES   *lgmres = (KSP_LGMRES *)(ksp->data);
  PetscReal    res_norm, res;             
  PetscReal    hapbnd, tt;
  PetscScalar  zero = 0.0;
  PetscScalar  tmp;
  PetscTruth   hapend = PETSC_FALSE;  /* indicates happy breakdown ending */
  int          ierr;
  int          loc_it;                /* local count of # of dir. in Krylov space */ 
  int          max_k = lgmres->max_k; /* max approx space size */
  int          max_it = ksp->max_it;  /* max # of overall iterations for the method */ 
  /* LGMRES_MOD - new variables*/
  int          aug_dim = lgmres->aug_dim;
  int          spot = 0;
  int          order = 0;
  int          it_arnoldi;             /* number of arnoldi steps to take */
  int          it_total;               /* total number of its to take (=approx space size)*/ 
  int          ii, jj;
  PetscReal    tmp_norm; 
  PetscScalar  inv_tmp_norm; 
  PetscScalar  *avec; 

  PetscFunctionBegin;

  /* Number of pseudo iterations since last restart is the number 
     of prestart directions */
  loc_it = 0;

  /* LGMRES_MOD: determine number of arnoldi steps to take */
  /* if approx_constant then we keep the space the same size even if 
     we don't have the full number of aug vectors yet*/
  if (lgmres->approx_constant) {
     it_arnoldi = max_k - lgmres->aug_ct;
  } else {
      it_arnoldi = max_k - aug_dim; 
  }

  it_total =  it_arnoldi + lgmres->aug_ct;


  /* initial residual is in VEC_VV(0)  - compute its norm*/ 
  ierr   = VecNorm(VEC_VV(0),NORM_2,&res_norm);CHKERRQ(ierr);
  res    = res_norm;     
 
  /* first entry in right-hand-side of hessenberg system is just 
     the initial residual norm */
  *GRS(0) = res_norm;

 /* check for the convergence */
  if (!res) {
     if (itcount) *itcount = 0;
     ksp->reason = KSP_CONVERGED_ATOL;
     PetscLogInfo(ksp,"GMRESCycle: Converged due to zero residual norm on entry\n");
     PetscFunctionReturn(0);
  }

  /* scale VEC_VV (the initial residual) */
  tmp = 1.0/res_norm; ierr = VecScale(&tmp,VEC_VV(0));CHKERRQ(ierr);

  /* FYI: AMS calls are for memory snooper */
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->rnorm = res;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);


  /* note: (lgmres->it) is always set one less than (loc_it) It is used in 
     KSPBUILDSolution_LGMRES, where it is passed to BuildLgmresSoln.  
     Note that when BuildLgmresSoln is called from this function, 
     (loc_it -1) is passed, so the two are equivalent */
  lgmres->it = (loc_it - 1);

   
  /* MAIN ITERATION LOOP BEGINNING*/


  /* keep iterating until we have converged OR generated the max number
     of directions OR reached the max number of iterations for the method */ 
  ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
 
  while (!ksp->reason && loc_it < it_total && ksp->its < max_it) { /* LGMRES_MOD: changed to it_total */
     KSPLogResidualHistory(ksp,res);
     lgmres->it = (loc_it - 1);
     KSPMonitor(ksp,ksp->its,res); 

    /* see if more space is needed for work vectors */
    if (lgmres->vv_allocated <= loc_it + VEC_OFFSET + 1) {
       ierr = LGMRESGetNewVectors(ksp,loc_it+1);CHKERRQ(ierr);
      /* (loc_it+1) is passed in as number of the first vector that should
         be allocated */
    }

    /*LGMRES_MOD: decide whether this is an arnoldi step or an aug step */ 
    if (loc_it < it_arnoldi) { /* arnoldi */
       ierr = KSP_PCApplyBAorAB(ksp,ksp->B,ksp->pc_side,VEC_VV(loc_it),VEC_VV(1+loc_it),VEC_TEMP_MATOP);CHKERRQ(ierr);
    } else { /*aug step */
       order = loc_it - it_arnoldi + 1; /* which aug step */ 
       for (ii=0; ii<aug_dim; ii++) {
           if (lgmres->aug_order[ii] == order) {
              spot = ii;
              break; /* must have this because there will be duplicates before aug_ct = aug_dim */ 
            }  
        }

       ierr = VecCopy(A_AUGVEC(spot), VEC_VV(1+loc_it)); CHKERRQ(ierr); 
       /*note: an alternate implementation choice would be to only save the AUGVECS and
         not A_AUGVEC and then apply the PC here to the augvec */
    }

    /* update hessenberg matrix and do Gram-Schmidt - new direction is in
       VEC_VV(1+loc_it)*/
    ierr = (*lgmres->orthog)(ksp,loc_it);CHKERRQ(ierr);

    /* new entry in hessenburg is the 2-norm of our new direction */
    ierr = VecNorm(VEC_VV(loc_it+1),NORM_2,&tt);CHKERRQ(ierr);
    *HH(loc_it+1,loc_it)   = tt;
    *HES(loc_it+1,loc_it)  = tt;


    /* check for the happy breakdown */
    hapbnd  = PetscAbsScalar(tt / *GRS(loc_it));/* GRS(loc_it) contains the res_norm from the last iteration  */
    if (hapbnd > lgmres->haptol) hapbnd = lgmres->haptol;
    if (tt > hapbnd) {
       tmp = 1.0/tt; 
       ierr = VecScale(&tmp,VEC_VV(loc_it+1));CHKERRQ(ierr); /* scale new direction by its norm */
    } else {
       PetscLogInfo(ksp,"Detected happy breakdown, current hapbnd = %g tt = %g\n",hapbnd,tt);
       hapend = PETSC_TRUE;
    }

    /* Now apply rotations to new col of hessenberg (and right side of system), 
       calculate new rotation, and get new residual norm at the same time*/
    ierr = LGMRESUpdateHessenberg(ksp,loc_it,hapend,&res);CHKERRQ(ierr);
    loc_it++;
    lgmres->it  = (loc_it-1);  /* Add this here in case it has converged */
 
    ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
    ksp->its++;
    ksp->rnorm = res;
    ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

    ierr = (*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);

    /* Catch error in happy breakdown and signal convergence and break from loop */
    if (hapend) {
      if (!ksp->reason) {
        SETERRQ1(0,"You reached the happy break down,but convergence was not indicated. Residual norm = %g",res);
      }
      break;
    }
  }
  /* END OF ITERATION LOOP */

  KSPLogResidualHistory(ksp,res);

  /* Monitor if we know that we will not return for a restart */
  if (ksp->reason || ksp->its >= max_it) {
    KSPMonitor(ksp, ksp->its, res);
  }

  if (itcount) *itcount    = loc_it;

  /*
    Down here we have to solve for the "best" coefficients of the Krylov
    columns, add the solution values together, and possibly unwind the
    preconditioning from the solution
   */
 
  /* Form the solution (or the solution so far) */
  /* Note: must pass in (loc_it-1) for iteration count so that BuildLgmresSoln
     properly navigates */

  ierr = BuildLgmresSoln(GRS(0),VEC_SOLN,VEC_SOLN,ksp,loc_it-1);CHKERRQ(ierr);


  /* LGMRES_MOD collect aug vector and A*augvector for future restarts -
     only if we will be restarting (i.e. this cycle performed it_total
     iterations)  */
  if (!ksp->reason && ksp->its < max_it && aug_dim > 0) {

     /*AUG_TEMP contains the new augmentation vector (assigned in  BuildLgmresSoln) */
    if (lgmres->aug_ct == 0) {
        spot = 0;
        lgmres->aug_ct++;
     } else if (lgmres->aug_ct < aug_dim) {
        spot = lgmres->aug_ct;
        lgmres->aug_ct++;
     } else { /* truncate */
        for (ii=0; ii<aug_dim; ii++) {
           if (lgmres->aug_order[ii] == aug_dim) {
              spot = ii;
            }  
        }
     } 

     

     ierr = VecCopy(AUG_TEMP, AUGVEC(spot)); CHKERRQ(ierr); 
     /*need to normalize */
     ierr = VecNorm(AUGVEC(spot), NORM_2, &tmp_norm); CHKERRQ(ierr);
     inv_tmp_norm = 1.0/tmp_norm;
     ierr = VecScale(&inv_tmp_norm, AUGVEC(spot)); CHKERRQ(ierr); 

     /*set new aug vector to order 1  - move all others back one */
     for (ii=0; ii < aug_dim; ii++) {
        AUG_ORDER(ii)++;
     } 
     AUG_ORDER(spot) = 1; 

     /*now add the A*aug vector to A_AUGVEC(spot)  - this is independ. of preconditioning type*/
     /* want V*H*y - y is in GRS, V is in VEC_VV and H is in HES */

 
     /* first do H+*y */
     ierr = VecSet(&zero, AUG_TEMP); CHKERRQ(ierr);
     VecGetArray(AUG_TEMP, &avec);
     for (ii=0; ii < it_total + 1; ii++) {
        for (jj=0; jj <= ii+1; jj++) {
           avec[jj] += *HES(jj ,ii) * *GRS(ii);
        }
     }

     /*now multiply result by V+ */
     ierr = VecSet(&zero, VEC_TEMP);
     ierr = VecMAXPY(it_total+1, avec, VEC_TEMP, &VEC_VV(0)); /*answer is in VEC_TEMP*/
     VecRestoreArray(AUG_TEMP, &avec);
  
     /*copy answer to aug location  and scale*/
     VecCopy(VEC_TEMP,  A_AUGVEC(spot)); 
     ierr = VecScale(&inv_tmp_norm, A_AUGVEC(spot)); CHKERRQ(ierr); 


  }
  PetscFunctionReturn(0);
}

/*  
    KSPSolve_LGMRES - This routine applies the LGMRES method.


   Input Parameter:
.     ksp - the Krylov space object that was set to use lgmres

   Output Parameter:
.     outits - number of iterations used

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_LGMRES"

int KSPSolve_LGMRES(KSP ksp)
{
  int        ierr;
  int        cycle_its; /* iterations done in a call to LGMREScycle */
  int        itcount;   /* running total of iterations, incl. those in restarts */
  KSP_LGMRES *lgmres = (KSP_LGMRES *)ksp->data;
  PetscTruth guess_zero = ksp->guess_zero;
  /*LGMRES_MOD variable */
  int ii;

  PetscFunctionBegin;
  if (ksp->calc_sings && !lgmres->Rsvd) {
     SETERRQ(1,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");
  }
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);

  /* initialize */
  itcount  = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  /*LGMRES_MOD*/
  for (ii=0; ii<lgmres->aug_dim; ii++) {
     lgmres->aug_order[ii] = 0;
  }

  while (!ksp->reason) {
     /* calc residual - puts in VEC_VV(0) */
    ierr     = KSPInitialResidual(ksp,VEC_SOLN,VEC_TEMP,VEC_TEMP_MATOP,VEC_VV(0),VEC_RHS);CHKERRQ(ierr);
    ierr     = LGMREScycle(&cycle_its,ksp);CHKERRQ(ierr);
    itcount += cycle_its;  
    if (itcount >= ksp->max_it) {
      ksp->reason = KSP_DIVERGED_ITS;
      break;
    }
    ksp->guess_zero = PETSC_FALSE; /* every future call to KSPInitialResidual() will have nonzero guess */
  }
  ksp->guess_zero = guess_zero; /* restore if user provided nonzero initial guess */
  PetscFunctionReturn(0);
}

/*

   KSPDestroy_LGMRES - Frees all memory space used by the Krylov method.

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_LGMRES" 
int KSPDestroy_LGMRES(KSP ksp)
{
  KSP_LGMRES *lgmres = (KSP_LGMRES*)ksp->data;
  int       i,ierr;

  PetscFunctionBegin;
  /* Free the Hessenberg matrices */
  if (lgmres->hh_origin) {ierr = PetscFree(lgmres->hh_origin);CHKERRQ(ierr);}

  /* Free pointers to user variables */
  if (lgmres->vecs) {ierr = PetscFree(lgmres->vecs);CHKERRQ(ierr);}

  /*LGMRES_MOD - free pointers for extra vectors */ 
  if (lgmres->augvecs) {ierr = PetscFree(lgmres->augvecs);CHKERRQ(ierr);}

  /* free work vectors */
  for (i=0; i < lgmres->nwork_alloc; i++) {
    ierr = VecDestroyVecs(lgmres->user_work[i],lgmres->mwork_alloc[i]);CHKERRQ(ierr);
  }
  if (lgmres->user_work)  {ierr = PetscFree(lgmres->user_work);CHKERRQ(ierr);}

  /*LGMRES_MOD - free aug work vectors also */
  /*this was all allocated as one "chunk" */
  ierr = VecDestroyVecs(lgmres->augvecs_user_work[0],lgmres->augwork_alloc);CHKERRQ(ierr);
  if (lgmres->augvecs_user_work)  {ierr = PetscFree(lgmres->augvecs_user_work);CHKERRQ(ierr);}
  if (lgmres->aug_order) {ierr = PetscFree(lgmres->aug_order);CHKERRQ(ierr);}

  if (lgmres->mwork_alloc) {ierr = PetscFree(lgmres->mwork_alloc);CHKERRQ(ierr);}
  if (lgmres->nrs) {ierr = PetscFree(lgmres->nrs);CHKERRQ(ierr);}
  if (lgmres->sol_temp) {ierr = VecDestroy(lgmres->sol_temp);CHKERRQ(ierr);}
  if (lgmres->Rsvd) {ierr = PetscFree(lgmres->Rsvd);CHKERRQ(ierr);}
  if (lgmres->Dsvd) {ierr = PetscFree(lgmres->Dsvd);CHKERRQ(ierr);}
  ierr = PetscFree(lgmres);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    BuildLgmresSoln - create the solution from the starting vector and the
                      current iterates.

    Input parameters:
        nrs - work area of size it + 1.
	vguess  - index of initial guess
	vdest - index of result.  Note that vguess may == vdest (replace
	        guess with the solution).
        it - HH upper triangular part is a block of size (it+1) x (it+1)  

     This is an internal routine that knows about the LGMRES internals.
 */
#undef __FUNCT__  
#define __FUNCT__ "BuildLgmresSoln"
static int BuildLgmresSoln(PetscScalar* nrs,Vec vguess,Vec vdest,KSP ksp,int it)
{
  PetscScalar  tt,zero = 0.0,one = 1.0;
  int          ierr,ii,k,j;
  KSP_LGMRES   *lgmres = (KSP_LGMRES *)(ksp->data);
  /*LGMRES_MOD */
  int          it_arnoldi, it_aug; 
  int          jj, spot = 0; 

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no lgmres steps have been performed */
  if (it < 0) {
    if (vdest != vguess) {
      ierr = VecCopy(vguess,vdest);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  /* so (it+1) lgmres steps HAVE been performed */

  /* LGMRES_MOD - determine if we need to use augvecs for the soln  - do not assume that
     this is called after the total its allowed for an approx space */
   if (lgmres->approx_constant) {
     it_arnoldi = lgmres->max_k - lgmres->aug_ct;
   } else {
     it_arnoldi = lgmres->max_k - lgmres->aug_dim; 
   }
   if (it_arnoldi >= it +1) {
      it_aug = 0;
      it_arnoldi = it+1;
   } else {
      it_aug = (it + 1) - it_arnoldi;   
   }

  /* now it_arnoldi indicates the number of matvecs that took place */
  lgmres->matvecs += it_arnoldi;

 
  /* solve the upper triangular system - GRS is the right side and HH is 
     the upper triangular matrix  - put soln in nrs */
  if (*HH(it,it) == 0.0) SETERRQ2(1,"HH(it,it) is identically zero; it = %d GRS(it) = %g",it,PetscAbsScalar(*GRS(it)));
  if (*HH(it,it) != 0.0) {
     nrs[it] = *GRS(it) / *HH(it,it);
  } else {
     nrs[it] = 0.0;
  }

  for (ii=1; ii<=it; ii++) {
    k   = it - ii;
    tt  = *GRS(k);
    for (j=k+1; j<=it; j++) tt  = tt - *HH(k,j) * nrs[j];
    nrs[k]   = tt / *HH(k,k);
  }

  /* Accumulate the correction to the soln of the preconditioned prob. in VEC_TEMP */
  ierr = VecSet(&zero,VEC_TEMP);CHKERRQ(ierr); /* set VEC_TEMP components to 0 */

  /*LGMRES_MOD - if augmenting has happened we need to form the solution 
    using the augvecs */
  if (it_aug == 0) { /* all its are from arnoldi */
     ierr = VecMAXPY(it+1,nrs,VEC_TEMP,&VEC_VV(0));CHKERRQ(ierr); 
  } else { /*use aug vecs */ 
     /*first do regular krylov directions */
     ierr = VecMAXPY(it_arnoldi,nrs,VEC_TEMP,&VEC_VV(0));CHKERRQ(ierr); 
     /*now add augmented portions - add contribution of aug vectors one at a time*/


     for (ii=0; ii<it_aug; ii++) {
        for (jj=0; jj<lgmres->aug_dim; jj++) {
           if (lgmres->aug_order[jj] == (ii+1)) {
              spot = jj;
              break; /* must have this because there will be duplicates before aug_ct = aug_dim */ 
            }  
        }
        ierr = VecAXPY(&nrs[it_arnoldi+ii],AUGVEC(spot),VEC_TEMP);CHKERRQ(ierr); 
      }
  }
  /* now VEC_TEMP is what we want to keep for augmenting purposes - grab before the
     preconditioner is "unwound" from right-precondtioning*/
  ierr = VecCopy(VEC_TEMP, AUG_TEMP); CHKERRQ(ierr); 

  ierr = KSPUnwindPreconditioner(ksp,VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);

  /* add solution to previous solution */
  /* put updated solution into vdest.*/
  if (vdest != vguess) {
    ierr = VecCopy(VEC_TEMP,vdest);CHKERRQ(ierr);
  }
  ierr = VecAXPY(&one,VEC_TEMP,vdest);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*

    LGMRESUpdateHessenberg - Do the scalar work for the orthogonalization.  
                            Return new residual.

    input parameters:

.        ksp -    Krylov space object
.	 it  -    plane rotations are applied to the (it+1)th column of the 
                  modified hessenberg (i.e. HH(:,it))
.        hapend - PETSC_FALSE not happy breakdown ending.

    output parameters:
.        res - the new residual
	
 */
#undef __FUNCT__  
#define __FUNCT__ "LGMRESUpdateHessenberg"
static int LGMRESUpdateHessenberg(KSP ksp,int it,PetscTruth hapend,PetscReal *res)
{
  PetscScalar   *hh,*cc,*ss,tt;
  int           j;
  KSP_LGMRES    *lgmres = (KSP_LGMRES *)(ksp->data);

  PetscFunctionBegin;
  hh  = HH(0,it);  /* pointer to beginning of column to update - so 
                      incrementing hh "steps down" the (it+1)th col of HH*/ 
  cc  = CC(0);     /* beginning of cosine rotations */ 
  ss  = SS(0);     /* beginning of sine rotations */

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  /* Note: this uses the rotation [conj(c)  s ; -s   c], c= cos(theta), s= sin(theta) */

  for (j=1; j<=it; j++) {
    tt  = *hh;
#if defined(PETSC_USE_COMPLEX)
    *hh = PetscConj(*cc) * tt + *ss * *(hh+1);
#else
    *hh = *cc * tt + *ss * *(hh+1);
#endif
    hh++;
    *hh = *cc++ * *hh - (*ss++ * tt);
    /* hh, cc, and ss have all been incremented one by end of loop */
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system (GRS)
        note: it affects GRS(it) and GRS(it+1)
     2) the new column of the Hessenberg matrix
        note: it affects HH(it,it) which is currently pointed to 
        by hh and HH(it+1, it) (*(hh+1))  
    thus obtaining the updated value of the residual...
  */

  /* compute new plane rotation */

  if (!hapend) {
#if defined(PETSC_USE_COMPLEX)
    tt        = PetscSqrtScalar(PetscConj(*hh) * *hh + PetscConj(*(hh+1)) * *(hh+1));
#else
    tt        = PetscSqrtScalar(*hh * *hh + *(hh+1) * *(hh+1));
#endif
    if (tt == 0.0) {SETERRQ(PETSC_ERR_KSP_BRKDWN,"Your matrix or preconditioner is the null operator");}
    *cc       = *hh / tt;   /* new cosine value */
    *ss       = *(hh+1) / tt;  /* new sine value */

    /* apply to 1) and 2) */
    *GRS(it+1) = - (*ss * *GRS(it));
#if defined(PETSC_USE_COMPLEX)
    *GRS(it)   = PetscConj(*cc) * *GRS(it);
    *hh        = PetscConj(*cc) * *hh + *ss * *(hh+1);
#else
    *GRS(it)   = *cc * *GRS(it);
    *hh        = *cc * *hh + *ss * *(hh+1);
#endif

    /* residual is the last element (it+1) of right-hand side! */
    *res      = PetscAbsScalar(*GRS(it+1));

  } else { /* happy breakdown: HH(it+1, it) = 0, therfore we don't need to apply 
            another rotation matrix (so RH doesn't change).  The new residual is 
            always the new sine term times the residual from last time (GRS(it)), 
            but now the new sine rotation would be zero...so the residual should
            be zero...so we will multiply "zero" by the last residual.  This might
            not be exactly what we want to do here -could just return "zero". */
 
    *res = 0.0;
  }
  PetscFunctionReturn(0);
}

/*

   LGMRESGetNewVectors - This routine allocates more work vectors, starting from 
                         VEC_VV(it) 
                         
*/
#undef __FUNCT__  
#define __FUNCT__ "LGMRESGetNewVectors" 
static int LGMRESGetNewVectors(KSP ksp,int it)
{
  KSP_LGMRES *lgmres = (KSP_LGMRES *)ksp->data;
  int        nwork = lgmres->nwork_alloc; /* number of work vector chunks allocated */
  int        nalloc;                      /* number to allocate */
  int        k,ierr;
 
  PetscFunctionBegin;
  nalloc = lgmres->delta_allocate; /* number of vectors to allocate 
                                      in a single chunk */

  /* Adjust the number to allocate to make sure that we don't exceed the
     number of available slots (lgmres->vecs_allocated)*/
  if (it + VEC_OFFSET + nalloc >= lgmres->vecs_allocated){
    nalloc = lgmres->vecs_allocated - it - VEC_OFFSET;
  }
  if (!nalloc) PetscFunctionReturn(0);

  lgmres->vv_allocated += nalloc; /* vv_allocated is the number of vectors allocated */

  /* work vectors */
  ierr = VecDuplicateVecs(ksp->vec_rhs,nalloc,&lgmres->user_work[nwork]);CHKERRQ(ierr);
  PetscLogObjectParents(ksp,nalloc,lgmres->user_work[nwork]); 
  /* specify size of chunk allocated */
  lgmres->mwork_alloc[nwork] = nalloc;

  for (k=0; k < nalloc; k++) {
    lgmres->vecs[it+VEC_OFFSET+k] = lgmres->user_work[nwork][k];
  }
 

  /* LGMRES_MOD - for now we are preallocating the augmentation vectors */
  

  /* increment the number of work vector chunks */
  lgmres->nwork_alloc++;
  PetscFunctionReturn(0);
}

/* 

   KSPBuildSolution_LGMRES

     Input Parameter:
.     ksp - the Krylov space object
.     ptr-

   Output Parameter:
.     result - the solution

   Note: this calls BuildLgmresSoln - the same function that LGMREScycle
   calls directly.  

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPBuildSolution_LGMRES"
int KSPBuildSolution_LGMRES(KSP ksp,Vec ptr,Vec *result)
{
  KSP_LGMRES *lgmres = (KSP_LGMRES *)ksp->data; 
  int        ierr;

  PetscFunctionBegin;
  if (!ptr) {
    if (!lgmres->sol_temp) {
      ierr = VecDuplicate(ksp->vec_sol,&lgmres->sol_temp);CHKERRQ(ierr);
      PetscLogObjectParent(ksp,lgmres->sol_temp);
    }
    ptr = lgmres->sol_temp;
  }
  if (!lgmres->nrs) {
    /* allocate the work area */
    ierr = PetscMalloc(lgmres->max_k*sizeof(PetscScalar),&lgmres->nrs);CHKERRQ(ierr);
    PetscLogObjectMemory(ksp,lgmres->max_k*sizeof(PetscScalar));
  }
 
  ierr = BuildLgmresSoln(lgmres->nrs,VEC_SOLN,ptr,ksp,lgmres->it);CHKERRQ(ierr);
  *result = ptr; 
  
  PetscFunctionReturn(0);
}

/*

   KSPView_LGMRES -Prints information about the current Krylov method 
                  being used.

 */
#undef __FUNCT__  
#define __FUNCT__ "KSPView_LGMRES" 
int KSPView_LGMRES(KSP ksp,PetscViewer viewer)
{
  KSP_LGMRES   *lgmres = (KSP_LGMRES *)ksp->data; 
  char         *cstr;
  int          ierr;
  PetscTruth   isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (lgmres->orthog == KSPGMRESClassicalGramSchmidtOrthogonalization) {
    if (lgmres->cgstype == KSP_GMRES_CGS_REFINEMENT_NONE) {
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with no iterative refinement";
    } else if (lgmres->cgstype == KSP_GMRES_CGS_REFINEMENT_ALWAYS) {
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with one step of iterative refinement";
    } else {
      cstr = "Classical (unmodified) Gram-Schmidt Orthogonalization with one step of iterative refinement when needed";
    }
  } else if (lgmres->orthog == KSPGMRESModifiedGramSchmidtOrthogonalization) {
    cstr = "Modified Gram-Schmidt Orthogonalization";
  } else {
    cstr = "unknown orthogonalization";
  }
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  LGMRES: restart=%d, using %s\n",lgmres->max_k,cstr);CHKERRQ(ierr);
    /*LGMRES_MOD */
    ierr = PetscViewerASCIIPrintf(viewer,"  LGMRES: aug. dimension=%d\n",lgmres->aug_dim);CHKERRQ(ierr);
    if (lgmres->approx_constant) {
       ierr = PetscViewerASCIIPrintf(viewer,"  LGMRES: approx. space size was kept constant.\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  LGMRES: number of matvecs=%d\n",lgmres->matvecs);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  LGMRES: happy breakdown tolerance %g\n",lgmres->haptol);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer,"%s restart %d",cstr,lgmres->max_k);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for KSP LGMRES",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);


}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_LGMRES"
int KSPSetFromOptions_LGMRES(KSP ksp)
{
  int         ierr, restart, aug,indx;
  PetscReal   haptol;
  KSP_LGMRES *lgmres = (KSP_LGMRES*) ksp->data;
  PetscTruth  flg;
  char        *types[] = {"none","ifneeded","always"};

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP LGMRES Options");CHKERRQ(ierr);

    

    ierr = PetscOptionsInt("-ksp_gmres_restart","For LGMRES, this is the maximum size of the approximation space","KSPGMRESSetRestart",lgmres->max_k,&restart,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGMRESSetRestart(ksp,restart);CHKERRQ(ierr); }
    ierr = PetscOptionsReal("-ksp_gmres_haptol","Tolerance for declaring exact convergence (happy ending)","KSPGMRESSetHapTol",lgmres->haptol,&haptol,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPGMRESSetHapTol(ksp,haptol);CHKERRQ(ierr); }
    ierr = PetscOptionsName("-ksp_gmres_preallocate","Preallocate all Krylov vectors","KSPGMRESSetPreAllocateVectors",&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetPreAllocateVectors(ksp);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroupBegin("-ksp_gmres_classicalgramschmidt","Use classical (unmodified) Gram-Schmidt (fast)","KSPGMRESSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESClassicalGramSchmidtOrthogonalization);CHKERRQ(ierr);}
    ierr = PetscOptionsLogicalGroup("-ksp_gmres_modifiedgramschmidt","Use modified Gram-Schmidt (slow but more stable)","KSPGMRESSetOrthogonalization",&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPGMRESSetOrthogonalization(ksp,KSPGMRESModifiedGramSchmidtOrthogonalization);CHKERRQ(ierr);}
    ierr = PetscOptionsEList("-ksp_gmres_cgs_refinement_type","Type of iterative refinement for classical (unmodified) Gram-Schmidt","KSPGMRESSetCGSRefinementType()",types,3,types[lgmres->cgstype],&indx,&flg);CHKERRQ(ierr);    
    if (flg) {
      ierr = KSPGMRESSetCGSRefinementType(ksp,(KSPGMRESCGSRefinementType)indx);CHKERRQ(ierr);
    }

    ierr = PetscOptionsName("-ksp_gmres_krylov_monitor","Graphically plot the Krylov directions","KSPSetMonitor",&flg);CHKERRQ(ierr);
    if (flg) {
      PetscViewers viewers;
      ierr = PetscViewersCreate(ksp->comm,&viewers);CHKERRQ(ierr);
      ierr = KSPSetMonitor(ksp,KSPGMRESKrylovMonitor,viewers,(int (*)(void*))PetscViewersDestroy);CHKERRQ(ierr);
    }

/* LGMRES_MOD - specify number of augmented vectors and whether the space should be a constant size*/
     ierr = PetscOptionsName("-ksp_lgmres_constant","Use constant approx. space size","KSPGMRESSetConstant",&flg);CHKERRQ(ierr);
    /*if (flg) {ierr = KSPGMRESSetConstant(ksp);CHKERRQ(ierr);}*/ /*<--doesn't like this */ 
    if (flg) { lgmres->approx_constant = 1; }                     /* in favor of this line....*/  

    ierr = PetscOptionsInt("-ksp_lgmres_augment","Number of error approximations to augment the Krylov space with","KSPLGMRESSetAugDim",lgmres->aug_dim,&aug,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPLGMRESSetAugDim(ksp,aug);CHKERRQ(ierr); }



  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN int KSPComputeExtremeSingularValues_GMRES(KSP,PetscReal *,PetscReal *);
EXTERN int KSPComputeEigenvalues_GMRES(KSP,int,PetscReal *,PetscReal *,int *);

/*functions for extra lgmres options here*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPLGMRESSetConstant_LGMRES" 
int KSPLGMRESSetConstant_LGMRES(KSP ksp)
{
  KSP_LGMRES *lgmres = (KSP_LGMRES *)ksp->data;
  PetscFunctionBegin;
  lgmres->approx_constant = 1;
   
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPLGMRESSetAugDim_LGMRES" 
int KSPLGMRESSetAugDim_LGMRES(KSP ksp,int aug_dim)
{
  KSP_LGMRES *lgmres = (KSP_LGMRES *)ksp->data;

  PetscFunctionBegin;

  if (aug_dim < 0) SETERRQ(1,"Augmentation dimension must be positive");
  if (aug_dim > (lgmres->max_k -1))  SETERRQ(1,"Augmentation dimension must be <= (restart size-1)");

  lgmres->aug_dim = aug_dim;

  PetscFunctionReturn(0);
}
EXTERN_C_END


/* end new lgmres functions */


/* use these options from gmres */
EXTERN_C_BEGIN
EXTERN int KSPGMRESSetHapTol_GMRES(KSP,double);
EXTERN int KSPGMRESSetPreAllocateVectors_GMRES(KSP);
EXTERN int KSPGMRESSetRestart_GMRES(KSP,int);
EXTERN int KSPGMRESSetOrthogonalization_GMRES(KSP,int (*)(KSP,int));
EXTERN int KSPGMRESSetCGSRefinementType_GMRES(KSP,KSPGMRESCGSRefinementType);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_LGMRES"
int KSPCreate_LGMRES(KSP ksp)
{
  KSP_LGMRES *lgmres;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscNew(KSP_LGMRES,&lgmres);CHKERRQ(ierr);
  PetscMemzero(lgmres,sizeof(KSP_LGMRES));
  PetscLogObjectMemory(ksp,sizeof(KSP_LGMRES));
  ksp->data                              = (void*)lgmres;
  ksp->ops->buildsolution                = KSPBuildSolution_LGMRES;

  ksp->ops->setup                        = KSPSetUp_LGMRES;
  ksp->ops->solve                        = KSPSolve_LGMRES;
  ksp->ops->destroy                      = KSPDestroy_LGMRES;
  ksp->ops->view                         = KSPView_LGMRES;
  ksp->ops->setfromoptions               = KSPSetFromOptions_LGMRES;
  ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_GMRES;
  ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_GMRES;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",
                                    "KSPGMRESSetPreAllocateVectors_GMRES",
                                     KSPGMRESSetPreAllocateVectors_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetOrthogonalization_C",
                                    "KSPGMRESSetOrthogonalization_GMRES",
                                     KSPGMRESSetOrthogonalization_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetRestart_C",
                                    "KSPGMRESSetRestart_GMRES",
                                     KSPGMRESSetRestart_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetHapTol_C",
                                    "KSPGMRESSetHapTol_GMRES",
                                     KSPGMRESSetHapTol_GMRES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPGMRESSetCGSRefinementType_C",
                                    "KSPGMRESSetCGSRefinementType_GMRES",
                                     KSPGMRESSetCGSRefinementType_GMRES);CHKERRQ(ierr);

  /*LGMRES_MOD add extra functions here - like the one to set num of aug vectors */
  ierr =  PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPLGMRESSetConstant_C",
                                     "KSPLGMRESSetConstant_LGMRES",
                                      KSPLGMRESSetConstant_LGMRES);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPLGMRESSetAugDim_C",
                                    "KSPLGMRESSetAugDim_LGMRES",
                                     KSPLGMRESSetAugDim_LGMRES);CHKERRQ(ierr);
 

  /*defaults */
  lgmres->haptol              = 1.0e-30;
  lgmres->q_preallocate       = 0;
  lgmres->delta_allocate      = LGMRES_DELTA_DIRECTIONS;
  lgmres->orthog              = KSPGMRESClassicalGramSchmidtOrthogonalization;
  lgmres->nrs                 = 0;
  lgmres->sol_temp            = 0;
  lgmres->max_k               = LGMRES_DEFAULT_MAXK;
  lgmres->Rsvd                = 0;
  lgmres->cgstype             = KSP_GMRES_CGS_REFINEMENT_NONE;
  /*LGMRES_MOD - new defaults */
  lgmres->aug_dim             = LGMRES_DEFAULT_AUGDIM;
  lgmres->aug_ct              = 0; /* start with no aug vectors */ 
  lgmres->approx_constant     = 0;
  lgmres->matvecs             = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
