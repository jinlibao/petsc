/*
   Private data structure used by the MLBCGS method. This data structure
  must be identical to the beginning of the KSP_FMLBCGS data structure
  so if you CHANGE anything here you must also change it there.
*/
#if !defined(__MLBCGS)
#define __MLBCGS

#include <petsc/private/kspimpl.h>        /*I "petscksp.h" I*/

typedef struct {
  PetscInt n;
  Vec guess;   /* if using right preconditioning with nonzero initial guess must keep that around to "fix" solution */
} KSP_MLBCGS;

PETSC_INTERN PetscErrorCode randn(PetscScalar*,PetscInt,PetscRandom);
PETSC_INTERN PetscErrorCode setQ(Vec*,Vec,PetscInt);
PETSC_INTERN PetscErrorCode KSPSetFromOptions_MLBCGS(PetscOptionItems *PetscOptionsObject,KSP);
PETSC_INTERN PetscErrorCode KSPSetUp_MLBCGS(KSP);
PETSC_INTERN PetscErrorCode KSPSolve_MLBCGS(KSP);
PETSC_INTERN PetscErrorCode KSPReset_MLBCGS(KSP);
PETSC_INTERN PetscErrorCode KSPDestroy_MLBCGS(KSP);

#endif
