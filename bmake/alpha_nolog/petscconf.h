#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.6 1998/04/25 23:25:59 balay Exp balay $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_alpha

#define HAVE_PWD_H 
#define HAVE_STRING_H 
#define HAVE_MALLOC_H 
#define HAVE_STDLIB_H 
#define HAVE_SWAPPED_BYTES 
#define HAVE_X11 
#define HAVE_DRAND48  
#define HAVE_GETDOMAINNAME  
#define HAVE_UNISTD_H 
#define HAVE_SYS_TIME_H 
#define HAVE_UNAME  

#define HAVE_64BITS
#define HAVE_FORTRAN_UNDERSCORE

#define HAVE_READLINK
#define HAVE_MEMMOVE
#define NEED_UTYPE_TYPEDEFS
#define USE_DBX_DEBUGGER

#endif
