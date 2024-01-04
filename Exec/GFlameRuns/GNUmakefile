TOP = ../..
AMREX_HOME       = $(TOP)/Submodules/amrex
AMREX_HYDRO_HOME = $(TOP)/Submodules/AMReX-Hydro

DIM        = 2
USE_MPI    = TRUE
USE_OMP    = FALSE
USE_CUDA   = FALSE

USE_TURBULENT_FORCING = FALSE
USE_FAST_FORCE        = FALSE
USE_VELOCITY          = TRUE
USE_LEVELSET          = TRUE

COMP	   = gcc
DEBUG	   = FALSE
PRECISION  = DOUBLE

USE_HYPRE         = FALSE
USE_METIS         = FALSE
USE_XBLAS         = FALSE
USE_SENSEI_INSITU = FALSE

ifeq (${USE_LEVELSET}, TRUE)
   EBASE     = iamr-levelset
else
   EBASE     = iamr
endif

ifeq (${USE_XBLAS}, TRUE)
   XTRADEFS   += -DXBLAS
   XTRAINCLOC += $(HOME)/tmp/xblas-1.0.248/src
   XTRALIBLOC += $(HOME)/tmp/xblas-1.0.248/src
   XTRALIBS   += -lxblas
endif

ifeq (${USE_VELOCITY}, TRUE)
  #AMRVIS_DIR defines the directory in which we will find pAmrvis (e.g. DataServices, AmrData and FABUTIL)
  AMRVIS_DIR = $(AMREX_HOME)/Src/Extern/amrdata
endif

Blocs   := .

include ./Make.package
include $(TOP)/Exec/Make.IAMR
