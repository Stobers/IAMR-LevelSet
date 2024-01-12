AMREX_HOME = ../../Submodules/amrex
DEBUG	      = FALSE
COMP          = gnu
PRECISION     = DOUBLE
USE_MPI       = FALSE
USE_OMP       = FALSE
EBASE         = probe
NEEDS_f90_SRC = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
CEXE_sources += $(EBASE).cpp
INCLUDE_LOCATIONS += .
VPATH_LOCATIONS   += .
Pdirs   := Base Boundary AmrCore Extern/amrdata LinearSolvers/MLMG
Ppack   += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
include $(Ppack)
INCLUDE_LOCATIONS += $(Blocs)
VPATH_LOCATIONS   += $(Blocs)
INCLUDE_LOCATIONS += $(AMREX_HOME)/Tools/C_util
VPATH_LOCATIONS += $(AMREX_HOME)/Tools/C_util
CEXE_sources += AMReX_Extrapolater.cpp
CEXE_headers += AMReX_Extrapolater.H
INCLUDE_LOCATIONS += $(AMREX_HOME)/Src/Amr
VPATH_LOCATIONS += $(AMREX_HOME)/Src/Amr
vpath %.c   : $(VPATH_LOCATIONS)
vpath %.h   : $(VPATH_LOCATIONS)
vpath %.cpp : $(VPATH_LOCATIONS)
vpath %.H   : $(VPATH_LOCATIONS)
vpath %.F   : $(VPATH_LOCATIONS)
vpath %.f   : $(VPATH_LOCATIONS)
vpath %.f90 : $(VPATH_LOCATIONS)
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
