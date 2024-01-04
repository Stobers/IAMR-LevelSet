#ifndef PROB_INIT_H_
#define PROB_INIT_H_


// This header is included by NavierStokes.H.  These are members of NavierStokes

//
// struct to hold initial conditions parameters
//
struct InitialConditions
{
    //
    // For initializing with random combination of cosine waves, used with forced
    // turbulence, where ICs are less important as the forcing takes over with time.
    //
    amrex::Real turb_scale = 1.0;
    amrex::Real density = 1.0;
#ifdef AMREX_USE_GFLAME
    amrex::Real hpos = 0.5;
    amrex::Real pertmag = 0.0;
    amrex::Real density_u = 1.0;
    amrex::Real density_b = 2.0;

    
#endif
};

//
// Problem initialization functions
//
void prob_initData();

void init_forced (amrex::Box const& vbx,
                  /* amrex::Array4<amrex::Real> const& press, */
                  amrex::Array4<amrex::Real> const& vel,
                  amrex::Array4<amrex::Real> const& scal,
                  const int nscal,
                  amrex::Box const& domain,
                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
                  InitialConditions IC);
#ifdef AMREX_USE_GFLAME
void init_gflame (amrex::Box const& vbx,
                  /* amrex::Array4<amrex::Real> const& press, */
                  amrex::Array4<amrex::Real> const& vel,
                  amrex::Array4<amrex::Real> const& scal,
                  const int nscal,
                  amrex::Box const& domain,
                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
                  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
                  InitialConditions IC);
#endif

//
// Problems parameters, to be read from inputs file
//
static int probtype;

#endif
