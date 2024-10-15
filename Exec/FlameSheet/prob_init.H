#ifndef PROB_INIT_H_
#define PROB_INIT_H_

struct ProbParm
{
    amrex::Real hpos    = 0.5;
    amrex::Real pertmag = 0.0;
};

// This header is included by NavierStokes.H.  These are members of NavierStokes
void prob_initData();

void init_flamesheet (amrex::Box const& vbx,
		      /* amrex::Array4<amrex::Real> const& press, */
		      amrex::Array4<amrex::Real> const& vel,
		      amrex::Array4<amrex::Real> const& scal,
		      const int nscal,
		      amrex::Box const& domain,
		      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
		      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
		      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& probhi,
		      ProbParm Prob);

//
// Problems parameters, to be read from inputs file
//
static int probtype;

#endif
