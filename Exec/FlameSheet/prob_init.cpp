
#include <NavierStokes.H>
#include <AMReX_ParmParse.H>
#include <iamr_constants.H>
#include <LevelSet.H>

using namespace amrex;

int NavierStokes::probtype = -1;

// Initialize state and pressure with problem-specific data
void NavierStokes::prob_initData ()
{
    // Create struct to hold initial conditions parameters
    ProbParm Prob;

    // Read problem parameters from inputs file
    {
	ParmParse pp("prob");
	pp.query("h_position",Prob.hpos);
	pp.query("h_pert",Prob.pertmag);
    }
    {
	ParmParse pp("ls");
	pp.query("unburnt_density",LevelSet::unburnt_density);
	pp.query("burnt_density",LevelSet::burnt_density);
	pp.query("nWidth",LevelSet::nWidth);
	pp.query("lF",LevelSet::lF);	
    }
    LevelSet::nSteps = 4;
    
    // Fill state and, optionally, pressure
    MultiFab& P_new = get_new_data(Press_Type);
    MultiFab& S_new = get_new_data(State_Type);
    const int nscal = NUM_STATE-Density;
    
    S_new.setVal(0.0);
    P_new.setVal(0.0);

    // Integer indices of the lower left and upper right corners of the
    // valid region of the entire domain.
    Box const&  domain = geom.Domain();
    auto const&     dx = geom.CellSizeArray();
    // Physical coordinates of the lower left corner of the domain
    auto const& problo = geom.ProbLoArray();
    auto const& probhi = geom.ProbHiArray();


#ifdef _OPENMP
#pragma omp parallel  if (Gpu::notInLaunchRegion())
#endif

    for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& vbx = mfi.tilebox();
	init_flamesheet(vbx, /*P_new.array(mfi),*/ S_new.array(mfi, Xvel),
		      S_new.array(mfi, Density), nscal,
		      domain, dx, problo, probhi, Prob);
    }
}

void NavierStokes::init_flamesheet (Box const& vbx,
                /* Array4<Real> const& press, */
                Array4<Real> const& vel,
                Array4<Real> const& scal,
                const int nscal,
                Box const& domain,
                GpuArray<Real, AMREX_SPACEDIM> const& dx,
                GpuArray<Real, AMREX_SPACEDIM> const& problo,
                GpuArray<Real, AMREX_SPACEDIM> const& probhi,
                ProbParm Prob)
{
  const auto domlo = amrex::lbound(domain);
  
  amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real x = problo[0] + (i - domlo.x + 0.5)*dx[0];
    Real y = problo[1] + (j - domlo.y + 0.5)*dx[1];
#if (AMREX_SPACEDIM == 3)
    Real z = problo[2] + (k - domlo.z + 0.5)*dx[2];
#else
    constexpr Real z = 0.0;
#endif

    const Real Lx    = (probhi[0] - problo[0]);
    const Real Ly    = (probhi[1] - problo[1]);
#if (AMREX_SPACEDIM == 3)
    const Real Lz    = (probhi[2] - problo[1]);
#else
    const Real Lz    = 1.0;
#endif

    // index of density and GField
    const int iD = 0;
    const int iG = 1;

    // set inital feild for density and GField
    Real pert = 8.*dx[1]*sin(4.*M_PI*x/Lx);
    Real dist=(y-Ly*Prob.hpos) - pert;
    scal(i,j,k,iG) = max(-LevelSet::nWidth*dx[1],min(LevelSet::nWidth*dx[1],dist));
    
    // set the density using the same tanh function as elsewhere
    scal(i,j,k,iD) = unburnt_density + (0.5 * (burnt_density - unburnt_density))
	* (1 + std::tanh(dist/(0.5*LevelSet::lF)));
  });
}
