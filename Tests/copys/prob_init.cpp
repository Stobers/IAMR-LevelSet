
#include <NavierStokes.H>
#include <AMReX_ParmParse.H>
#include <iamr_constants.H>


using namespace amrex;

int NavierStokes::probtype = -1;

// Initialize state and pressure with problem-specific data
void NavierStokes::prob_initData ()
{
    // Create struct to hold initial conditions parameters
    InitialConditions IC;

    ParmParse pp("ls");
    pp.query("h_position",IC.hpos);
    pp.query("h_pert",IC.pertmag);
    pp.query("unburnt_density",IC.density_u);
    IC.density = IC.density_u; 
    pp.query("burnt_density",IC.density_b);

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
	init_levelset(vbx, /*P_new.array(mfi),*/ S_new.array(mfi, Xvel),
		      S_new.array(mfi, Density), nscal,
		      domain, dx, problo, probhi, IC);
        }
}


void NavierStokes::init_levelset (Box const& vbx,
                /* Array4<Real> const& press, */
                Array4<Real> const& vel,
                Array4<Real> const& scal,
                const int nscal,
                Box const& domain,
                GpuArray<Real, AMREX_SPACEDIM> const& dx,
                GpuArray<Real, AMREX_SPACEDIM> const& problo,
                GpuArray<Real, AMREX_SPACEDIM> const& probhi,
                InitialConditions IC)
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
    Real G_y1 = (Ly * IC.hpos) - 2 * (0.5 * dx[1]);
    Real G_y2 = (Ly * IC.hpos) + 2 * (0.5 * dx[1]);
    if (IC.pertmag > 0) {
#if (AMREX_SPACEDIM == 2)
	Real pert = IC.pertmag * (1.000 * std::sin(2*Pi*4*x/Lx)
				  + 1.023 * std::sin(2*Pi*2*(x-.004598)/Lx)
				  + 0.945 * std::sin(2*Pi*3*(x-.00712435)/Lx)
				  + 1.017 * std::sin(2*Pi*5*(x-.0033)/Lx)
				  + .982 * std::sin(2*Pi*5*(x-.014234)/Lx));
#else
	Real pert = IC.pertmag * (1.000 * std::sin(2*Pi*4*x/Lx) * std::sin(2*Pi*5*z/Lz)
				  + 1.023 * std::sin(2*Pi*2*(x-.004598)/Lx) * std::sin(2*Pi*4*(z-.0053765)/Lz)
				  + 0.945 * std::sin(2*Pi*3*(x-.00712435)/Lx) * std::sin(2*Pi*3*(z-.02137)/Lz)
				  + 1.017 * std::sin(2*Pi*5*(x-.0033)/Lx) * std::sin(2*Pi*6*(z-.018)/Lz)
				  + .982 * std::sin(2*Pi*5*(x-.014234)/Lx));

//	Real pert = IC.pertmag * (1.000 * std::sin(2*Pi*4*x/Lx)
//				  + 1.023 * std::sin(2*Pi*2*(x-.004598)/Lx)
//				  + 0.945 * std::sin(2*Pi*3*(x-.00712435)/Lx)
//				  + 1.017 * std::sin(2*Pi*5*(x-.0033)/Lx)
//				  + .982 * std::sin(2*Pi*5*(x-.014234)/Lx));

#endif
	G_y1 += pert;
	G_y2 += pert;
    }

    Real dist = 0;
    dist = pow(y - (G_y1+G_y2)/2,1);
    if (y <= G_y1) {
	scal(i,j,k,iG) = dist;
	scal(i,j,k,iD) = IC.density_u;	
    }
    else {
	scal(i,j,k,iG) = dist;
	scal(i,j,k,iD) = IC.density_b;
//	vel(i,j,k,1) = IC.density_u / (IC.density_u - IC.density_b);
//	vel(i,j,k,1) = 5;
    }
  });
}
