
#include <NavierStokes.H>
#include <AMReX_ParmParse.H>
#include <iamr_constants.H>

#ifdef AMREX_USE_TURBULENT_FORCING
#include <TurbulentForcing_def.H>
#endif

using namespace amrex;

int NavierStokes::probtype = -1;

// Initialize state and pressure with problem-specific data
void NavierStokes::prob_initData ()
{
    // Create struct to hold initial conditions parameters
    InitialConditions IC;

    // Read problem parameters from inputs file
    ParmParse pp("prob");
    pp.query("probtype",probtype);
    pp.query("density_ic",IC.density);
    pp.query("turb_scale",IC.turb_scale);

#ifdef AMREX_USE_GFLAME
    pp.query("h_position",IC.hpos);
    pp.query("h_pert",IC.pertmag);
    Print() << "GFlame overiding density to unburnt density \n";
    pp.query("unburnt_density",IC.density_u);
    IC.density = IC.density_u; 
    pp.query("burnt_density",IC.density_b);
#endif
    // Fill state and, optionally, pressure
    MultiFab& P_new = get_new_data(Press_Type);
    MultiFab& S_new = get_new_data(State_Type);
//    const int nscal = NUM_STATE-GField;
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

#ifdef AMREX_USE_TURBULENT_FORCING
    // Initialize data structures used for homogenous isentropic forced turbulence.
    if (level == 0)
        TurbulentForcing::init_turbulent_forcing(problo,probhi);
#endif

#ifdef _OPENMP
#pragma omp parallel  if (Gpu::notInLaunchRegion())
#endif

    // For initialising turbulence
    if ( probtype == 1 )
    {
#if (AMREX_USE_TURBULENT_FORCING == false)
	amrex::Abort("NavierStokes::prob_init: cannot start turbulence forcing not compiled");
#endif
	Print() << "initialising velocity feild" << std::endl;
        // Random combination of cosine waves to be used with forced turbulence,
        // where ICs are less important as the forcing takes over with time.
        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& vbx = mfi.tilebox();
            init_forced(vbx, /*P_new.array(mfi),*/ S_new.array(mfi, Xvel),
                        S_new.array(mfi, Density), nscal,
                        domain, dx, problo, probhi, IC);
        }
    }
#ifdef AMREX_USE_GFLAME
    else
    {
        // Random combination of cosine waves to be used with forced turbulence,
        // where ICs are less important as the forcing takes over with time.
        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& vbx = mfi.tilebox();
            init_gflame(vbx, /*P_new.array(mfi),*/ S_new.array(mfi, Xvel),
                        S_new.array(mfi, Density), nscal,
                        domain, dx, problo, probhi, IC);
        }
    }
#else
    else
    {
	amrex::Abort("NavierStokes::prob_init: unknown probtype");
    }
#endif
}

void NavierStokes::init_forced (Box const& vbx,
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
    //
    // Fill Velocity
    //
    AMREX_D_TERM(vel(i,j,k,0) =  IC.turb_scale * std::cos(TwoPi*y/Ly) * std::cos(TwoPi*z/Lz);,
		 vel(i,j,k,1) =  IC.turb_scale * std::cos(TwoPi*x/Lx) * std::cos(TwoPi*z/Lz);,
		 vel(i,j,k,2) =  IC.turb_scale * std::cos(TwoPi*x/Lx) * std::cos(TwoPi*y/Ly););

    //
    // Scalars, ordered as Density, Tracer(s)
    //
    scal(i,j,k,0) = IC.density;

    // Tracers
    for ( int nt=1; nt<nscal; nt++)
    {
      scal(i,j,k,nt) = 1.0;
    }
  });
}

#ifdef AMREX_USE_GFLAME
void NavierStokes::init_gflame (Box const& vbx,
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
#endif
	G_y1 += pert;
	G_y2 += pert;
    }
    Real dist = y - (G_y1+G_y2)/2;
    if (y > G_y1 && y < G_y2) {
	scal(i,j,k,iG) = 0;
	scal(i,j,k,iD) = IC.density_u;
    }
    else if (y < G_y1) {
	scal(i,j,k,iG) = dist/Lx;
	scal(i,j,k,iD) = IC.density_u;
    }
    else {
	scal(i,j,k,iG) = dist/Lx;
	scal(i,j,k,iD) = IC.density_b;
    }
    //    NavierStokesBase::gfieldinit();
  });
}
#endif
