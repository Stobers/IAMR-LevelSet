#include <LevelSet.H>
#include <NavierStokesBase.H>
#include <iamr_constants.H>

#include <algorithm>
#include <cfloat>
#include <iomanip>
#include <array>
#include <iostream>

#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_MLMG.H>
#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLEBTensorOp.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_EB_utils.H>
#include <AMReX_EB_Redistribution.H>
#else
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLTensorOp.H>
#endif

using namespace amrex;

namespace
{
    bool initialized = false;
}

Real LevelSet::unburnt_density;
Real LevelSet::burnt_density;
int LevelSet::nSteps = 24;
int LevelSet::nWidth = 12;
Real LevelSet::lF;
Real LevelSet::sF;
Real LevelSet::markstein = 0;
int LevelSet::verbose = 0;


void
LevelSet::Finalize ()
{
    initialized = false;
}



LevelSet::LevelSet (Amr*               Parent,
		    NavierStokesBase*  Caller,
		    LevelSet*         Coarser)
    :
    parent(Parent),
    navier_stokes(Caller),
    grids(navier_stokes->boxArray()),
    dmap(navier_stokes->DistributionMap()),
    level(navier_stokes->Level()),
    coarser(Coarser),
    finer(nullptr)

{
    if (!initialized)
    {
	{
	    ParmParse pp("ls");
	    pp.query("v", verbose);
	    pp.get("unburnt_density", unburnt_density);
	    pp.get("burnt_density", burnt_density);
	    pp.query("nSteps", nSteps);
	    pp.query("nWidth", nWidth);
	    pp.get("lF", lF);
	    pp.get("sF", sF);
	    pp.query("markstein_number", markstein);	    
	}
	
        amrex::ExecOnFinalize(LevelSet::Finalize);
        initialized = true;
    }

    if (level > 0)
    {
        crse_ratio = parent->refRatio(level-1);
        coarser->finer = this;
    }
}




//
//
// ---- Public Functions
//
//

// reinitialises the GField
void
LevelSet::redistance(MultiFab& gField)
{
    if (LevelSet::verbose == 1) {
	Print() << " *** LS *** LevelSet redistancing levelset \n";
    }

    const int nGrowGradG = 0;
    MultiFab gradGField(grids,dmap,AMREX_SPACEDIM+1,nGrowGradG,MFInfo(),navier_stokes->Factory());
    const int nGrowSField = 0;
    MultiFab sField  = MultiFab(grids,dmap,1,nGrowSField,MFInfo(), navier_stokes->Factory());
    
    set_sfield(gField, sField);
    for (int n=0; n<nSteps; n++) {
      Print() << " *** LS *** " << n << std::endl;
	calc_gradG(gField, sField, gradGField);
	update_gField(gField, sField, gradGField);
    }
}


//
// sets the density based off the GField
//
void
LevelSet::set_rhofromG(MultiFab& gField, MultiFab& density)
{
    for (MFIter mfi(density,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box&  bx       = mfi.tilebox();
	auto const& rho      = density.array(mfi,Density);
	auto const& g        = gField.array(mfi,GField);
	amrex::ParallelFor(bx, [rho, g] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
	    {
		rho(i,j,k) = unburnt_density + (0.5 * (burnt_density - unburnt_density)) * (1 + std::tanh(g(i,j,k)/(0.5*lF)));
	    });
    }
}


//
// calculates the modified divU
//
void
LevelSet::calc_divU(MultiFab& div_u, MultiFab& density, MultiFab& gradG, MultiFab& flamespeed)
{
    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));

    const int nGrow = 1;
    
    FillPatchIterator fpi(ns_level,density,nGrow,
			  navier_stokes->state[State_Type].prevTime(),
			  State_Type,Density,1);
    MultiFab& rho_fpi = fpi.get_mf();

    for (MFIter mfi(gradG,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box&  bx       = mfi.tilebox();
	auto const& rho      = rho_fpi.array(mfi);
	auto const& grd      = gradG.array(mfi,MagGradg);
	auto const& divu     = div_u.array(mfi);
	auto const& sloc     = flamespeed.array(mfi);
	const auto dx        = navier_stokes->geom.CellSizeArray();
	amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
	    {
		Real nx = grd(i,j,k,1)/(grd(i,j,k,0)+1e-99);
		Real ny = grd(i,j,k,2)/(grd(i,j,k,0)+1e-99);
#if (AMREX_SPACEDIM == 3)
		Real nz = grd(i,j,k,3)/(grd(i,j,k,0)+1e-99);
#endif
		Real dndrho = (nx * ((1/rho(i+1,j,k)) - (1/rho(i-1,j,k))) / (2*dx[0]))
		            + (ny * ((1/rho(i,j+1,k)) - (1/rho(i,j-1,k))) / (2*dx[1]));
#if (AMREX_SPACEDIM == 3)
		dndrho     +=  nz * ((1/rho(i,j,k+1)) - (1/rho(i,j,k-1))) / (2*dx[2]);
#endif
		    
		divu(i,j,k) = unburnt_density * sloc(i,j,k) * dndrho;
	    });
    }
}

//
// measures the gradient of the GField (only used for divU)
//
void
LevelSet::get_gradG(MultiFab& gField, MultiFab& gradGField)
{
    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));

    const int g_nGrow = 1;
    FillPatchIterator fpiG(ns_level,gField,g_nGrow,
			   navier_stokes->state[State_Type].prevTime(),
			   State_Type,GField,1);
    MultiFab& gfpi = fpiG.get_mf();


    const int nGrowSField = 0;
    MultiFab sField  = MultiFab(grids,dmap,1,nGrowSField,MFInfo(), navier_stokes->Factory());
    
    set_sfield(gField, sField);
    calc_gradG(gField, sField, gradGField);

}

//
// measures the levelset curvature
//
void
LevelSet::calc_curvature(MultiFab& gField, MultiFab& kappa)
{
    // this does not yet do 3D!
    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));

    const int nGrow = 1;
    FillPatchIterator fpi(ns_level,gField,nGrow,
			  navier_stokes->state[State_Type].prevTime(),
			  State_Type,GField,1);
    MultiFab& gfpi = fpi.get_mf();
    
    for (MFIter mfi(kappa,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box&  bx     = mfi.tilebox();
	auto const& g      = gfpi.array(mfi);
	auto const& kap    = kappa.array(mfi);
	auto const& dx     = navier_stokes->geom.CellSizeArray();
	amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
	    {
		if (fabs(g(i,j,k)) < nWidth*dx[0]) {
		    Real dg_dxdy = (((g(i,j,k) - g(i,j-1,k)) / dx[1]) - ((g(i-1,j,k) - g(i-1,j-1,k)) / dx[1])) / dx[0];
		    Real dg_dxx = (g(i+1,j,k) - 2 * g(i,j,k) + g(i-1,j,k)) / pow(dx[0],2);
		    Real dg_dyy = (g(i,j+1,k) - 2 * g(i,j,k) + g(i,j-1,k)) / pow(dx[1],2);
		    Real dg_dx = (g(i,j,k) - g(i-1,j,k)) / dx[0];
		    Real dg_dy = (g(i,j,k) - g(i,j-1,k)) / dx[1];
		    
		    kap(i,j,k) = ((dg_dxx * pow(dg_dy,2))
				  - (2 * dg_dy * dg_dx * dg_dxdy)
				  + (dg_dyy * pow(dg_dx,2)))
			/ pow(pow(dg_dx,2)+pow(dg_dy,2),3./2.);
		    kap(i,j,k) = -kap(i,j,k);
		}
		else {
		    kap(i,j,k) = 0;
		}
	    });
    }
}

//
// models the levelset speed
//
void
LevelSet::calc_flamespeed(MultiFab& gField, MultiFab& flamespeed)
{
    MultiFab kappa = MultiFab(grids,dmap,1,1,MFInfo(), navier_stokes->Factory());
    calc_curvature(gField, kappa);

    for (MFIter mfi(kappa,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box&  bx     = mfi.tilebox();
	auto const& kap    = kappa.array(mfi);
	auto const& sloc   = flamespeed.array(mfi);
	auto const& g      = gField.array(mfi,GField);
	auto const& dx     = navier_stokes->geom.CellSizeArray();
	amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
	    {
		if (fabs(g(i,j,k)) < nWidth*dx[0]) {
		  sloc(i,j,k) = max(1.e-5,min(4*sF,sF * (1 - markstein * kap(i,j,k) * lF)));
		}
		else {
		  sloc(i,j,k) = sF;
		}
	    });
    }    
}




//
//
// ---- Private Functions
//
//


void
LevelSet::set_sfield(MultiFab& gField, MultiFab& sField)
{
    for (MFIter mfi(gField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.tilebox();
	Array4<Real> const& g = gField.array(mfi,GField);
	Array4<Real> const& s = sField.array(mfi);
	const Real* dx = navier_stokes->geom.CellSize();
	ParallelFor(bx, [g, s, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	    {
		s(i,j,k) = g(i,j,k) / std::sqrt(pow(g(i,j,k),2) + pow(dx[0],2));
	    });
    }
}


void
LevelSet::calc_gradG(MultiFab& gField, MultiFab& sField, MultiFab& gradGField)
{
    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));
    const int g_nGrow = 1;
    FillPatchIterator fpiG(ns_level,gField,g_nGrow,
			   navier_stokes->state[State_Type].prevTime(),
			   State_Type,GField,1);
    MultiFab& gfpi = fpiG.get_mf();
    
    for (MFIter mfi(gradGField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.tilebox();
	Array4<Real> const& g   = gfpi.array(mfi);
	Array4<Real> const& s   = sField.array(mfi);
	Array4<Real> const& grd = gradGField.array(mfi);
	const Real* dx = navier_stokes->geom.CellSize();
	ParallelFor(bx, [g, s, grd, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	    {
		// upwind in x
		if ((g(i+1,j,k) - g(i-1,j,k)) * s(i,j,k) > 0) {
		    grd(i,j,k,0) = (g(i,j,k) - g(i-1,j,k)) / dx[0]; // backward difference
		} else {
		    grd(i,j,k,0) = (g(i+1,j,k) - g(i,j,k)) / dx[0]; // forward difference
		}
		// upwind in y
		if ((g(i,j+1,k) - g(i,j-1,k)) * s(i,j,k) > 0) {
		    grd(i,j,k,1) = (g(i,j,k) - g(i,j-1,k)) / dx[1]; // backward difference
		} else {
		    grd(i,j,k,1) = (g(i,j+1,k) - g(i,j,k)) / dx[1]; // forward difference
		}
#if AMREX_SPACEDIM==3
		// upwind in z
		if ((g(i,j,k+1) - g(i,j,k-1)) * s(i,j,k) > 0) {
		    grd(i,j,k,2) = (g(i,j,k) - g(i,j,k-1)) / dx[2]; // backward difference
		} else {
		    grd(i,j,k,2) = (g(i,j,k+1) - g(i,j,k)) / dx[2]; // forward difference
		}
#endif
		Real modGradG2 = pow(grd(i,j,k,0),2) + pow(grd(i,j,k,1),2);
#if AMREX_SPACEDIM==3
		modGradG2 += pow(grd(i,j,k,2),2);
#endif
		grd(i,j,k,AMREX_SPACEDIM) = std::sqrt(modGradG2);
	    });
    }
}


void
LevelSet::update_gField(MultiFab& gField, MultiFab& sField, MultiFab& gradGField)
{
    const Real* dx = navier_stokes->geom.CellSize();
    const Real tau = 0.5 * dx[0];  // pseudo time step
   
    // updates gfield using forward time
    for (MFIter mfi(gField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.tilebox();
	Array4<Real> const& g   = gField.array(mfi,GField);
	Array4<Real> const& grd = gradGField.array(mfi);
	Array4<Real> const& s   = sField.array(mfi);
	ParallelFor(bx, [g, grd, s, tau, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	    {
	      g(i,j,k) = g(i,j,k) - tau * s(i,j,k) * (grd(i,j,k,AMREX_SPACEDIM) - 1);
	      g(i,j,k) = max(-nWidth*dx[0],min(nWidth*dx[0],g(i,j,k)));
	    });
    }
}
