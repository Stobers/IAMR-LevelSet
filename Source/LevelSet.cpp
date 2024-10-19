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
int  LevelSet::nSteps = 40;
int  LevelSet::nWidth = 12;
Real LevelSet::lF;
Real LevelSet::sF;
Real LevelSet::markstein = 0;
int  LevelSet::verbose = 0;

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

	    Print() << "verbose = " << verbose << std::endl;
	    Print() << "unburnt_density = " << unburnt_density << std::endl;
	    Print() << "burnt_density = " << burnt_density << std::endl;
	    Print() << "nSteps = " << nSteps << std::endl;
	    Print() << "nWidth = " << nWidth << std::endl;
	    Print() << "lF = " << lF << std::endl;
	    Print() << "sF = " << sF << std::endl;
	    Print() << "markstein = " << markstein << std::endl;
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
    if (LevelSet::verbose > 0) {
      Print() << "LevelSet: redistancing levelset \n";
    }

    Print() << "verbose = " << verbose << std::endl;
    Print() << "unburnt_density = " << unburnt_density << std::endl;
    Print() << "burnt_density = " << burnt_density << std::endl;
    Print() << "nSteps = " << nSteps << std::endl;
    Print() << "nWidth = " << nWidth << std::endl;
    Print() << "lF = " << lF << std::endl;
    Print() << "sF = " << sF << std::endl;
    Print() << "markstein = " << markstein << std::endl;

    // build multifabs
    const int nGrowGradG = 0;
    MultiFab gradGField(grids,dmap,AMREX_SPACEDIM+1,nGrowGradG,
			MFInfo(),navier_stokes->Factory());
    const int nGrowSField = 0;
    MultiFab sField  = MultiFab(grids,dmap,1,nGrowSField,
				MFInfo(),navier_stokes->Factory());

    // set sField
    for (MFIter mfi(gField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.tilebox();
	Array4<Real> const& g = gField.array(mfi,GField);
	Array4<Real> const& s = sField.array(mfi);
	const Real* dx = navier_stokes->geom.CellSize();
	setS(g,s,dx,bx);
    }

    // loop to |gradG| = 1
    for (int n=0; n<nSteps; n++) {
      if (LevelSet::verbose > 2) {
	Print() << "*** LevelSet ***: re-initialising levelset, step ="
		<< n << " / " << LevelSet::nSteps << std::endl;
      }

      NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));
      const int g_nGrow = 2;
      FillPatchIterator fpiG(ns_level,gField,g_nGrow,
			     navier_stokes->state[State_Type].prevTime(),
			     State_Type,GField,1);
      MultiFab& g_fpi = fpiG.get_mf();
    
      for (MFIter mfi(gField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
	  const Box& bx = mfi.tilebox();
	  Array4<Real> const& gfpi   = g_fpi.array(mfi);
	  Array4<Real> const& g      = gField.array(mfi,GField);
	  Array4<Real> const& s      = sField.array(mfi);
	  Array4<Real> const& grd    = gradGField.array(mfi);
	  const Real* dx = navier_stokes->geom.CellSize();
	  gradG(gfpi,s,grd,dx,bx);
	  updateG(g,s,grd,dx,bx);
      }
    }
}

//
// sets the density based off the GField
//
void
LevelSet::set_rhofromG(MultiFab& gField, MultiFab& density)
{
    if (LevelSet::verbose > 1) {
	Print() << "** LevelSet ** : setting rho from G\n";
    }

    for (MFIter mfi(density,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box&  bx       = mfi.tilebox();
	auto const& rho      = density.array(mfi,Density);
	auto const& g        = gField.array(mfi,GField);
	amrex::ParallelFor(bx, [rho, g] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
	    {
		rho(i,j,k) = unburnt_density +
		  (0.5 * (burnt_density - unburnt_density)) *
		  (1 + std::tanh(g(i,j,k)/(0.5*lF)));
	    });
    }
}

//
//
//

void
LevelSet::setS(Array4<Real> const& g,Array4<Real> const& s,const Real* dx, const Box& bx)
{
    Real eps = 2*dx[0];
    ParallelFor(bx, [g, s, dx, eps] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	{
	  s(i,j,k) = g(i,j,k) / std::sqrt(pow(g(i,j,k),2) + 1e-200);
	});
}

//
//
//

void
LevelSet::gradG(Array4<Real> const& g,
		Array4<Real> const& s,
		Array4<Real> const& grd,
		const Real* dx,
		const Box& bx)
{
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
	    Real modGradG2 = pow(grd(i,j,k,0),2) + pow(grd(i,j,k,1),2);
#if (AMREX_SPACEDIM == 3)
	    // upwind in z
	    if ((g(i,j,k+1) - g(i,j,k-1)) * s(i,j,k) > 0) {
		grd(i,j,k,2) = (g(i,j,k) - g(i,j,k-1)) / dx[2]; // backward difference
	    } else {
		grd(i,j,k,2) = (g(i,j,k+1) - g(i,j,k)) / dx[2]; // forward difference
	    }
	    modGradG2 += pow(grd(i,j,k,2),2);
#endif
	    grd(i,j,k,AMREX_SPACEDIM) = std::sqrt(modGradG2);
	});
}

void
LevelSet::updateG(Array4<Real> const& g,
		  Array4<Real> const& s,
		  Array4<Real> const& grd,
		  const Real* dx,
		  const Box& bx)
{
    const Real tau = 0.5 * dx[0];  // pseudo time step    
    ParallelFor(bx, [g, grd, s, tau, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	{
	    g(i,j,k) = g(i,j,k) - tau * s(i,j,k) * (grd(i,j,k,AMREX_SPACEDIM) - 1);
	    g(i,j,k) = max(-nWidth*dx[0],min(nWidth*dx[0],g(i,j,k)));
	});
}




void
LevelSet::flamespeed(Array4<Real> const& g,
		     Array4<Real> const& sloc,
		     const Real* dx,
		     const Box& bx)
{
    ParallelFor(bx, [g, sloc, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	{
	    // calculate curvautre
	    Real kap;
	    if (fabs(g(i,j,k)) < nWidth*dx[0]) {
		Real dg_dxdy = (((g(i,j,k) - g(i,j-1,k)) / dx[1]) - ((g(i-1,j,k) - g(i-1,j-1,k)) / dx[1])) / dx[0];
		Real dg_dxx = (g(i+1,j,k) - 2 * g(i,j,k) + g(i-1,j,k)) / pow(dx[0],2);
		Real dg_dyy = (g(i,j+1,k) - 2 * g(i,j,k) + g(i,j-1,k)) / pow(dx[1],2);
		Real dg_dx = (g(i,j,k) - g(i-1,j,k)) / dx[0];
		Real dg_dy = (g(i,j,k) - g(i,j-1,k)) / dx[1];
		
		kap = -(((dg_dxx * pow(dg_dy,2))
			      - (2 * dg_dy * dg_dx * dg_dxdy)
			      + (dg_dyy * pow(dg_dx,2)))
			/ pow(pow(dg_dx,2)+pow(dg_dy,2),3./2.));
	    }
	    else {
		kap = 0;
	    }

	    // override for now
	    kap = 0;

	    // model sloc
	    if (fabs(g(i,j,k)) < nWidth*dx[0]) {
		sloc(i,j,k) = max(1.e-5,min(4*sF,sF * (1 - markstein * kap * lF)));
	    }
	    else {
		sloc(i,j,k) = sF;
	    }	    
	});
}

//
//
//
void
LevelSet::divU(Array4<Real> const& g,
	       Array4<Real> const& div_u,
	       Array4<Real> const& rho,
	       Array4<Real> const& grd,
	       Array4<Real> const& sloc,
	       const Real* dx,
	       const Box& bx)
{
  amrex::ParallelFor(bx, [g,grd,rho,div_u,sloc]
  AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real dRhoInvDn = (unburnt_density - burnt_density) *
      pow(1./std::cosh(g(i,j,k)/(0.5*lF)),2) *
      grd(i,j,k,AMREX_SPACEDIM) / pow(rho(i,j,k),2) / lF;
    div_u(i,j,k) = unburnt_density * sloc(i,j,k) * (dRhoInvDn);
  });
}
