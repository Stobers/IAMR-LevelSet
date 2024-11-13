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

// some useful macros
#define SIGN(x)     ( (x > 0) ? 1 : ((x < 0) ? -1 : 0) )
#define MINABS(a,b) ( (fabs(a)<fabs(b)) ? (a) : (b) )
#define MINMOD(a,b) ( (((a)*(b))<=0) ? (0.) : (MINABS((a),(b))) )
// use second order stencil
#define DOSO

using namespace amrex;

namespace
{
    bool initialized = false;
}

Real LevelSet::unburnt_density;
Real LevelSet::burnt_density;
int  LevelSet::initSteps = 128;
int  LevelSet::nSteps = 24;
int  LevelSet::nWidth = 12;
Real LevelSet::lF;
Real LevelSet::sF;
Real LevelSet::markstein = 0;
int  LevelSet::verbose = 0;

// AJA: there must be a better way; we should be able to get these from NS
int GField = 4;

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
	    pp.query("initSteps", initSteps);
	    pp.query("nSteps", nSteps);
	    pp.query("nWidth", nWidth);
	    pp.get("lF", lF);
	    pp.get("sF", sF);
	    pp.query("markstein_number", markstein);

	    Print() << "verbose = " << verbose << std::endl;
	    Print() << "unburnt_density = " << unburnt_density << std::endl;
	    Print() << "burnt_density = " << burnt_density << std::endl;
	    Print() << "initSteps = " << initSteps << std::endl;
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

//
// "Hamiltonian" (see e.g. Chene)
// (without the -1)
//
Real calcHG(Real a, Real b, Real c, Real d, Real s)
{
  Real HG;
  if (s<=0) {
    Real ap = fmax(a,0.);
    Real bm = fmin(b,0.);
    Real cp = fmax(c,0.);
    Real dm = fmin(d,0.);
    HG = sqrt( fmax(ap*ap,bm*bm) + fmax(cp*cp,dm*dm) );
  } else {
    Real am = fmin(a,0.);
    Real bp = fmax(b,0.);
    Real cm = fmin(c,0.);
    Real dp = fmax(d,0.);
    HG = sqrt( fmax(am*am,bp*bp) + fmax(cm*cm,dp*dp) );
  }
  return HG;
}

// reinitialises the GField
void
LevelSet::redistance(MultiFab& gField, int a_nSteps)
{
    if (LevelSet::verbose > 0) {
      Print() << "LevelSet: redistancing levelset \n";
    }

    Print() << "verbose = " << verbose << std::endl;
    Print() << "unburnt_density = " << unburnt_density << std::endl;
    Print() << "burnt_density = " << burnt_density << std::endl;
    Print() << "initSteps = " << initSteps << std::endl;
    Print() << "nSteps = " << nSteps << std::endl;
    Print() << "a_nSteps = " << a_nSteps << std::endl;
    Print() << "nWidth = " << nWidth << std::endl;
    Print() << "lF = " << lF << std::endl;
    Print() << "sF = " << sF << std::endl;
    Print() << "markstein = " << markstein << std::endl;

    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));
    
    // build multifab for gradient of G
    const int gradG_nGrow = 0;
    const int gradG_nComp = AMREX_SPACEDIM+2; // gradient, magnitude, and RS term
    MultiFab gradGField(grids,dmap,gradG_nComp,gradG_nGrow,
			MFInfo(),navier_stokes->Factory());

    // fill grown sField with an FPI
    const int sField_nGrow = 2;
    const int sField_nComp = 1;
    FillPatchIterator sFieldFPI(ns_level,gField,sField_nGrow,
				navier_stokes->state[State_Type].prevTime(),
				State_Type,GField,sField_nComp);
    MultiFab& sField = sFieldFPI.get_mf();

    //
    // loop to |gradG| = 1
    //
    for (int n=0; n<a_nSteps; n++) {
      
      if (LevelSet::verbose > 2) {
	Print() << "*** LevelSet ***: re-initialising levelset, step ="
		<< n << " / " << a_nSteps << std::endl;
      }

      // get grown G by fpi
      const int gGrown_nGrow = 2;
      const int gGrown_nComp = 1;
      FillPatchIterator gGrownFPI(ns_level,gField,gGrown_nGrow,
				  navier_stokes->state[State_Type].prevTime(),
				  State_Type,GField,gGrown_nComp);
      MultiFab& gGrownField = gGrownFPI.get_mf();

      // evaluate the first-order gradient, russo term, and update G
      for (MFIter mfi(gField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
	const Box&          bx     = mfi.tilebox();
	Array4<Real> const& gGrown = gGrownField.array(mfi);
	Array4<Real> const& g      = gField.array(mfi,GField);
	Array4<Real> const& s      = sField.array(mfi);
	Array4<Real> const& grd    = gradGField.array(mfi);
	const Real*         dx     = navier_stokes->geom.CellSize();

	ParallelFor(bx, [gGrown, g, s, grd, dx]
	AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	{
	  // dx min for time adaptive stepping
	  Real dxmin=fmin(dx[0],dx[1]);
	 
	  // signS
	  Real signS=0.;
	  if (s(i,j,k)>0) signS=1.; else if (s(i,j,k)<0) signS=-1.;

	  // calculate one-sided differences
#ifdef DOSO
	  Real Dxxp = (gGrown(i+2,j,k)-2.*gGrown(i+1,j,k)+gGrown(i  ,j,k))/(dx[0]*dx[0]);
	  Real Dxx0 = (gGrown(i+1,j,k)-2.*gGrown(i  ,j,k)+gGrown(i-1,j,k))/(dx[0]*dx[0]);
	  Real Dxxm = (gGrown(i  ,j,k)-2.*gGrown(i-1,j,k)+gGrown(i-2,j,k))/(dx[0]*dx[0]);
	  
	  Real Dyyp = (gGrown(i,j+2,k)-2.*gGrown(i,j+1,k)+gGrown(i,j  ,k))/(dx[1]*dx[1]);
	  Real Dyy0 = (gGrown(i,j+1,k)-2.*gGrown(i,j  ,k)+gGrown(i,j-1,k))/(dx[1]*dx[1]);
	  Real Dyym = (gGrown(i,j  ,k)-2.*gGrown(i,j-1,k)+gGrown(i,j-2,k))/(dx[1]*dx[1]);
	  
	  Real Dxp = (gGrown(i+1,j,k) - gGrown(i,j,k))/dx[0] - 0.5*dx[0]*MINMOD(Dxx0,Dxxp);
	  Real Dxm = (gGrown(i,j,k) - gGrown(i-1,j,k))/dx[0] + 0.5*dx[0]*MINMOD(Dxx0,Dxxm);
	  Real Dyp = (gGrown(i,j+1,k) - gGrown(i,j,k))/dx[1] - 0.5*dx[1]*MINMOD(Dyy0,Dyyp);
	  Real Dym = (gGrown(i,j,k) - gGrown(i,j-1,k))/dx[1] + 0.5*dx[1]*MINMOD(Dyy0,Dyym);
#else
	  Real Dxp = (gGrown(i+1,j,k) - gGrown(i,j,k))/dx[0]; // forward difference
	  Real Dxm = (gGrown(i,j,k) - gGrown(i-1,j,k))/dx[0]; // backward difference
	  Real Dyp = (gGrown(i,j+1,k) - gGrown(i,j,k))/dx[1]; // forward difference
	  Real Dym = (gGrown(i,j,k) - gGrown(i,j-1,k))/dx[1]; // backward difference
#endif
	  // near-interface corrections
	  if (s(i,j,k)*s(i+1,j,k)<0) { // correct Dxp
	    Real Sm   = s(i-1,j,k);
	    Real S0   = s(i,j,k);
	    Real Sp   = s(i+1,j,k);
	    Real Sp2  = s(i+2,j,k);
	    Real Sxx0 = MINMOD( Sm-2.*S0+Sp, S0-2.*Sp+Sp2);
	    Real D    = (0.5*Sxx0 - S0 - Sp);
	    D         = D*D - 4.*S0*Sp;
	    Real dxp  = fabs(Sxx0>1.e-10)
	      ? dx[0] * ( 0.5 + ( S0-Sp-SIGN(S0-Sp)*sqrt(D) ) / Sxx0 )
	      : dx[0] * ( S0 / (S0-Sp) );
#ifdef DOSO
	    Dxp = (0.-gGrown(i,j,k))/dxp - 0.5*dxp*MINMOD(Dxx0,Dxxp);
#else
	    Dxp = (0.-gGrown(i,j,k))/dxp;
#endif
	    dxmin = fmin(dxmin,dxp);
	  }
	  if (s(i,j,k)*s(i-1,j,k)<0) { // correct Dxm
	    Real Sm2  = s(i-2,j,k);
	    Real Sm   = s(i-1,j,k);
	    Real S0   = s(i,j,k);
	    Real Sp   = s(i+1,j,k);
	    Real Sxx0 = MINMOD( Sm-2.*S0+Sp, S0-2.*Sm+Sm2);
	    Real D    = (0.5*Sxx0 - S0 - Sm);
	    D         = D*D - 4.*S0*Sm;
	    Real dxm  = fabs(Sxx0>1.e-10)
	      ? dx[0] * ( 0.5 + ( S0-Sm-SIGN(S0-Sm)*sqrt(D) ) / Sxx0 )
	      : dx[0] * ( S0 / (S0-Sm) );
#ifdef DOSO
	    Dxm = (gGrown(i,j,k)-0)/dxm + 0.5*dxm*MINMOD(Dxx0,Dxxm);
#else
	    Dxm = (gGrown(i,j,k)-0)/dxm;
#endif
	    dxmin = fmin(dxmin,dxm);
	  }
	  if (s(i,j,k)*s(i,j+1,k)<0) { // correct Dyp
	    Real Sm   = s(i,j-1,k);
	    Real S0   = s(i,j,k);
	    Real Sp   = s(i,j+1,k);
	    Real Sp2  = s(i,j+2,k);
	    Real Syy0 = MINMOD( Sm-2.*S0+Sp, S0-2.*Sp+Sp2);
	    Real D    = (0.5*Syy0 - S0 - Sp);
	    D         = D*D - 4.*S0*Sp;
	    Real dyp  = fabs(Syy0>1.e-10)
	      ? dx[1] * ( 0.5 + ( S0-Sp-SIGN(S0-Sp)*sqrt(D) ) / Syy0 )
	      : dx[1] * ( S0 / (S0-Sp) );
#ifdef DOSO
	    Dyp = (0.-gGrown(i,j,k))/dyp - 0.5*dyp*MINMOD(Dyy0,Dyyp);
#else
	    Dyp = (0.-gGrown(i,j,k))/dyp;
#endif
	    dxmin = fmin(dxmin,dyp);
	  }
	  if (s(i,j,k)*s(i,j-1,k)<0) { // correct Dym
	    Real Sm2  = s(i,j-2,k);
	    Real Sm   = s(i,j-1,k);
	    Real S0   = s(i,j,k);
	    Real Sp   = s(i,j+1,k);
	    Real Syy0 = MINMOD( Sm-2.*S0+Sp, S0-2.*Sm+Sm2);
	    Real D    = (0.5*Syy0 - S0 - Sm);
	    D         = D*D - 4.*S0*Sm;
	    Real dym  = fabs(Syy0>1.e-10)
	      ? dx[1] * ( 0.5 + ( S0-Sm-SIGN(S0-Sm)*sqrt(D) ) / Syy0 )
	      : dx[1] * ( S0 / (S0-Sm) );
#ifdef DOSO
	    Dym = (gGrown(i,j,k)-0)/dym + 0.5*dym*MINMOD(Dyy0,Dyym);
#else
	    Dym = (gGrown(i,j,k)-0)/dym;
#endif
	    dxmin = fmin(dxmin,dym);
	  }
	  
	  // evaluate hamiltonian
	  Real HG = calcHG(Dxp,Dxm,Dyp,Dym,signS);

	  // pseudo time-step
	  const Real tau = 0.45 * dxmin;
	  
	  // combine and update
	  g(i,j,k) = g(i,j,k) - tau * signS * (HG - 1.);
	  
	  // limit to nWidth cells
	  g(i,j,k) = max(-nWidth*dx[0],min(nWidth*dx[0],g(i,j,k)));
	});
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
LevelSet::flamespeed(Array4<Real> const& g,
		     Array4<Real> const& sloc,
		     const Real* dx,
		     const Box& bx)
{
  // let's cap the curvature
  // is this really necessary?
  const Real kapMax=1./(3.*lF);
  ParallelFor(bx, [g, sloc, dx, kapMax] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    // default to sF
    sloc(i,j,k) = sF;
    // calculate curvature (default to flat)
    Real kap(0.);
    // do some on-the-fly averaging
    int nAvg = 1;
    Real kapDiv = 1./(Real)((2*nAvg+1)*(2*nAvg+1));
    // only bother near the surface
    if (fabs(g(i,j,k)) < (nWidth-3)*dx[0]) {
      for (int ii=i-nAvg; ii<=i+nAvg; ii++) {
	for (int jj=j-nAvg; jj<=j+nAvg; jj++) {
	  // use 9-point laplacian (and assume modGradG=1)
	  kap -= (    g(ii-1,jj+1,k) +  2.*g(ii,jj+1,k) +    g(ii+1,jj+1,k) +
		   2.*g(ii-1,jj  ,k) - 12.*g(ii,jj  ,k) + 2.*g(ii+1,jj  ,k) +
		      g(ii-1,jj-1,k) +  2.*g(ii,jj-1,k) +    g(ii+1,jj-1,k) );
	}
      }
      kap *= kapDiv;
      // keep curvature under control
      kap  = max(-kapMax,min(kapMax,kap));  // apply min/max
      // flame speed model
      sloc(i,j,k) = sF * max(5e-1, (1. - markstein * kap * lF));
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
	       Array4<Real> const& sloc,
	       const Real*         dx,
	       const Box&          bx)
{
  Print() << " *** entered LevelSet::divU " << std::endl;
  amrex::ParallelFor(bx, [g,rho,div_u,sloc]
  AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    // calculate d/dn(1/rho)
    // (assumes modGradG=1)
    Real tanhG     = std::tanh(g(i,j,k)/(0.5*lF));
    Real sech2     = 1.-tanhG*tanhG;
    Real rho2      = rho(i,j,k)*rho(i,j,k);
    Real dRhoInvDn = (unburnt_density-burnt_density) * sech2 / rho2 / lF; 
    // evaluate divU
    div_u(i,j,k) = unburnt_density * sloc(i,j,k) * (dRhoInvDn);
  });
}
