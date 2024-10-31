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
int  LevelSet::nSteps = 24;
int  LevelSet::nFOSteps = 5;
int  LevelSet::nWidth = 12;
Real LevelSet::lF;
Real LevelSet::sF;
Real LevelSet::markstein = 0;
int  LevelSet::verbose = 0;

// AJA: there must be a better way; we should be able to get these from NS
int GField = 4;
int SmoothGField = 5;

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
	    pp.query("nFOSteps", nFOSteps);
	    pp.query("nWidth", nWidth);
	    pp.get("lF", lF);
	    pp.get("sF", sF);
	    pp.query("markstein_number", markstein);

	    Print() << "verbose = " << verbose << std::endl;
	    Print() << "unburnt_density = " << unburnt_density << std::endl;
	    Print() << "burnt_density = " << burnt_density << std::endl;
	    Print() << "nSteps = " << nSteps << std::endl;
	    Print() << "nFOSteps = " << nFOSteps << std::endl;
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
    Print() << "nFOSteps = " << nFOSteps << std::endl;
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
    const int sField_nGrow = 1;
    const int sField_nComp = 1;
    FillPatchIterator sFieldFPI(ns_level,gField,sField_nGrow,
				navier_stokes->state[State_Type].prevTime(),
				State_Type,GField,sField_nComp);
    MultiFab& sField = sFieldFPI.get_mf();

    // build multifab for RS data using s
    const int rsField_nGrow = 0;
    const int rsField_nComp = 3;
    MultiFab  rsField       = MultiFab(grids,dmap,rsField_nComp,rsField_nGrow,
				       MFInfo(),navier_stokes->Factory());
    // set rsField
    for (MFIter mfi(sField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
	const Box&          bx     = mfi.tilebox();
	Array4<Real> const& s      = sField.array(mfi);
	Array4<Real> const& rs     = rsField.array(mfi);
	const Real*         dx     = navier_stokes->geom.CellSize();
	setRS(s,rs,dx,bx);
      }

    //
    // loop to |gradG| = 1
    //
    for (int n=0; n<nSteps; n++) {
      
      if (LevelSet::verbose > 2) {
	Print() << "*** LevelSet ***: re-initialising levelset, step ="
		<< n << " / " << LevelSet::nSteps << std::endl;
      }

      // get grown G by fpi
      const int gGrown_nGrow = 1;
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
	  Array4<Real> const& rs     = rsField.array(mfi);
	  Array4<Real> const& grd    = gradGField.array(mfi);
	  const Real*         dx     = navier_stokes->geom.CellSize();
	  foGradG(gGrown,s,grd,dx,bx);
	  rsTerm(gGrown,rs,grd,dx,bx);
	  updateG(g,s,rs,grd,dx,bx);
      }
    }
    //
    // now make a copy of G and do a few steps at first order to make it smooth
    //
    const int copy_srcComp=GField;
    const int copy_dstComp=SmoothGField;
    const int copy_nComp=1;
    const int copy_nGrow=0;
    MultiFab::Copy(gField,gField,copy_srcComp,copy_dstComp,copy_nComp,copy_nGrow);
    
    for (int n=0; n<nFOSteps; n++) {
      
      if (LevelSet::verbose > 2) {
	Print() << "*** LevelSet ***: re-initialising levelset, FOstep ="
		<< n << " / " << LevelSet::nFOSteps << std::endl;
      }

      // get grown G by fpi
      const int gGrown_nGrow = 1;
      const int gGrown_nComp = 1;
      FillPatchIterator gGrownFPI(ns_level,gField,gGrown_nGrow,
				  navier_stokes->state[State_Type].prevTime(),
				  State_Type,SmoothGField,gGrown_nComp);
      MultiFab& gGrownField = gGrownFPI.get_mf();

      // evaluate the first-order gradient, russo term, and update G
      for (MFIter mfi(gField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
	  const Box&          bx      = mfi.tilebox();
	  Array4<Real> const& gGrown  = gGrownField.array(mfi);
	  Array4<Real> const& gSmooth = gField.array(mfi,SmoothGField);
	  Array4<Real> const& s       = sField.array(mfi);
	  Array4<Real> const& grd     = gradGField.array(mfi);
	  const Real*         dx      = navier_stokes->geom.CellSize();
	  foGradG(gGrown,s,grd,dx,bx);
	  updateSmoothG(gSmooth,s,grd,dx,bx);
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
// the RS approach makes decisions based on S, sign of S and gradients of S
// this all only needs doing once at the beginning
// but could be done inline to same memory...
//

void
LevelSet::setRS(Array4<Real> const& s,Array4<Real> const& rs,const Real* dx, const Box& bx)
{
  ParallelFor(bx, [s, rs, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    // RS frac
    double rsFrac=0.;
    if ( (s(i,j,k)*s(i+1,j,k)<0) || (s(i,j,k)*s(i-1,j,k)<0) ||
	 (s(i,j,k)*s(i,j+1,k)<0) || (s(i,j,k)*s(i,j-1,k)<0) ) {
      rsFrac = 1.;
    } else {
      rsFrac = 0.;
    }
    // sign S
    Real signS=0.;
    if (s(i,j,k)>0) signS=1.; else if (s(i,j,k)<0) signS=-1.;
    // D
    Real dSdxC = (s(i+1,j,k)-s(i-1,j,k))/(2.*dx[0]);
    Real dSdyC = (s(i,j+1,k)-s(i,j-1,k))/(2.*dx[1]);
    Real D = fabs(s(i,j,k)) / sqrt( dSdxC*dSdxC + dSdyC*dSdyC + 1e-200 );
    // set rs data
    rs(i,j,k,0) = rsFrac;
    rs(i,j,k,1) = signS;
    rs(i,j,k,2) = D;
  });
}

//
// first-order upwinded gradient
//

void
LevelSet::foGradG(Array4<Real> const& g,
		  Array4<Real> const& s,
		  Array4<Real> const& grd,
		  const Real*         dx,
		  const Box&          bx)
{
  ParallelFor(bx, [g, s, grd, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    // upwind in x
    if ( (g(i+1,j,k) - g(i-1,j,k)) * s(i,j,k) > 0) {
      grd(i,j,k,0) = (g(i,j,k) - g(i-1,j,k)) / dx[0]; // backward difference
    } else {
      grd(i,j,k,0) = (g(i+1,j,k) - g(i,j,k)) / dx[0]; // forward difference
    }
    // upwind in y
    if ( (g(i,j+1,k) - g(i,j-1,k)) * s(i,j,k) > 0) {
      grd(i,j,k,1) = (g(i,j,k) - g(i,j-1,k)) / dx[1]; // backward difference
    } else {
      grd(i,j,k,1) = (g(i,j+1,k) - g(i,j,k)) / dx[1]; // forward difference
    }
    // mod grad G squared
    Real modGradG2 = pow(grd(i,j,k,0),2) + pow(grd(i,j,k,1),2);
    // set modGradG
    grd(i,j,k,AMREX_SPACEDIM) = std::sqrt(modGradG2);
  });
}

//
// RS term
//

void
LevelSet::rsTerm(Array4<Real> const& g,
		 Array4<Real> const& rs,
		 Array4<Real> const& grd,
		 const Real*         dx,
		 const Box&          bx)
{
  int rsComp = AMREX_SPACEDIM+1; // tag RS term onto the end of gradient
  ParallelFor(bx, [g, rs, grd, dx, rsComp]
  AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    Real D      = rs(i,j,k,2);
    grd(i,j,k,rsComp) = ( fabs(g(i,j,k)) - D ) / dx[0]; 
  });
}

//
//
//

void
LevelSet::updateG(Array4<Real> const& g,
		  Array4<Real> const& s,
		  Array4<Real> const& rs,
		  Array4<Real> const& grd,
		  const Real* dx,
		  const Box& bx)
{
  const Real tau = 0.5 * dx[0];  // pseudo time step    
  ParallelFor(bx, [g, s, rs, grd, tau, dx]
  AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    // FO gradient
    Real FO     = grd(i,j,k,AMREX_SPACEDIM);

    // RS "gradient"
    Real rsFrac = rs(i,j,k,0);
    Real signS  = rs(i,j,k,1);
    Real RS     = grd(i,j,k,AMREX_SPACEDIM+1);

    // combine and update
    g(i,j,k) = g(i,j,k) - tau * signS * ( rsFrac*RS + (1.-rsFrac)*(FO-1.) );

    // limit to nWidth cells
    g(i,j,k) = max(-nWidth*dx[0],min(nWidth*dx[0],g(i,j,k)));
  });
}

//
//
//

void
LevelSet::updateSmoothG(Array4<Real> const& gSmooth,
			Array4<Real> const& s,
			Array4<Real> const& grd,
			const Real* dx,
			const Box& bx)
{
  const Real tau = 0.5 * dx[0];  // pseudo time step    
  ParallelFor(bx, [gSmooth, s, grd, tau, dx]
  AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    // sign S
    Real signS=0.;
    if (s(i,j,k)>0) signS=1.; else if (s(i,j,k)<0) signS=-1.;
    // update
    gSmooth(i,j,k) = gSmooth(i,j,k) - tau * signS * ( grd(i,j,k,AMREX_SPACEDIM) -1. );
    // limit to nWidth cells
    gSmooth(i,j,k) = max(-nWidth*dx[0],min(nWidth*dx[0],gSmooth(i,j,k)));
  });
}

//
//
//

void calc4ptNormGrad(double a, double b, double c, double d,
		     const double *dx, double *grad)
{
  double mag;
  grad[0] = ( (b-a) + (d-c) ) / (2.*dx[0]);
  grad[1] = ( (a-c) + (b-d) ) / (2.*dx[1]);
  mag = sqrt(grad[0]*grad[0]+grad[1]*grad[1]);
  grad[0] /= mag;
  grad[1] /= mag;
  return;
}

double calc4ptDiv(double *a, double *b, double *c, double *d, const double *dx)
{
  double divx = ( (b[0]-a[0]) + (d[0]-c[0]) ) / (2.*dx[0]);
  double divy = ( (a[1]-c[1]) + (b[1]-d[1]) ) / (2.*dx[1]);
  return(divx+divy);
}

void
LevelSet::flamespeed(Array4<Real> const& g,
		     Array4<Real> const& sloc,
		     const Real* dx,
		     const Box& bx)
{
  const Real kapMax=0.25/(dx[0]+dx[1]); // let's cap the curvature
  ParallelFor(bx, [g, sloc, dx, kapMax] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    // calculate curvature
    Real kap(0.);
    sloc(i,j,k) = sF;
    if (fabs(g(i,j,k)) < (nWidth-2)*dx[0]) {
      double a[AMREX_SPACEDIM], b[AMREX_SPACEDIM], c[AMREX_SPACEDIM], d[AMREX_SPACEDIM];
      calc4ptNormGrad(g(i-1,j+1,k), g(i,j+1,k), g(i-1,j,k), g(i,j,k), dx, a);
      calc4ptNormGrad(g(i,j+1,k), g(i+1,j+1,k), g(i,j,k), g(i+1,j,k), dx, b);
      calc4ptNormGrad(g(i-1,j,k), g(i,j,k), g(i-1,j-1,k), g(i,j-1,k), dx, c);
      calc4ptNormGrad(g(i,j,k), g(i+1,j,k), g(i,j-1,k), g(i+1,j-1,k), dx, d);
      kap = - calc4ptDiv(a,b,c,d,dx);

      // keep curvature under control
      kap  = max(-kapMax,min(kapMax,kap));  // apply min/max
      kap *= 0.5*(1-std::tanh(2.*(fabs(g(i,j,k))-2*dx[0])/dx[0])); // numerical delta fn

      // flame speed model
      sloc(i,j,k) = sF * max(1e-2,min(4., (1. - markstein * kap * lF)));
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
    // calculate d/dn(1/rho)
    Real dRhoInvDn = (unburnt_density - burnt_density) *
      pow(1./std::cosh(g(i,j,k)/(0.5*lF)),2) *
      grd(i,j,k,AMREX_SPACEDIM) / pow(rho(i,j,k),2) / lF;
    // evaluate divU
    div_u(i,j,k) = unburnt_density * sloc(i,j,k) * (dRhoInvDn);
  });
}
