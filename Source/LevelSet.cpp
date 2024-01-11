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
Real LevelSet::tau_factor = 0.1;
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
	    pp.query("tau_factor", tau_factor);
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
//		s(i,j,k) = g(i,j,k) / std::sqrt(pow(g(i,j,k),2) + pow(dx[0],2));
		Real alpha = 2;
		Real eps = alpha * dx[0];
		Real hdx;
		if (g(i,j,k) < -eps) {
		    hdx = 0;
		}
		else if (std::abs(g(i,j,k)) <= eps) {
		    hdx = 0.5 * (1 + (g(i,j,k)/eps) + (1/3.14) * sin(3.14*g(i,j,k)/eps));
		}
		else {
		    hdx = 1;
		}
		s(i,j,k) = 2 * (hdx - 0.5);
	    });
    }
}

void
LevelSet::update_sfield(MultiFab& gField, MultiFab& sField, MultiFab& gradGField)
{
    for (MFIter mfi(gField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.tilebox();
	Array4<Real> const& g = gField.array(mfi,GField);
	Array4<Real> const& s = sField.array(mfi);
	Array4<Real> const& grd = gradGField.array(mfi,MagGradg);
	const Real* dx = navier_stokes->geom.CellSize();
	ParallelFor(bx, [g, s, grd,  dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	    {
		s(i,j,k) = g(i,j,k)
		    / std::sqrt(pow(g(i,j,k),2) + pow(grd(i,j,k,0),2) * pow(dx[0],2));
	    });
    }
}

void
LevelSet::calc_gradG(MultiFab& gField, MultiFab& sField, MultiFab& gradGField)
{
    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));
    const int g_nGrow = 4;
    FillPatchIterator fpiG(ns_level,gField,g_nGrow,
			   navier_stokes->state[State_Type].prevTime(),
			   State_Type,GField,1);
    MultiFab& gfpi = fpiG.get_mf();
    
    for (MFIter mfi(gradGField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.tilebox();
	Array4<Real> const& g   = gfpi.array(mfi);
	Array4<Real> const& s   = sField.array(mfi);
	Array4<Real> const& grd = gradGField.array(mfi,MagGradg);
	const Real* dx = navier_stokes->geom.CellSize();
	ParallelFor(bx, [g, s, grd, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
		{		
		    Real gm, gp;
		    Real ap,bp,cp,dp;
		    Real am,bm,cm,dm;
		    
		    // x
		    gm = (g(i,j,k) - g(i-1,j,k)) / dx[0];
		    gp = (g(i+1,j,k) - g(i,j,k)) / dx[0];
		    ap = std::max(gm,0.);
		    am = std::min(gm,0.);
		    bp = std::max(gp,0.);
		    bm = std::min(gp,0.);
		    
		    // y	    
		    gm = (g(i,j,k) - g(i,j-1,k)) / dx[1];
		    gp = (g(i,j+1,k) - g(i,j,k)) / dx[1];
		    cp = std::max(gm,0.);
		    cm = std::min(gm,0.);
		    dp = std::max(gp,0.);
		    dm = std::min(gp,0.);
		    
		    // z
#if (AMREX_SPACEDIM == 3)
		    Real ep,fp;
		    Real em,fm;
		    gm = (g(i,j,k) - g(i,j,k-1)) / dx[2];
		    gp = (g(i,j,k+1) - g(i,j,k)) / dx[2];		    
		    ep = std::max(gm,0.);
		    em = std::min(gm,0.);
		    fp = std::max(gp,0.);
		    fm = std::min(gp,0.);
#endif
		    if (s(i,j,k) > 0) {
			grd(i,j,k,0) = std::sqrt(std::max(std::abs(ap),std::abs(bm))
						 + std::max(std::abs(cp),std::abs(dm))
#if (AMREX_SPACEDIM == 3)
						 + std::max(std::abs(ep),std::abs(fm))
#endif
			    );
		    }
		    else if (s(i,j,k) < 0) {
			grd(i,j,k,0) = std::sqrt(std::max(std::abs(am),std::abs(bp))
						 + std::max(std::abs(cm),std::abs(dp))
#if (AMREX_SPACEDIM == 3)
						 + std::max(std::abs(em),std::abs(fp))
#endif
			    );			
		    }
		    else {
			grd(i,j,k,0) = 0;
		    }
		    
		    grd(i,j,k,1) = (g(i+1,j,k) - g(i-1,j,k)) / (2*dx[0]);
		    grd(i,j,k,2) = (g(i,j+1,k) - g(i,j-1,k)) / (2*dx[1]);
#if (AMREX_SPACEDIM == 3)
		    grd(i,j,k,3) = (g(i,j,k+1) - g(i,j,k-1)) / (2*dx[1]);
#endif
		});
    }    
}


void
LevelSet::calc_gradG2(MultiFab& gField, MultiFab& sField, MultiFab& gradGField)
{
    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));
    const int g_nGrow = 10;
    FillPatchIterator fpiG(ns_level,gField,g_nGrow,
			   navier_stokes->state[State_Type].prevTime(),
			   State_Type,GField,1);
    MultiFab& gfpi = fpiG.get_mf();
    
    for (MFIter mfi(gradGField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.tilebox();
	Array4<Real> const& g   = gfpi.array(mfi);
	Array4<Real> const& s   = sField.array(mfi);
	Array4<Real> const& grd = gradGField.array(mfi,MagGradg);
	const Real* dx = navier_stokes->geom.CellSize();
	ParallelFor(bx, [g, s, grd, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
		{
		    Real I, J;
		    Real d1, d2, a, b, c;
		    Real dm, dp;

		    // ------ x ------//
		    // --- upwind --- //
		    I = i-1;
		    // 1st Order
		    d1 = (g(I+1,j,k) - g(I,j,k)) / dx[0];
		    // 2nd Order
		    a = (g(I-1,j,k) - 2*g(I,j,k) + g(I+1,j,k)) / pow(dx[0],2);
		    b = (g(I,j,k) - 2*g(I+1,j,k) + g(I+2,j,k)) / pow(dx[0],2);
		    if (std::abs(a) <= std::abs(b)) {
			c = a;
		    }
		    else {
			c = b;
		    }
		    d2 = d1 - (dx[0] / 2) * c * (2 * (I - i) + 1);
		    dm = d2;
		    I = i;
		    // 1st Order
		    d1 = (g(I+1,j,k) - g(I,j,k)) / dx[0];
		    // 2nd Order
		    a = (g(I-1,j,k) - 2*g(I,j,k) + g(I+1,j,k)) / pow(dx[0],2);
		    b = (g(I,j,k) - 2*g(I+1,j,k) + g(I+2,j,k)) / pow(dx[0],2);
		    if (std::abs(a) <= std::abs(b)) {
			c = a;
		    }
		    else {
			c = b;
		    }
		    d2 = d1 - (dx[0] / 2) * c * (2 * (I - i) + 1);
		    dp = d2;

		    // gradx
		    if (dp * s(i,j,k) < 0 && dm * s(i,j,k) < -dp * s(i,j,k)) {
			grd(i,j,k,1) = dp;
		    }
		    else if (dm * s(i,j,k) > 0 && dp * s(i,j,k) > -dm * s(i,j,k)) {
			grd(i,j,k,1) = dm;
		    }
		    else if (dm * s(i,j,k) < 0 && dp * s(i,j,k) > 0) {
			grd(i,j,k,1) = (dp + dm) / 2;
		    }
		    else {
			grd(i,j,k,1) = 1e-99;
 		    }


		    // ------ y ------//
		    // --- upwind --- //
		    // minus
		    J = j-1;
		    // 1st Order
		    d1 = (g(i,J+1,k) - g(i,J,k)) / dx[1];
		    // 2nd Order
		    a = (g(i,J-1,k) - 2*g(i,J,k) + g(i,J+1,k)) / pow(dx[1],2);
		    b = (g(i,J,k) - 2*g(i,J+1,k) + g(i,J+2,k)) / pow(dx[1],2);
		    if (std::abs(a) <= std::abs(b)) {
			c = a;
		    }
		    else {
			c = b;
		    }
		    d2 = d1 - (dx[1] / 2) * c * (2 * (J - j) + 1);
		    dm = d2;
		    // plus
		    J = j;
		    // 1st Order
		    d1 = (g(i,J+1,k) - g(i,J,k)) / dx[1];
		    // 2nd Order
		    a = (g(i,J-1,k) - 2*g(i,J,k) + g(i,J+1,k)) / pow(dx[1],2);
		    b = (g(i,J,k) - 2*g(i,J+1,k) + g(i,J+2,k)) / pow(dx[1],2);
		    if (std::abs(a) <= std::abs(b)) {
			c = a;
		    }
		    else {
			c = b;
		    }
		    d2 = d1 - (dx[1] / 2) * c * (2 * (J - j) + 1);
		    dp = d2;

		    // grady
		    if (dp * s(i,j,k) < 0 && dm * s(i,j,k) < -dp * s(i,j,k)) {
			grd(i,j,k,2) = dp;
		    }
		    else if (dm * s(i,j,k) > 0 && dp * s(i,j,k) > -dm * s(i,j,k)) {
			grd(i,j,k,2) = dm;
		    }
		    else if (dm * s(i,j,k) < 0 && dp * s(i,j,k) > 0) {
			grd(i,j,k,2) = (dp + dm) / 2;
		    }
		    else {
			grd(i,j,k,2) = 1e-99;
		    }

#if (AMREX_SPACEDIM == 3)
		    Real K;
		    // ------ z ------//
		    // --- upwind --- //
		    // minus
		    K = k-1;
		    // 1st Order
		    d1 = (g(i,j,K+1) - g(i,j,K)) / dx[2];
		    // 2nd Order
		    a = (g(i,j,K-1) - 2*g(i,j,K) + g(i,j,K+1)) / pow(dx[2],2);
		    b = (g(i,j,K) - 2*g(i,j,K+1) + g(i,j,K+2)) / pow(dx[2],2);
		    if (std::abs(a) <= std::abs(b)) {
			c = a;
		    }
		    else {
			c = b;
		    }
		    d2 = d1 - (dx[1] / 2) * c * (2 * (K - k) + 1);
		    dm = d2;
		    // plus
		    K = k;
		    // 1st Order
		    d1 = (g(i,j,K+1) - g(i,j,K)) / dx[2];
		    // 2nd Order
		    a = (g(i,j,K-1) - 2*g(i,j,K) + g(i,j,K+1)) / pow(dx[2],2);
		    b = (g(i,j,K) - 2*g(i,j,K+1) + g(i,j,K+2)) / pow(dx[2],2);
		    if (std::abs(a) <= std::abs(b)) {
			c = a;
		    }
		    else {
			c = b;
		    }
		    d2 = d1 - (dx[1] / 2) * c * (2 * (K - k) + 1);
		    dp = d2;

		    // gradz
		    if (dp * s(i,j,k) < 0 && dm * s(i,j,k) < -dp * s(i,j,k)) {
			grd(i,j,k,3) = dp;
		    }
		    else if (dm * s(i,j,k) > 0 && dp * s(i,j,k) > -dm * s(i,j,k)) {
			grd(i,j,k,3) = dm;
		    }
		    else if (dm * s(i,j,k) < 0 && dp * s(i,j,k) > 0) {
			grd(i,j,k,3) = (dp + dm) / 2;
		    }
		    else {
			grd(i,j,k,3) = 1e-99;
		    }
#endif
		    grd(i,j,k,0) = std::sqrt(pow(grd(i,j,k,1),2)+pow(grd(i,j,k,2),2)
#if (AMREX_SPACEDIM == 3)
					     +pow(grd(i,j,k,3),2)
#endif
			);
		});
    }
}


void
LevelSet::update_gField(MultiFab& gField, MultiFab& sField, MultiFab& gradGField)
{
    const Real* dx = navier_stokes->geom.CellSize();
    const Real tau = tau_factor * dx[0];  // pseudo time step
   
    // updates gfield using forward time
    for (MFIter mfi(gField,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box& bx = mfi.tilebox();
	Array4<Real> const& g   = gField.array(mfi,GField);
	Array4<Real> const& grd = gradGField.array(mfi,MagGradg);
	Array4<Real> const& s   = sField.array(mfi);
	ParallelFor(bx, [g, grd, s, tau] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
	    {
		g(i,j,k) = g(i,j,k) - tau * s(i,j,k) * (grd(i,j,k,0) - 1);
	    });
    }
}



void
LevelSet::redistance(MultiFab& gField,MultiFab& gradGField)
{
    if (LevelSet::verbose == 1) {
	Print() << "LevelSet redistancing levelset \n";
    }
    MultiFab sField  = MultiFab(grids,dmap,1,1,MFInfo(), navier_stokes->Factory());
    Real nsteps = 10 / tau_factor;
    set_sfield(gField, sField);
    for (int n=0; n<nsteps; n++) {
	calc_gradG2(gField, sField, gradGField);
	update_gField(gField, sField, gradGField);
        update_sfield(gField, sField, gradGField);
    }
}


void
LevelSet::set_rhofromG(MultiFab& gField, MultiFab& density)
{
    for (MFIter mfi(density,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
	const Box&  bx       = mfi.tilebox();
	auto const& rho      = density.array(mfi,Density);
	auto const& g        = gField.array(mfi,GField);
	const auto dx        = navier_stokes->geom.CellSizeArray();
	amrex::ParallelFor(bx, [rho, g, dx] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
	    {
//		rho(i,j,k) = unburnt_density + (0.5 * (burnt_density - unburnt_density)) * (1 + std::tanh(g(i,j,k)/(1.6*dx[0])));
//		rho(i,j,k) = unburnt_density + (0.5 * (burnt_density - unburnt_density)) * (1 + std::tanh(g(i,j,k)/(7*dx[0])));
		rho(i,j,k) = unburnt_density + (0.5 * (burnt_density - unburnt_density)) * (1 + std::tanh(g(i,j,k)/(0.5*lF)));
	    });
    }
}


void
LevelSet::calc_divU(MultiFab& div_u, MultiFab& density, MultiFab& gradG, MultiFab& flamespeed)
{
    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));

    int nGrow = 2;
    
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
		    + (ny * ((1/rho(i,j+1,k)) - (1/rho(i,j-1,k))) / (2*dx[1]))
#if (AMREX_SPACEDIM == 3)
		    + (nz * ((1/rho(i,j,k)) - (1/rho(i,j,k-1))) / dx[2])
#endif
		    ;
		divu(i,j,k) = unburnt_density * sloc(i,j,k) * dndrho;
	    });
    }
}

void
LevelSet::calc_curvature(MultiFab& gField, MultiFab& kappa)
{
    // this does not yet do 3D!
    NavierStokesBase& ns_level = *(NavierStokesBase*) &(parent->getLevel(level));

    const int nGrow = 4;
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
		if (g(i,j,k) < 10*dx[0] && g(i,j,k) > -10*dx[0]) {
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
		if (g(i,j,k) < 10*dx[0] && g(i,j,k) > -10*dx[0]) {
		    sloc(i,j,k) = sF * (1 - markstein * kap(i,j,k) * lF);
		}
		else {
		    sloc(i,j,k) = sF;
		}
		if (sloc(i,j,k) <= 0) {
		    sloc(i,j,k) = 1.0e-5;		    
		}
		else if (sloc(i,j,k) > 4*sF) {
		    sloc(i,j,k) = 4*sF;
		}
	    });
    }    
}
