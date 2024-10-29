
#ifndef IAMR_LEVELSET_H_
#define IAMR_LEVELSET_H_

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_LevelBld.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_ErrorList.H>
#include <AMReX_FluxRegister.H>
#include <FluxBoxes.H>

#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLTensorOp.H>

class NavierStokesBase;
class LevelSet
{
public:
    LevelSet();

    LevelSet(amrex::Amr* Parent,
	     NavierStokesBase* Caller,
	     LevelSet* coarser);

    //
    // redistances the gfield
    //
    void redistance(amrex::MultiFab& gField);

    //
    // calculates the flame speed using a markstine number model
    //
    void flamespeed(amrex::Array4<amrex::Real> const& g,
		    amrex::Array4<amrex::Real> const& sloc,
		    const amrex::Real* dx,
		    const amrex::Box& bx);

    //
    // sets rho from the gfield
    //
    void set_rhofromG(amrex::MultiFab& gField, amrex::MultiFab& density);

    //
    // calculates the divU
    //
    void divU(amrex::Array4<amrex::Real> const& g,
	      amrex::Array4<amrex::Real> const& div_u,
	      amrex::Array4<amrex::Real> const& rho,
	      amrex::Array4<amrex::Real> const& grd,
	      amrex::Array4<amrex::Real> const& sloc,
	      const amrex::Real* dx,
	      const amrex::Box& bx);

    void foGradG(amrex::Array4<amrex::Real> const& g,
		 amrex::Array4<amrex::Real> const& s,
		 amrex::Array4<amrex::Real> const& grd,
		 const amrex::Real* dx,
		 const amrex::Box& bx);
    
    void rsTerm(amrex::Array4<amrex::Real> const& g,
		amrex::Array4<amrex::Real> const& rs,
		amrex::Array4<amrex::Real> const& grd,
		const amrex::Real* dx,
		const amrex::Box& bx);
  
    //
    // public variables
    //
    static int nSteps;
    static int nWidth;
    static int verbose;
    static amrex::Real unburnt_density;
    static amrex::Real burnt_density;
    static amrex::Real lF;
    static amrex::Real sF;
    static amrex::Real markstein;

protected:
    static void Finalize ();

    amrex::Amr* parent;
    NavierStokesBase* navier_stokes;
    const amrex::BoxArray& grids;
    const amrex::DistributionMapping& dmap;
    const int level;

private:

    void setRS(amrex::Array4<amrex::Real> const& s,
	       amrex::Array4<amrex::Real> const& rs,
	       const amrex::Real*                dx,
	       const amrex::Box&                 bx);

    void updateG(amrex::Array4<amrex::Real> const& g,
		 amrex::Array4<amrex::Real> const& s,
		 amrex::Array4<amrex::Real> const& rs,
		 amrex::Array4<amrex::Real> const& grd,
		 const amrex::Real*                dx,
		 const amrex::Box&                 bx,
		 const int                         forceFO);

    LevelSet* coarser;
    LevelSet* finer;
    amrex::IntVect crse_ratio;
};
#endif
