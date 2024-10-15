
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
    void calc_flamespeed(amrex::MultiFab& gField, amrex::MultiFab& flamespeed);

    //
    // sets rho from the gfield
    //
    void set_rhofromG(amrex::MultiFab& gField,amrex::MultiFab& density);

    //
    // calculates the divU
    //
    void calc_divU(amrex::MultiFab& div_u, amrex::MultiFab& density, amrex::MultiFab& gradG, amrex::MultiFab& flamespeed);

    static int verbose;

    void get_gradG(amrex::MultiFab& gField, amrex::MultiFab& gradGField);



    static amrex::Real unburnt_density;
    static amrex::Real burnt_density;
    static int nSteps;
    static int nWidth;
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
    // --- functions called from redistance --- // 
    //
    // sets an initial guess of S
    //
    void set_sfield(amrex::MultiFab& gField,amrex::MultiFab& sField);

    //
    // updates s based of |gradG|
    //
    void update_sfield(amrex::MultiFab& gField,amrex::MultiFab& sField,amrex::MultiFab& gradGField);

    //
    // calculates the redistance equation specific |gradG|
    //
    void calc_gradG(amrex::MultiFab& gField,amrex::MultiFab& sField,amrex::MultiFab& gradGField);

    //
    // propagates the levelset during the redistance function 
    //
    void update_gField(amrex::MultiFab& gField,amrex::MultiFab& sField,amrex::MultiFab& gradGField);

    //
    // calculates the curvature of the gField
    //
    void calc_curvature(amrex::MultiFab& gField, amrex::MultiFab& kappa);

    
    LevelSet* coarser;
    LevelSet* finer;
    amrex::IntVect crse_ratio;
};
#endif
