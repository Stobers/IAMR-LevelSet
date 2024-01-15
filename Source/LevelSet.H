
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
#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLEBTensorOp.H>
#else
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLTensorOp.H>
#endif


class NavierStokesBase;

class LevelSet
{
public:
    LevelSet();

    LevelSet(amrex::Amr* Parent,
	     NavierStokesBase* Caller,
	     LevelSet* coarser);


    // levelset redistance functions
    void redistance(amrex::MultiFab& gField,amrex::MultiFab& gradGField);
    void set_sfield(amrex::MultiFab& gField,amrex::MultiFab& sField);
    void update_sfield(amrex::MultiFab& gField,amrex::MultiFab& sField,amrex::MultiFab& gradGField);
    void calc_gradG(amrex::MultiFab& gField,amrex::MultiFab& sField,amrex::MultiFab& gradGField);
    void update_gField(amrex::MultiFab& gField,amrex::MultiFab& sField,amrex::MultiFab& gradGField);

    void set_rhofromG(amrex::MultiFab& gField,amrex::MultiFab& density);

    void calc_divU(amrex::MultiFab& div_u, amrex::MultiFab& density, amrex::MultiFab& gradG, amrex::MultiFab& flamespeed);

    void calc_flamespeed(amrex::MultiFab& gField, amrex::MultiFab& flamespeed);
    void calc_curvature(amrex::MultiFab& gField, amrex::MultiFab& kappa);

    static int verbose;


protected:
    static void Finalize ();

    amrex::Amr* parent;
    NavierStokesBase* navier_stokes;
    const amrex::BoxArray& grids;
    const amrex::DistributionMapping& dmap;
    const int level;

    static amrex::Real unburnt_density;
    static amrex::Real burnt_density;
    static amrex::Real tau_factor;
    static amrex::Real lF;
    static amrex::Real sF;
    static amrex::Real markstein;

private:
    LevelSet* coarser;
    LevelSet* finer;
    amrex::IntVect crse_ratio;
};

#endif
