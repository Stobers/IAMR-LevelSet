#include <NavierStokesBase.H>
#include "NS_derive.H"
#ifdef AMREX_USE_EB
#include <AMReX_EBFArrayBox.H>
#endif

// some useful macros
#define SIGN(x)     ( (x > 0) ? 1 : ((x < 0) ? -1 : 0) )
#define MINABS(a,b) ( (fabs(a)<fabs(b)) ? (a) : (b) )
#define MINMOD(a,b) ( (((a)*(b))<=0) ? (0.) : (MINABS((a),(b))) )
#define DOSO

using namespace amrex;

namespace derive_functions
{
  void der_vel_avg (const Box& bx, FArrayBox& derfab, int dcomp, int ncomp,
            const FArrayBox& datfab, const Geometry& /*geomdata*/,
            Real /*time*/, const int* /*bcrec*/, int level)

  {
    amrex::ignore_unused(ncomp);
    AMREX_ASSERT(derfab.box().contains(bx));
    AMREX_ASSERT(datfab.box().contains(bx));
    AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
    AMREX_ASSERT(datfab.nComp() >= AMREX_SPACEDIM*2);
    AMREX_ASSERT(ncomp == AMREX_SPACEDIM*2);
    auto const in_dat = datfab.array();
    auto          der = derfab.array(dcomp);
    amrex::Real inv_time;
    amrex::Real inv_time_fluct;

    if (NavierStokesBase::time_avg[level] == 0){
      inv_time = 1.0;
    }else{
      inv_time = 1.0 / NavierStokesBase::time_avg[level];
    }

    if (NavierStokesBase::time_avg_fluct[level] == 0){
      inv_time_fluct = 1.0;
    }else{
      inv_time_fluct = 1.0 / NavierStokesBase::time_avg_fluct[level];
    }

    amrex::ParallelFor(bx, AMREX_SPACEDIM, [inv_time,inv_time_fluct,der,in_dat]
    AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        der(i,j,k,n) = in_dat(i,j,k,n) * inv_time;
        der(i,j,k,n+AMREX_SPACEDIM) = sqrt(in_dat(i,j,k,n+AMREX_SPACEDIM) * inv_time_fluct);
    });
  }

  //
  //  Compute cell-centered pressure as average of the
  //  surrounding nodal values
  //
  void deravgpres (const Box& bx, FArrayBox& derfab, int dcomp, int ncomp,
           const FArrayBox& datfab, const Geometry& /*geomdata*/,
           Real /*time*/, const int* /*bcrec*/, int /*level*/)

  {
    amrex::ignore_unused(ncomp);
    AMREX_ASSERT(derfab.box().contains(bx));
    AMREX_ASSERT(Box(datfab.box()).enclosedCells().contains(bx));
    AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
    AMREX_ASSERT(datfab.nComp() >= 1);
    AMREX_ASSERT(ncomp == 1);

    auto const in_dat = datfab.array();
    auto          der = derfab.array(dcomp);
#if (AMREX_SPACEDIM == 2 )
    Real factor = 0.25;
#elif (AMREX_SPACEDIM == 3 )
    Real factor = 0.125;
#endif

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      der(i,j,k) =  factor * (  in_dat(i+1,j,k)     + in_dat(i,j,k)
                + in_dat(i+1,j+1,k)   + in_dat(i,j+1,k)
#if (AMREX_SPACEDIM == 3 )
                + in_dat(i+1,j,k+1)   + in_dat(i,j,k+1)
                + in_dat(i+1,j+1,k+1) + in_dat(i,j+1,k+1)
#endif
                );
    });
  }

  //
  //  Compute magnitude of vorticity
  //
  void dermgvort (const Box& bx, FArrayBox& derfab, int dcomp, int ncomp,
          const FArrayBox& datfab, const Geometry& geomdata,
          Real /*time*/, const int* /*bcrec*/, int /*level*/)

  {
    amrex::ignore_unused(ncomp);
    AMREX_ASSERT(derfab.box().contains(bx));
    AMREX_ASSERT(datfab.box().contains(bx));
    AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
    AMREX_ASSERT(datfab.nComp() >= AMREX_SPACEDIM);
    AMREX_ASSERT(ncomp == 1);

    AMREX_D_TERM(const amrex::Real idx = geomdata.InvCellSize(0);,
                 const amrex::Real idy = geomdata.InvCellSize(1);,
                 const amrex::Real idz = geomdata.InvCellSize(2););

    amrex::Array4<amrex::Real const> const& dat_arr = datfab.const_array();
    amrex::Array4<amrex::Real>       const&vort_arr = derfab.array(dcomp);

#ifdef AMREX_USE_EB
    const auto& ebfab = static_cast<EBFArrayBox const&>(datfab);
    const EBCellFlagFab& flags = ebfab.getEBCellFlagFab();
    auto typ = flags.getType(bx);
    if (typ == FabType::covered)
    {
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            vort_arr(i,j,k) = 0.0;
        });
    } else if (typ == FabType::singlevalued)
    {
    const auto& flag_fab = flags.const_array();
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      constexpr amrex::Real c0 = -1.5;
      constexpr amrex::Real c1 = 2.0;
      constexpr amrex::Real c2 = -0.5;
      if (flag_fab(i,j,k).isCovered()) {
        vort_arr(i,j,k) = 0.0;
      } else {
        amrex::Real vx = 0.0;
        amrex::Real uy = 0.0;

#if ( AMREX_SPACEDIM == 2 )

        // Need to check if there are covered cells in neighbours --
        // -- if so, use one-sided difference computation (but still quadratic)
        if (!flag_fab(i,j,k).isConnected( 1,0,0)) {
          vx = - (c0 * dat_arr(i  ,j,k,1)
              + c1 * dat_arr(i-1,j,k,1)
              + c2 * dat_arr(i-2,j,k,1)) * idx;
        } else if (!flag_fab(i,j,k).isConnected(-1,0,0)) {
          vx = (c0 * dat_arr(i  ,j,k,1)
            + c1 * dat_arr(i+1,j,k,1)
            + c2 * dat_arr(i+2,j,k,1)) * idx;
        } else {
          vx = 0.5 * (dat_arr(i+1,j,k,1) - dat_arr(i-1,j,k,1)) * idx;
        }
        // Do the same in y-direction
        if (!flag_fab(i,j,k).isConnected( 0,1,0)) {
          uy = - (c0 * dat_arr(i,j  ,k,0)
              + c1 * dat_arr(i,j-1,k,0)
              + c2 * dat_arr(i,j-2,k,0)) * idy;
        } else if (!flag_fab(i,j,k).isConnected(0,-1,0)) {
          uy = (c0 * dat_arr(i,j  ,k,0)
            + c1 * dat_arr(i,j+1,k,0)
            + c2 * dat_arr(i,j+2,k,0)) * idy;
        } else {
          uy = 0.5 * (dat_arr(i,j+1,k,0) - dat_arr(i,j-1,k,0)) * idy;
        }

        vort_arr(i,j,k) = amrex::Math::abs(vx-uy);


#elif ( AMREX_SPACEDIM == 3 )

        amrex::Real wx = 0.0;
        amrex::Real wy = 0.0;
        amrex::Real uz = 0.0;
        amrex::Real vz = 0.0;
        // Need to check if there are covered cells in neighbours --
        // -- if so, use one-sided difference computation (but still quadratic)
        if (!flag_fab(i,j,k).isConnected( 1,0,0)) {
          // Covered cell to the right, go fish left
          vx = - (c0 * dat_arr(i  ,j,k,1)
              + c1 * dat_arr(i-1,j,k,1)
              + c2 * dat_arr(i-2,j,k,1)) * idx;
          wx = - (c0 * dat_arr(i  ,j,k,2)
              + c1 * dat_arr(i-1,j,k,2)
              + c2 * dat_arr(i-2,j,k,2)) * idx;
        } else if (!flag_fab(i,j,k).isConnected(-1,0,0)) {
          // Covered cell to the left, go fish right
          vx = (c0 * dat_arr(i  ,j,k,1)
            + c1 * dat_arr(i+1,j,k,1)
            + c2 * dat_arr(i+2,j,k,1)) * idx;
          wx = (c0 * dat_arr(i  ,j,k,2)
            + c1 * dat_arr(i+1,j,k,2)
            + c2 * dat_arr(i+2,j,k,2)) * idx;
        } else {
          // No covered cells right or left, use standard stencil
          vx = 0.5 * (dat_arr(i+1,j,k,1) - dat_arr(i-1,j,k,1)) * idx;
          wx = 0.5 * (dat_arr(i+1,j,k,2) - dat_arr(i-1,j,k,2)) * idx;
        }
        // Do the same in y-direction
        if (!flag_fab(i,j,k).isConnected(0, 1,0)) {
          uy = - (c0 * dat_arr(i,j  ,k,0)
              + c1 * dat_arr(i,j-1,k,0)
              + c2 * dat_arr(i,j-2,k,0)) * idy;
          wy = - (c0 * dat_arr(i,j  ,k,2)
              + c1 * dat_arr(i,j-1,k,2)
              + c2 * dat_arr(i,j-2,k,2)) * idy;
        } else if (!flag_fab(i,j,k).isConnected(0,-1,0)) {
          uy = (c0 * dat_arr(i,j  ,k,0)
            + c1 * dat_arr(i,j+1,k,0)
            + c2 * dat_arr(i,j+2,k,0)) * idy;
          wy = (c0 * dat_arr(i,j  ,k,2)
            + c1 * dat_arr(i,j+1,k,2)
            + c2 * dat_arr(i,j+2,k,2)) * idy;
        } else {
          uy = 0.5 * (dat_arr(i,j+1,k,0) - dat_arr(i,j-1,k,0)) * idy;
          wy = 0.5 * (dat_arr(i,j+1,k,2) - dat_arr(i,j-1,k,2)) * idy;
        }
        // Do the same in z-direction
        if (!flag_fab(i,j,k).isConnected(0,0, 1)) {
          uz = - (c0 * dat_arr(i,j,k  ,0)
              + c1 * dat_arr(i,j,k-1,0)
              + c2 * dat_arr(i,j,k-2,0)) * idz;
          vz = - (c0 * dat_arr(i,j,k  ,1)
              + c1 * dat_arr(i,j,k-1,1)
              + c2 * dat_arr(i,j,k-2,1)) * idz;
        } else if (!flag_fab(i,j,k).isConnected(0,0,-1)) {
          uz = (c0 * dat_arr(i,j,k  ,0)
            + c1 * dat_arr(i,j,k+1,0)
            + c2 * dat_arr(i,j,k+2,0)) * idz;
          vz = (c0 * dat_arr(i,j,k  ,1)
            + c1 * dat_arr(i,j,k+1,1)
            + c2 * dat_arr(i,j,k+2,1)) * idz;
        } else {
          uz = 0.5 * (dat_arr(i,j,k+1,0) - dat_arr(i,j,k-1,0)) * idz;
          vz = 0.5 * (dat_arr(i,j,k+1,1) - dat_arr(i,j,k-1,1)) * idz;
        }

        vort_arr(i,j,k) = std::sqrt((wy-vz)*(wy-vz) + (uz-wx)*(uz-wx) + (vx-uy)*(vx-uy));

#endif
      }
    });
    } else // non-EB
#endif
      {
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if ( AMREX_SPACEDIM == 2 )

      amrex::Real vx = 0.5 * (dat_arr(i+1,j,k,1) - dat_arr(i-1,j,k,1)) * idx;
      amrex::Real uy = 0.5 * (dat_arr(i,j+1,k,0) - dat_arr(i,j-1,k,0)) * idy;

      vort_arr(i,j,k) = amrex::Math::abs(vx-uy);


#elif ( AMREX_SPACEDIM == 3 )

      amrex::Real vx = 0.5 * (dat_arr(i+1,j,k,1) - dat_arr(i-1,j,k,1)) * idx;
      amrex::Real wx = 0.5 * (dat_arr(i+1,j,k,2) - dat_arr(i-1,j,k,2)) * idx;

      amrex::Real uy = 0.5 * (dat_arr(i,j+1,k,0) - dat_arr(i,j-1,k,0)) * idy;
      amrex::Real wy = 0.5 * (dat_arr(i,j+1,k,2) - dat_arr(i,j-1,k,2)) * idy;

      amrex::Real uz = 0.5 * (dat_arr(i,j,k+1,0) - dat_arr(i,j,k-1,0)) * idz;
      amrex::Real vz = 0.5 * (dat_arr(i,j,k+1,1) - dat_arr(i,j,k-1,1)) * idz;

      vort_arr(i,j,k) = std::sqrt((wy-vz)*(wy-vz) + (uz-wx)*(uz-wx) + (vx-uy)*(vx-uy));
#endif
    });
      }
  }

      void derkeng (const Box& bx, FArrayBox& derfab, int dcomp, int ncomp,
        const FArrayBox& datfab, const Geometry& /*geomdata*/,
        Real /*time*/, const int* /*bcrec*/, int /*level*/)

  {
    amrex::ignore_unused(ncomp);
    AMREX_ASSERT(derfab.box().contains(bx));
    AMREX_ASSERT(Box(datfab.box()).contains(bx));
    AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
    AMREX_ASSERT(datfab.nComp() >= 1);
    AMREX_ASSERT(ncomp == 1);
    auto const in_dat = datfab.array();
    auto          der = derfab.array(dcomp);

    amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      const Real rho = in_dat(i,j,k,0);
      const Real vx  = in_dat(i,j,k,1);
      const Real vy  = in_dat(i,j,k,2);
#if (AMREX_SPACEDIM ==3)
      const Real vz  = in_dat(i,j,k,3);
#endif

      der(i,j,k) =  0.5 * rho * ( vx*vx + vy*vy
#if (AMREX_SPACEDIM == 3 )
                  + vz*vz
#endif
                  );
    });
  }

#ifdef USE_LEVELSET
  void dergradG (const Box& bx, FArrayBox& derfab, int dcomp, int ncomp,
		 const FArrayBox& datfab, const Geometry& geomdata,
		 Real /*time*/, const int* /*bcrec*/, int /*level*/)
    
  {
    amrex::ignore_unused(ncomp);
    AMREX_ASSERT(derfab.box().contains(bx));
    AMREX_ASSERT(datfab.box().contains(bx));
    AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
    AMREX_ASSERT(datfab.nComp() >= AMREX_SPACEDIM+1);
    AMREX_ASSERT(ncomp == AMREX_SPACEDIM*2);
    auto const in_dat = datfab.array();
    auto          der = derfab.array(dcomp);

    const Real* dx = geomdata.CellSize();

    const Real LSnWidth=LevelSet::nWidth;
    const Real LSsF=LevelSet::sF;
    const Real LSlF=LevelSet::lF;
    const Real LSmarkstein=LevelSet::markstein;
    
    // let's cap the curvature
    // is this really necessary?
    const Real kapMax=1./(3.*LSlF);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      // signS
      Real signS=0.;
      if (in_dat(i,j,k)>0) signS=1.; else if (in_dat(i,j,k)<0) signS=-1.;
      
      // calculate one-sided differences
#ifdef DOSO
      Real Dxxp = (in_dat(i+2,j,k)-2.*in_dat(i+1,j,k)+in_dat(i  ,j,k))/(dx[0]*dx[0]);
      Real Dxx0 = (in_dat(i+1,j,k)-2.*in_dat(i  ,j,k)+in_dat(i-1,j,k))/(dx[0]*dx[0]);
      Real Dxxm = (in_dat(i  ,j,k)-2.*in_dat(i-1,j,k)+in_dat(i-2,j,k))/(dx[0]*dx[0]);
      
      Real Dyyp = (in_dat(i,j+2,k)-2.*in_dat(i,j+1,k)+in_dat(i,j  ,k))/(dx[1]*dx[1]);
      Real Dyy0 = (in_dat(i,j+1,k)-2.*in_dat(i,j  ,k)+in_dat(i,j-1,k))/(dx[1]*dx[1]);
      Real Dyym = (in_dat(i,j  ,k)-2.*in_dat(i,j-1,k)+in_dat(i,j-2,k))/(dx[1]*dx[1]);
      
      Real Dxp = (in_dat(i+1,j,k) - in_dat(i,j,k))/dx[0] - 0.5*dx[0]*MINMOD(Dxx0,Dxxp);
      Real Dxm = (in_dat(i,j,k) - in_dat(i-1,j,k))/dx[0] + 0.5*dx[0]*MINMOD(Dxx0,Dxxm);
      Real Dyp = (in_dat(i,j+1,k) - in_dat(i,j,k))/dx[1] - 0.5*dx[1]*MINMOD(Dyy0,Dyyp);
      Real Dym = (in_dat(i,j,k) - in_dat(i,j-1,k))/dx[1] + 0.5*dx[1]*MINMOD(Dyy0,Dyym);
#else
      Real Dxp = (in_dat(i+1,j,k) - in_dat(i,j,k))/dx[0]; // forward difference
      Real Dxm = (in_dat(i,j,k) - in_dat(i-1,j,k))/dx[0]; // backward difference
      Real Dyp = (in_dat(i,j+1,k) - in_dat(i,j,k))/dx[1]; // forward difference
      Real Dym = (in_dat(i,j,k) - in_dat(i,j-1,k))/dx[1]; // backward difference
#endif
      // near-interface corrections
      if (in_dat(i,j,k)*in_dat(i+1,j,k)<0) { // correct Dxp
	Real Sm   = in_dat(i-1,j,k);
	Real S0   = in_dat(i,j,k);
	Real Sp   = in_dat(i+1,j,k);
	Real Sp2  = in_dat(i+2,j,k);
	Real Sxx0 = MINMOD( Sm-2.*S0+Sp, S0-2.*Sp+Sp2);
	Real D    = (0.5*Sxx0 - S0 - Sp);
	D         = D*D - 4.*S0*Sp;
	Real dxp  = fabs(Sxx0>1.e-10)
	  ? dx[0] * ( 0.5 + ( S0-Sp-SIGN(S0-Sp)*sqrt(D) ) / Sxx0 )
	  : dx[0] * ( S0 / (S0-Sp) );
#ifdef DOSO
	Dxp = (0.-in_dat(i,j,k))/dxp - 0.5*dxp*MINMOD(Dxx0,Dxxp);
#else
	Dxp = (0.-in_dat(i,j,k))/dxp;
#endif
      }
      if (in_dat(i,j,k)*in_dat(i-1,j,k)<0) { // correct Dxm
	Real Sm2  = in_dat(i-2,j,k);
	Real Sm   = in_dat(i-1,j,k);
	Real S0   = in_dat(i,j,k);
	Real Sp   = in_dat(i+1,j,k);
	Real Sxx0 = MINMOD( Sm-2.*S0+Sp, S0-2.*Sm+Sm2);
	Real D    = (0.5*Sxx0 - S0 - Sm);
	D         = D*D - 4.*S0*Sm;
	Real dxm  = fabs(Sxx0>1.e-10)
	  ? dx[0] * ( 0.5 + ( S0-Sm-SIGN(S0-Sm)*sqrt(D) ) / Sxx0 )
	  : dx[0] * ( S0 / (S0-Sm) );
#ifdef DOSO
	Dxm = (in_dat(i,j,k)-0)/dxm + 0.5*dxm*MINMOD(Dxx0,Dxxm);
#else
	Dxm = (in_dat(i,j,k)-0)/dxm;
#endif
      }
      if (in_dat(i,j,k)*in_dat(i,j+1,k)<0) { // correct Dyp
	Real Sm   = in_dat(i,j-1,k);
	Real S0   = in_dat(i,j,k);
	Real Sp   = in_dat(i,j+1,k);
	Real Sp2  = in_dat(i,j+2,k);
	Real Syy0 = MINMOD( Sm-2.*S0+Sp, S0-2.*Sp+Sp2);
	Real D    = (0.5*Syy0 - S0 - Sp);
	D         = D*D - 4.*S0*Sp;
	Real dyp  = fabs(Syy0>1.e-10)
	  ? dx[1] * ( 0.5 + ( S0-Sp-SIGN(S0-Sp)*sqrt(D) ) / Syy0 )
	  : dx[1] * ( S0 / (S0-Sp) );
#ifdef DOSO
	Dyp = (0.-in_dat(i,j,k))/dyp - 0.5*dyp*MINMOD(Dyy0,Dyyp);
#else
	Dyp = (0.-in_dat(i,j,k))/dyp;
#endif
      }
      if (in_dat(i,j,k)*in_dat(i,j-1,k)<0) { // correct Dym
	Real Sm2  = in_dat(i,j-2,k);
	Real Sm   = in_dat(i,j-1,k);
	Real S0   = in_dat(i,j,k);
	Real Sp   = in_dat(i,j+1,k);
	Real Syy0 = MINMOD( Sm-2.*S0+Sp, S0-2.*Sm+Sm2);
	Real D    = (0.5*Syy0 - S0 - Sm);
	D         = D*D - 4.*S0*Sm;
	Real dym  = fabs(Syy0>1.e-10)
	  ? dx[1] * ( 0.5 + ( S0-Sm-SIGN(S0-Sm)*sqrt(D) ) / Syy0 )
	  : dx[1] * ( S0 / (S0-Sm) );
#ifdef DOSO
	Dym = (in_dat(i,j,k)-0)/dym + 0.5*dym*MINMOD(Dyy0,Dyym);
#else
	Dym = (in_dat(i,j,k)-0)/dym;
#endif
      }
      // upwind in x      
      if ((in_dat(i+1,j,k) - in_dat(i-1,j,k)) * in_dat(i,j,k) > 0) {
	der(i,j,k,0) = Dxm; // backward difference
      } else {
	der(i,j,k,0) = Dxp; // forward difference
      }
      // upwind in y
      if ((in_dat(i,j+1,k) - in_dat(i,j-1,k)) * in_dat(i,j,k) > 0) {
	der(i,j,k,1) = Dym; // backward difference
      } else {
	der(i,j,k,1) = Dyp; // forward difference
      }
      Real modGradG2 = pow(der(i,j,k,0),2) + pow(der(i,j,k,1),2);
      der(i,j,k,AMREX_SPACEDIM) = std::sqrt(modGradG2);
      
      // calculate flame speed and curvature
      // default to sF and flat
      Real sloc(LSsF);
      Real kap(0.);
      // do some on-the-fly averaging
      int nAvg = 1;
      Real kapDiv = 1./(Real)((2*nAvg+1)*(2*nAvg+1));
      // only bother near the surface
      if (fabs(in_dat(i,j,k)) < (LSnWidth-3)*dx[0]) {
	for (int ii=i-nAvg; ii<=i+nAvg; ii++) {
	  for (int jj=j-nAvg; jj<=j+nAvg; jj++) {
	    // use 9-point laplacian (and assume modGradG=1)
	    kap -= (in_dat(ii-1,jj+1,k) +  2.*in_dat(ii,jj+1,k) +    in_dat(ii+1,jj+1,k) +
		 2.*in_dat(ii-1,jj  ,k) - 12.*in_dat(ii,jj  ,k) + 2.*in_dat(ii+1,jj  ,k) +
		    in_dat(ii-1,jj-1,k) +  2.*in_dat(ii,jj-1,k) +    in_dat(ii+1,jj-1,k) );
	  }
	}
	kap *= kapDiv;
	// keep curvature under control
	kap  = max(-kapMax,min(kapMax,kap));  // apply min/max
	// flame speed model
	sloc = LSsF * max(5e-1, (1. - LSmarkstein * kap * LSlF));
      }
      der(i,j,k,AMREX_SPACEDIM+1) = kap;
      der(i,j,k,AMREX_SPACEDIM+2) = sloc;
    });
  }
#endif // LEVELSET
    
  //
  // Null function
  //
  void dernull (const Box& /*bx*/,
        FArrayBox& /*derfab*/, int /*dcomp*/, int /*ncomp*/,
        const FArrayBox& /*datfab*/, const Geometry& /*geomdata*/,
        Real /*time*/, const int* /*bcrec*/, int /*level*/)

  {
    //
    // Do nothing.
    //
  }
}
