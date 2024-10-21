#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_DataServices.H>
#include <AMReX_WritePlotFile.H>

using namespace amrex;

int
main (int   argc,
      char* argv[])
{
  Initialize(argc,argv);
  {
    ParmParse pp;

    // Open first plotfile header and create an amrData object pointing into it
    int nPlotFiles = pp.countval("infiles");
    if (nPlotFiles<2) {
	Abort("needs at least 2 plot files");
    }
    Vector<std::string> plotFileNames; pp.getarr("infiles",plotFileNames,0,nPlotFiles);
    Real axis=0; pp.query("vaxis",axis);
    
    const int nVars(1);
    Vector<std::string> whichVar(nVars);
    whichVar[0] = "gfield";
    
    Vector<int>  destFills(nVars);
    for (int c=0; c<nVars; c++)
	destFills[c] = c;

    DataServices::SetBatchMode();
    Amrvis::FileType fileType(Amrvis::NEWPLT);

    Vector<DataServices *> dataServicesPtrVector(nPlotFiles);                                         // DataServices array for each plot
    Vector<AmrData *>      amrDataPtrVector(nPlotFiles);                                              // DataPtrVector for each plot
    Vector<Real>           time(nPlotFiles);
    Vector<Real>           volume(nPlotFiles);
    Real                   area;
    
    for(int iPlot = 0; iPlot < nPlotFiles; ++iPlot) {
	if (ParallelDescriptor::IOProcessor())
	    std::cout << "Loading " << plotFileNames[iPlot] << std::endl;
	
	dataServicesPtrVector[iPlot] = new DataServices(plotFileNames[iPlot], fileType);               // Populate DataServices array
	
	if( ! dataServicesPtrVector[iPlot]->AmrDataOk())                                               // Check AmrData ok
	    DataServices::Dispatch(DataServices::ExitRequest, NULL);                                    // Exit if not
	
	amrDataPtrVector[iPlot] = &(dataServicesPtrVector[iPlot]->AmrDataRef());                        // Populate DataPtrVector
	
	time[iPlot] = amrDataPtrVector[iPlot]->Time();	
    }

    for (int iPlot=0; iPlot<nPlotFiles; iPlot++) {
    
	int finestLevel = amrDataPtrVector[iPlot]->FinestLevel();    
	int inFinestLevel(-1);    pp.query("finestLevel",inFinestLevel);
	if (inFinestLevel>-1 && inFinestLevel<finestLevel) {
	    finestLevel = inFinestLevel;
            if (ParallelDescriptor::IOProcessor())
	        std::cout << "Finest level: " << finestLevel << std::endl;
	}

	Vector<Real> probLo=amrDataPtrVector[iPlot]->ProbLo();
	Vector<Real> probHi=amrDataPtrVector[iPlot]->ProbHi();
#if (AMREX_SPACEDIM == 2)
	// 0 = x2x, 1=y2y
	if (axis==0) {
	    area = probHi[0] - probLo[0];
	    Print() << area << std::endl;
	}
	else {
	    area = probHi[1] - probLo[1];
	}
#else
	// 0 = x2x * y2y,  1 = x2x * z2z,  2 = y2y * z2z
	if (axis==0) {
	    area = (probHi[0] - probLo[0]) * (probHi[1] - probLo[1]);
	}
	else if (axis==1) {
	    area = (probHi[0] - probLo[0]) * (probHi[2] - probLo[2]);
	}
	else {
	    area = (probHi[1] - probLo[1]) * (probHi[2] - probLo[2]);
	}
#endif
	
	const Real *dx = amrDataPtrVector[iPlot]->DxLevel()[finestLevel].dataPtr();
	Real dxyz = dx[0]*dx[1]
#if (AMREX_SPACEDIM == 3)
	    *dx[2]
#endif
	    ;

	int ngrow(0);
	MultiFab mf;
        const BoxArray& ba = amrDataPtrVector[iPlot]->boxArray(finestLevel);
        DistributionMapping dm(ba);
	mf.define(ba, dm, nVars, ngrow);

        if (ParallelDescriptor::IOProcessor())
	    std::cout << "Processing " << iPlot << "/" << nPlotFiles << std::endl;
	amrDataPtrVector[iPlot]->FillVar(mf, finestLevel, whichVar, destFills);
	for (int n=0; n<nVars; n++)
	    amrDataPtrVector[iPlot]->FlushGrids(amrDataPtrVector[iPlot]->StateNumber(whichVar[n]));
	Real vol(0);
	for(MFIter ntmfi(mf); ntmfi.isValid(); ++ntmfi) {
	    const FArrayBox &myFab = mf[ntmfi];
	    Vector<const Real *> varPtr(nVars);
	    for (int v=0; v<nVars; v++)
		varPtr[v] = myFab.dataPtr(v);
            const Box& vbx = ntmfi.validbox();
	    const int  *lo  = vbx.smallEnd().getVect();
	    const int  *hi  = vbx.bigEnd().getVect();

	    int ix = hi[0]-lo[0]+1;
	    int jx = hi[1]-lo[1]+1;
#if (AMREX_SPACEDIM == 2)
	    for (int j=0; j<jx; j++) {
		Real y=probLo[1] + dx[1]*(0.5+(Real)(j+lo[1]));
		for (int i=0; i<ix; i++) {
		    Real x=probLo[0] + dx[0]*(0.5+(Real)(i+lo[0]));
		    int cell = j*ix+i;
		    Real g = varPtr[0][cell];
		    if (g < 0) {
			vol += dxyz;
		    }
		}
	    }
#else
	    int kx = hi[2]-lo[2]+1;
	    for (int k=0; k<kx; k++) {
		Real z=probLo[2] + dx[2]*(0.5+(Real)(k+lo[2]));
		for (int j=0; j<jx; j++) {
		    Real y=probLo[1] + dx[1]*(0.5+(Real)(j+lo[1]));
		    for (int i=0; i<ix; i++) {
			Real x=probLo[0] + dx[0]*(0.5+(Real)(i+lo[0]));
			int cell = (k*jx+j)*ix+i;
			Real g = varPtr[0][cell];
       			if (g < 0) {
			    vol += dxyz;
			}
		    }
		}
	    }
#endif
	}
	ParallelDescriptor::ReduceRealSum(vol);
	volume[iPlot] = vol;
    }
    if (ParallelDescriptor::IOProcessor())
        std::cout << "   ...done." << std::endl;

    if (ParallelDescriptor::IOProcessor()) {
	FILE *file = fopen("flamespeed.dat","w");
	Real sum_flamespeed = 0.0;
	for (int iPlot=1; iPlot<nPlotFiles; iPlot++) {
	    Real dV = std::abs(volume[iPlot] - volume[iPlot-1]);
	    Real dt = time[iPlot] - time[iPlot-1];
	    Real flamespeed = (dV / dt) / area;
	    fprintf(file,"%e %e\n",time[iPlot],flamespeed);
	    sum_flamespeed += flamespeed;
	}
	fclose(file);
	Print() << "global flame speed = " << sum_flamespeed / (nPlotFiles-2) << std::endl;
    }
  }
  Finalize();
  return 0;
}
