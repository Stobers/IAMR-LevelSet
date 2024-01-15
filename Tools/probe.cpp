#include <string>
#include <iostream>
#include <set>

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_DataServices.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;


void writeDat2D(Vector<Vector<Real>> vect, std::string filename, int dim1, int dim2) {
  FILE *file = fopen(filename.c_str(),"w");
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      fprintf(file,"%e ",vect[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
  return;
}

static
void
print_usage (int,
             char* argv[])
{
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " infile infile=f1 [options] \n\tOptions:\n";
  exit(1);
}

std::string
getFileRoot(const std::string& infile)
{
  std::vector<std::string> tokens = Tokenize(infile,std::string("/"));
  return tokens[tokens.size()-1];
}

int
main (int   argc,
      char* argv[])
{
  Initialize(argc,argv);
  {
    if (argc < 2)
      print_usage(argc,argv);

    ParmParse pp;

    if (pp.contains("help"))
      print_usage(argc,argv);

    std::string plotFileName; pp.get("infile",plotFileName);
    Vector<int> is_per(BL_SPACEDIM,1);
    pp.queryarr("is_per",is_per,0,BL_SPACEDIM);
    DataServices::SetBatchMode();
    Amrvis::FileType fileType(Amrvis::NEWPLT);

    DataServices dataServices(plotFileName, fileType);
    if( ! dataServices.AmrDataOk()) {
      DataServices::Dispatch(DataServices::ExitRequest, NULL);
      // ^^^ this calls ParallelDescriptor::EndParallel() and exit()
    }
    AmrData& amrData = dataServices.AmrDataRef();
    int finestLevel = amrData.FinestLevel();
    
    pp.query("finestLevel",finestLevel);
    //finestLevel = 0; //Override for the minute
    int Nlev = finestLevel + 1;
    int nVars = pp.countval("vars");
    
    Vector<std::string> varNames(nVars);
    pp.getarr("vars",varNames);
    const int nCompIn  = nVars;
    Vector<int> destFillComps(nCompIn);
    for (int i = 0; i < nCompIn; i++) {
      destFillComps[i] = i;
    }
    int axis;
    pp.get("axis",axis);
    Vector<int> coord(AMREX_SPACEDIM);
    pp.getarr("coord",coord);
    IntVect lowcoord(coord);
    IntVect highcoord(lowcoord);
    Vector<Real> tmp(nVars+1,0.0);
    int outlength = (amrData.ProbDomain()[finestLevel]).length(axis);
    Vector<Vector<Real>> outdata(outlength,tmp);
    Vector<MultiFab*> indata(Nlev);
    Real dx = amrData.DxLevel()[finestLevel][axis];
    Real xlo = amrData.ProbLo()[0];
    for (int i = 0; i < outlength; i++) {
      outdata[i][0] = xlo+(0.5+i)*dx;
    }
    int refRatio = 2;
    const int nGrow = 0;
    for (int lev=Nlev-1; lev>=0; lev--)
    {
      
      int probeLength = (amrData.ProbDomain()[lev]).length(axis);
      highcoord[axis] = probeLength-1;
      Box probe(lowcoord,highcoord);
      BoxArray probeBA(probe);
      //const DistributionMapping dm(probeBA);
      indata[lev] = new MultiFab(probeBA,DistributionMapping(probeBA),nCompIn,nGrow);
      Print() << "Reading data for level " << lev << std::endl;
      amrData.FillVar(*indata[lev],lev,varNames,destFillComps); //Problem
      Print() << "Data has been read for level " << lev << std::endl;
      int scale = 1;
      for (MFIter mfi(*indata[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
	
        const Box& bx = mfi.tilebox();
	Array4<Real> const& inbox  = (*indata[lev]).array(mfi);
	IntVect d;
#if AMREX_SPACEDIM==2
	AMREX_PARALLEL_FOR_3D ( bx, i, j, k,
        {
	  d[0] = i;
	  d[1] = j;
	  for (int n = 1; n < nCompIn+1; n++) {
	    outdata[scale*d[axis]][n] = inbox(i,j,k,n-1);
	  }
	});
#elif AMREX_SPACEDIM==3
	AMREX_PARALLEL_FOR_3D ( bx, i, j, k,
        {
	  d[0] = i;
	  d[1] = j;
	  d[2] = k;
	  for (int n = 1; n < nCompIn+1; n++) {
	    outdata[scale*d[axis]][n] = inbox(i,j,k,n-1);
	  }
	});
#endif
	scale *= refRatio;
      }
      
      Print() << "Probe done for level " << lev << std::endl;
    }
    for (int n = 1; n<nVars+1; n++) {
      ParallelDescriptor::ReduceRealSum(outdata[n].data(),outlength);
    }

    std::string outfile(getFileRoot(plotFileName) + "_probe.dat");
    Print() << "Writing new data to " << outfile << std::endl;
    writeDat2D(outdata,outfile,outlength,nVars+1);
  }
  Finalize();
  return 0;
}
