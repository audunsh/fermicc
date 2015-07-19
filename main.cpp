#define ARMA_64BIT_WORD
#include <armadillo>

#include <fstream>
#include <iomanip>
#include <time.h>
#include <string>
#include "basis/electrongas.h"
#include "solver/ccsolve.h"
#include "solver/ccd.h"
#include "solver/initializer.h"
#include "solver/flexmat.h"
#include "solver/ccd_pt.h"
#include "solver/ccdt.h"
#include "solver/ccdt_mp.h"
#include "solver/ccd_mp.h"
#include "solver/amplitude.h"
#include "solver/bccd.h"
#include <omp.h>

using namespace std;
using namespace arma;


int main()
{
    //TODO LIST
    //1. Speed up initialization
    //2. Experiment with uniquely reduced t3amps in setup
    //3. parallellization
    electrongas fgas;
    fgas.generate_state_list2(3.0,1.0, 14);
    cout << "[Main] " << setprecision(8) << "Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    cout << "[Main] G.Baardsens results:" << 1.9434 << endl;


    uint Np = fgas.iNbstates-fgas.iNparticles; //conflicting notation here
    uint Nh = fgas.iNparticles;


    //triples diagrams corresponds to eachother when added in separately, but not together (deviation)
    bccd solver1(fgas);
    cout << endl;
    ccd_pt solver2(fgas, 0);


    /*
    double tm = omp_get_wtime();

    #pragma omp parallel for num_threads(4)
    for(uint i = 0; i < 1000000; ++i){
        mat block(100,100);
        block*=0;
    }*/

    //cout << "ts:" << omp_get_wtime()-tm << endl;

    //umat test(4,1);
    //test(0,0) = 0;
    //test.print();
    return 0;

}
