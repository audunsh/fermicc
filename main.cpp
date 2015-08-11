#define ARMA_64BIT_WORD
#include <armadillo>

#include <iomanip>
#include <fstream>
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
#include "solver/sccdt_mp.h"
#include <omp.h>

using namespace std;
using namespace arma;


int main(int argc, char *argv[] )
{
    //default setup
    int i0 = 3;
    int i1 = 4;
    double rs = 1.0;
    uint iterations = 20;
    uint uiStatAlloc = 1000000;

    if ( argc == 6 ){
        i0 = atoi(argv[1]);
        i1 = atoi(argv[2]);
        rs = atof(argv[3]);

        iterations = atoi(argv[4]);
        uiStatAlloc = atoi(argv[5]);
    }
    else{
        cout << "FermiCC, by Audun Skau Hansen 2015." << endl;
        cout << "Usage:" << endl;
        cout << "./FermiCC [start] [end] [rs] [iterations] [static allocation]" << endl;
        cout << endl;
        cout << "start/end        :       Number of the first shell (start) and up to (but not including) the last shell (end)."<< endl;
        cout << "rs               :       Wigner-Seiz radi. See thesis." << endl;
        cout << "iterations       :       Maximum number of iterations." << endl;
        cout << "static allocation:       Maximum size of staticly allocated vectors. (use 100000 on smaller calculations.)" << endl;

        cout << "Running default calculation, i.e.:" << endl;
        cout << "./FermiCC 3 4 1.0 20 100000" << endl;

    }


    //electrongas fgas;
    //fgas.generate_state_list2(4.0,2.0, 14);
    //cout << "[Main] " << setprecision(16) << "Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    //cout << "[Main] G.Baardsens results:" << 1.9434 << endl;


    //uint Np = fgas.iNbstates-fgas.iNparticles; //conflicting notation here
    //uint Nh = fgas.iNparticles;


    //bccd solver1(fgas,.3);

    for(int i = i0; i < i1; ++i){
        electrongas fgas;
        fgas.generate_state_list2(i,rs, 14);
        //cout << setprecision(16) << r_s << endl;
        //sccdt_mp solve(fgas, .3);
        bccd solver1(fgas,.3);
        solver1.uiStatAlloc = uiStatAlloc;
        solver1.solve(iterations);
        cout << "Number of states, correlation energy, difference in last convergence, number of iterations passed" << endl;
        cout << setprecision(9) << fgas.iNbstates << "     " << solver1.dCorrelationEnergy << "     " << solver1.convergence_diff << "     "<< solver1.convergence << endl;

    }





    //ccd_pt solver2(fgas, .3);
    //cout << endl;
    //ccd solver(fgas, .2);
    //sccdt_mp(fgas, .3);



    /*
    vec results(16);

    for(uint i = 0; i < 16; ++i){
        double r_s = .5 + i*.1;
        electrongas fgas;
        fgas.generate_state_list2(3.0,r_s, 14);
        cout << setprecision(16) << r_s << endl;
        sccdt_mp solve(fgas, .3);
        results(i) = solve.correlation_energy;
        cout << r_s << endl;
    }
    for(uint i = 0; i < 16; ++i){
        cout << setprecision(16) << .5 + i*.1 << "      " << results(i) << endl;
    }*/

    return 0;


}
