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
    int i0 = 3;
    int i1 = 5;
    if ( argc == 3 ){
        i0 = atoi(argv[1]);
        i1 = atoi(argv[2]);
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
        fgas.generate_state_list2(i,1.0, 14);
        //cout << setprecision(16) << r_s << endl;
        //sccdt_mp solve(fgas, .3);
        bccd solver1(fgas,.3);
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
