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





using namespace std;
using namespace arma;

int main()
{
    electrongas fgas;
    fgas.generate_state_list(15.0,1.0, 14);

    //cout << "Energy per particle:" << fgas.eref(14)/14.0 << " (a.u)"  << endl;
    cout << "# Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    //cout << "Energy per particle (analytic):" << 2*fgas.analytic_energy(14)/14.0 << " (rydberg)"  << endl;
    cout << "# G.Baardsens results:" << 1.9434 << endl;

    ccd solver(fgas);

    /*
    vec vVppp = zeros(600000000);
    cout << vVppp.size() << endl;
    int e = 0;
    for(int i = 0; i<2400000; ++i){
        e += 1;
    }*/

    //cout << e << endl;

    //ccsolve solver2(fgas);
    //solver2.CCSD_SG(14);


    return 0;

}

