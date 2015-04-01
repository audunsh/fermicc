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
    fgas.generate_state_list(6.0,1.0, 14);

    fgas.mu = 7.695;
    fgas.mu = 0; //2.5*3.1415;
    cout << endl;
    //cout << "Energy per particle:" << fgas.eref(14)/14.0 << " (a.u)"  << endl;
    cout << "Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    //cout << "Energy per particle (analytic):" << 2*fgas.analytic_energy(14)/14.0 << " (rydberg)"  << endl;
    cout << "Compared to:" << 1.9434 << endl;

    ccd solver(fgas);

    return 0;

}

