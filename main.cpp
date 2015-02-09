#define ARMA_64BIT_WORD
#include <armadillo>

#include <fstream>
#include <iomanip>
#include <time.h>
#include <string>
#include "basis/electrongas.h"
#include "solver/ccsolve.h"




using namespace std;
using namespace arma;

int main()
{
    electrongas fgas;
    fgas.generate_state_list(1,1.0);
    //cout << fgas.v(0,2,0,2) << endl;
    //double ref_e = fgas.eref(65);
    //cout << "Energy:" << ref_e/65.0 << endl;
    ccsolve solver(fgas);
    //solver.scan_amplitudes();
    double e_corr = solver.CCSD_SG(4);
    return 0;

}

