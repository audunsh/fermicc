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
    fgas.generate_state_list(3.0,1.0, 14);
    for(int i = 0; i<14; i++){
        cout << i << " " << fgas.h(i,i) << endl;
    }
    cout << endl;
    cout << "Energy per particle:" << fgas.eref(14)/14.0 << " (a.u)"  << endl;
    cout << "Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    cout << "Energy per particle (analytic):" << 2*fgas.analytic_energy(14)/14.0 << " (rydberg)"  << endl;


    rowvec A;
    A.set_size(3);
    A(0) = 1;
    rowvec B;
    B.set_size(3);
    B(0) = 1.5;
    cout << fgas.absdiff2(A,B) << endl;

    //cout << fgas.v(0,2,0,2) << endl;
    //double ref_e = fgas.eref(65);
    //cout << "Energy:" << ref_e/65.0 << endl;
    //ccsolve solver(fgas);
    //solver.scan_amplitudes();
    //double e_corr = solver.CCSD_SG(14);
    return 0;

}

