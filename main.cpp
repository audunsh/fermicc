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


#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Sparse>



using namespace std;
using namespace arma;


int main()
{
    electrongas fgas;
    fgas.generate_state_list(10.0,1.0, 14);

    //cout << "Energy per particle:" << fgas.eref(14)/14.0 << " (a.u)"  << endl;
    cout << "# Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    //cout << "Energy per particle (analytic):" << 2*fgas.analytic_energy(14)/14.0 << " (rydberg)"  << endl;
    cout << "# G.Baardsens results:" << 1.9434 << endl;

    ccd solver(fgas);
    //arma::u32 sz = 600000000;
    //Eigen::VectorXcd vVppp(600000000);
    //vec vVppp = zeros(sz);
    //vec vVppp2(sz);
    //
    //double * aux_mem = new double[sz];
    //vec vVppp(aux_mem, sz, false, true);

    /*
    cout << vVppp.size() << endl;
    int e = 0;
    for(int i = 0; i<240000000; ++i){
        e += 1;
    }
    //delete aux_mem;
    */

    //cout << e << endl;

    //ccsolve solver2(fgas);
    //solver2.CCSD_SG(14);


    return 0;

}

