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
    fgas.generate_state_list(5.0,1.0, 14);

    //cout << "Energy per particle:" << fgas.eref(14)/14.0 << " (a.u)"  << endl;
    cout << "# Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    //cout << "Energy per particle (analytic):" << 2*fgas.analytic_energy(14)/14.0 << " (rydberg)"  << endl;
    cout << "# G.Baardsens results:" << 1.9434 << endl;

    ccd solver(fgas);



    //ccsolve solver2(fgas);


    //solver2.CCSD_SG(14);

    /*
    //compare amplitudes
    int Np = solver.iSetup.iNp;
    int Nh = solver.iSetup.iNh;
    int Ns = Np+Nh;
    int a,b,i,j;
    double d1, d2;
    for(a = 0; a<Np; a++){
        for(b = 0; b<Np; b++){
            for(i = 0; i<Nh; i++){
                for(j=0;j<Nh;j++){
                    d1 = solver.vhpph.pq_rs()(i+a*Nh, b+j*Np);
                    d2 = solver2.v(i,a+Nh,b+Nh,j);
                    //cout << a << " " << b << " " <<i << " " << j << " " << d1 << " " << d2 << endl;
                    if(d1!=d2){
                        cout << a << " " << b << " " <<i << " " << j << " " << d1 << " " << d2 << endl;
                    }
                }
            }
        }
    }
    */


    return 0;

}

