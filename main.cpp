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




using namespace std;
using namespace arma;


int main()
{
    electrongas fgas;
    fgas.generate_state_list2(5.0,1.0, 14);

    //cout << "Energy per particle:" << fgas.eref(14)/14.0 << " (a.u)"  << endl;
    //cout << "[Main] Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    cout << "[Main]" << setprecision(15) << "# Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;

    cout << "[Main] G.Baardsens results:" << 1.9434 << endl;

    //ccsolve solver2(fgas);
    //solver2.CCSD_SG(2);
    ccd_pt solver(fgas, .5);

    //fgas.print();
    //cout << pow(2, 2.0/3.0) << endl;



    /*
    double val = 0;
    int count = 0;
    int Ns = fgas.iNbstates;
    int Nh = fgas.iNparticles;
    for(int a = Nh; a < Ns; ++a){
        for(int i = 0; i < Nh; ++i){
            for(int j = 0; j < Nh; ++j){
                for(int k = 0; k < Nh; ++k){
                    if(fgas.v2(i,a,j,k) != 0){
                        //cout << fgas.v2(i,a,b,c) << endl;
                        count += 1;
                    }

                }
            }
        }
    }
    cout << "Number of nonzero entries: " << count << endl;
    */




    /*

    double val = 0;
    int Ns = fgas.iNbstates;
    int Nh = fgas.iNparticles;
    for(int a = Nh; a < Ns; ++a){
        for(int b = Nh; b < Ns; ++b){
            for(int i = 0; i < Nh; ++i){
                for(int j = 0; j < Nh; ++j){
                    val += .25*fgas.v2(a,b,i,j)*fgas.v2(i,j,a,b)/(fgas.vEnergy(i) +fgas.vEnergy(j) - fgas.vEnergy(a) - fgas.vEnergy(b));


                }
            }
        }
    }
    cout << val/Nh << endl;
    */


    return 0;

}

