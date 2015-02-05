#include <iomanip>
#include <armadillo>
#include "basis/electrongas.h"
#include "solver/ccsolve.h"

using namespace std;
using namespace arma;

ccsolve::ccsolve()
{
}

ccsolve::ccsolve(electrongas f)
{
    eBasis = f;
    iNs = eBasis.iNbstates;
    initialize_amplitudes();


}

double ccsolve::v(int a, int b, int i, int j){}

double ccsolve::h(int a, int i){}

void ccsolve::initialize_amplitudes(){
    md_t1amps.set_size(iNs,iNs);
    fm_t2amps.set_size(iNs,iNs);
    //for(int i=0; i<iNs+1; i++){
    //    fm_t2amps.set_size(iNs,iNs);
    //}
    cout << md_t1amps.size() << endl;
    //cout << fm_t2amps << endl;

}

void ccsolve::update_intermediates(){}

double ccsolve::t1(int a, int i){}

double ccsolve::t2(int a, int b, int i, int j){}

double ccsolve::t3(int a, int b, int c, int i, int j, int k){}





