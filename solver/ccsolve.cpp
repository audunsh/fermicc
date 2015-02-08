#include <iomanip>
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"
#include "solver/ccsolve.h"
//#include "solver/t3amps.h"


using namespace std;
using namespace arma;

ccsolve::ccsolve()
{
}

ccsolve::ccsolve(electrongas f)
{
    eBasis = f;
    iNs = eBasis.iNbstates;
    iNs2 = iNs*iNs;
    initialize_amplitudes();
}

double ccsolve::v(int a, int b, int i, int j){
    return eBasis.v(a,b,i,j);
}

double ccsolve::h(int a, int i){
    return eBasis.h(a,i);
}

void ccsolve::initialize_amplitudes(){
    //initializing amplitude tensors
    t1a.set_size(iNs,iNs);
    t2a.set_size(iNs2,iNs2);
    t3a.set_size(iNs2*iNs, iNs2*iNs);
}

void ccsolve::update_intermediates(){}

double ccsolve::t1(int a, int i){
    return t1a(a,i);
}

double ccsolve::t2(int a, int b, int i, int j){
    return t2a(a + iNs*b, i + iNs*j);
}

double ccsolve::t3(int a, int b, int c, int i, int j, int k){
    return t3a(a + iNs*b + iNs2*c,i + iNs*j + iNs2*k);
}
