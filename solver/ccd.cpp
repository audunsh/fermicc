#include "ccd.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "solver/flexmat.h"
#include "basis/electrongas.h"
#include "solver/initializer.h"

using namespace std;
using namespace arma;

ccd::ccd(electrongas bs){
    ebs = bs;
    iSetup = initializer(bs);
    iSetup.sVhhhh();
    iSetup.sVpppp();
    iSetup.sVhpph();
    iSetup.sVhhpp();
    flexmat T(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNh);
    T.ab_ij().print();
    T.ba_ij().print();
}

