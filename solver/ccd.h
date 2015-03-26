#ifndef CCD_H
#define CCD_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "basis/electrongas.h"
#include "solver/flexmat.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"

using namespace std;
using namespace arma;


class ccd
{
public:
    ccd(electrongas bs);


    electrongas ebs;
    initializer iSetup;

    //flexmat Vhh();
    flexmat vhhhh;
    flexmat vpppp;
    flexmat vhhpp;
    flexmat vpphh;
    flexmat vhpph;
    flexmat T;
    //flexmat Vpppp();
    //flexmat Vhhpp();

    //flexmat Vpphh();
    //flexmat Vhpph();
    //flexmat T();

    void energy();

    //sp_mat spE;

};

#endif // CCD_H
