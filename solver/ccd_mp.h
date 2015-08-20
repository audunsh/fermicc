#ifndef CCD_MP_H
#define CCD_MP_H

#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "basis/electrongas.h"
#include "solver/flexmat.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"

using namespace std;
using namespace arma;


class ccd_mp
{
public:
    ccd_mp(electrongas bs, double a);
    double alpha;

    electrongas ebs;
    initializer iSetup;

    flexmat vhhhh;
    flexmat vpppp;
    flexmat vhhpp;
    flexmat vpphh;
    flexmat vhpph;
    flexmat T;
    flexmat Tprev; //for use with relaxation


    void advance();

    void solve(int iters, double relaxation, double threshold);
    double dEnergy;

    void L1_dense_multiplication(); //block ladder calculation

    void check_matrix_consistency();

    void energy();

    //The diagrams contributing to the CCD energy
    sp_mat L1, L2, L3, Q1, Q2, Q3, Q4;
    flexmat fmL3, fmQ1, fmQ2, fmQ3, fmQ4;

    flexmat fmI1, fmI2, fmI3, fmI4, fmI2temp, fmI3temp; //intermediates for CCD terms

    double CCSD_SG_energy();
    double correlation_energy;

    int iterations;


};


#endif // CCD_MP_H
