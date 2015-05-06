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
    void advance();
    void advance_intermediates(); //advance using intermediates
    void L1_block_multiplication(); //block ladder calculation
    void L1_dense_multiplication(); //block ladder calculation

    void check_matrix_consistency();

    void energy();

    //The diagrams contributing to the CCD energy
    sp_mat L1, L2, L3, Q1, Q2, Q3, Q4;
    flexmat fmL3, fmQ1, fmQ2, fmQ3, fmQ4;

    flexmat fmI1, fmI2, fmI3, fmI4, fmI2temp, fmI3temp; //intermediates for CCD terms

    double CCSD_SG_energy();
    double correlation_energy;


};

#endif // CCD_H
